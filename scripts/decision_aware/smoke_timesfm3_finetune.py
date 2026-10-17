#!/usr/bin/env python
"""TimesFM-3 parameter-efficient fine-tuning smoke test.

This is deliberately a *trainability* check rather than a paper experiment.
The public TimesFM-3 forecaster currently exposes an inference-only ``decode``
method, while the underlying PyTorch module remains differentiable.  The smoke
test therefore calls the undecorated implementation, freezes the complete
backbone, and adds a small LoRA update to the output head.  It verifies:

* gradients reach the adapter while the pretrained backbone stays frozen;
* a tiny real electricity-price batch can be overfit for 24/24/4-hour tasks;
* adapter state can be saved, loaded, and reproduce predictions exactly.

Passing this script does not by itself authorize a formal fine-tuning result.
The private/experimental training path and the final loss/adapter protocol must
still be frozen before any four-fold comparison.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models" / "workers"))


def _configure_timesfm_imports() -> None:
    """Reuse the vendored TimesFM source and its pure-Python dependencies.

    The project environment owns CUDA PyTorch and the data stack.  The isolated
    TimesFM worker environment owns the upstream source dependencies.  Appending
    (rather than prepending) its site-packages keeps the already-imported project
    PyTorch build authoritative.
    """
    source = ROOT / "external" / "timesfm" / "src"
    site_packages = (
        ROOT / "external" / "timesfm" / ".venv" / "Lib" / "site-packages"
    )
    if not source.exists() or not site_packages.exists():
        raise FileNotFoundError("local TimesFM source or worker environment is missing")
    sys.path.insert(0, str(source))
    if str(site_packages) not in sys.path:
        sys.path.append(str(site_packages))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    RTDeliveryDayDataset,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from runtime_env import configure_runtime_environment  # noqa: E402


MODEL_ID = "google/timesfm-3.0-pytorch"
TASKS = ("da", "rt_at_da", "rt")
HORIZONS = {"da": 24, "rt_at_da": 24, "rt": 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.scale = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.lora_up(self.lora_down(values)) * self.scale

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "lora_down.weight": self.lora_down.weight.detach().cpu().clone(),
            "lora_up.weight": self.lora_up.weight.detach().cpu().clone(),
        }

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = {"lora_down.weight", "lora_up.weight"}
        if set(state) != expected:
            raise ValueError(f"adapter keys mismatch: {sorted(state)}")
        self.lora_down.weight.data.copy_(state["lora_down.weight"])
        self.lora_up.weight.data.copy_(state["lora_up.weight"])


@dataclass(frozen=True)
class SmokeBatch:
    context: torch.Tensor
    target: torch.Tensor
    past_future_covariates: torch.Tensor
    market: str
    node: str


def _raw_window(dataset: Any, task: str, local_index: int) -> tuple[np.ndarray, ...]:
    if task == "rt":
        if not isinstance(dataset, RTDeliveryDayDataset):
            raise TypeError("rolling RT smoke expects RTDeliveryDayDataset")
        hourly = dataset.hourly_dataset
        _, indices = dataset.days[local_index]
        window = hourly.windows[indices[0]]
        context_end = int(window.issue_position)
        price = hourly.price_rt
        calendar = hourly.streams["cal"]
        target_positions = np.asarray(window.target_positions, dtype=np.int64)
        context_start = int(window.context_start)
        hist_cov = np.column_stack([
            calendar[context_start:context_end],
            hourly.price_da[context_start:context_end],
        ])
        future_cov = np.column_stack([
            calendar[target_positions],
            hourly.price_da[target_positions],
        ])
    else:
        hourly = dataset
        window = hourly.windows[local_index]
        context_start = int(window.context_start)
        context_end = int(window.context_end)
        target_positions = np.asarray(window.target_positions, dtype=np.int64)
        price = hourly.price_da if task == "da" else hourly.price_rt
        calendar = hourly.streams["cal"]
        hist_cov = calendar[context_start:context_end]
        future_cov = calendar[target_positions]
    context = price[context_start:context_end]
    target = price[target_positions]
    if context.shape != (168,):
        raise ValueError(f"{task} context shape is {context.shape}, expected (168,)")
    if target.shape != (HORIZONS[task],):
        raise ValueError(
            f"{task} target shape is {target.shape}, expected {(HORIZONS[task],)}"
        )
    covariates = np.concatenate([hist_cov, future_cov], axis=0).T
    arrays = (context, target, covariates)
    if not all(np.isfinite(item).all() for item in arrays):
        raise ValueError(f"{task} smoke batch contains non-finite values")
    return tuple(np.asarray(item, dtype=np.float32) for item in arrays)


def build_smoke_batch(
    cfg: PilotConfig,
    task: str,
    source_market: str,
    sample_index: int,
) -> SmokeBatch:
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    nodes = sorted(
        node for node, market in mappings.node_to_market.items()
        if market == source_market
    )
    if not nodes:
        raise ValueError(f"source market {source_market} has no configured nodes")
    node = nodes[0]
    train, _ = build_global_task_datasets(
        cfg,
        task,
        [(source_market, node)],
        mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
    )
    dataset = train.datasets[0]
    local_index = min(max(int(sample_index), 0), len(dataset) - 1)
    context, target, covariates = _raw_window(dataset, task, local_index)
    return SmokeBatch(
        context=torch.from_numpy(context).reshape(1, 1, -1),
        target=torch.from_numpy(target).reshape(1, 1, -1),
        past_future_covariates=torch.from_numpy(covariates).unsqueeze(0),
        market=source_market,
        node=node,
    )


def _decode_with_grad(
    model: nn.Module,
    batch: SmokeBatch,
    horizon: int,
) -> torch.Tensor:
    """Call the original implementation beneath TimesFM-3's no_grad wrapper."""
    from timesfm3.torch.model import TimesFM3Torch

    decode = getattr(TimesFM3Torch.decode, "__wrapped__", None)
    if decode is None:
        raise RuntimeError(
            "TimesFM-3 decode no longer exposes __wrapped__; upstream API changed"
        )
    logits = decode(
        model,
        batch.context,
        horizon=horizon,
        past_future_covariates=batch.past_future_covariates,
    )
    if logits.shape[:3] != (1, 1 + batch.past_future_covariates.shape[1], horizon):
        raise ValueError(f"unexpected TimesFM-3 logits shape: {tuple(logits.shape)}")
    median_index = len(model.quantiles) // 2
    return logits[:, :1, :, median_index]


def _adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_down" in name or "lora_up" in name
    ]
    if not parameters:
        raise ValueError("no LoRA adapter parameters found")
    return parameters


def run_task(
    evaluator: Any,
    base_head_state: dict[str, torch.Tensor],
    batch: SmokeBatch,
    task: str,
    output_dir: Path,
    *,
    rank: int,
    alpha: float,
    steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = evaluator.model
    batch = SmokeBatch(
        context=batch.context.to(device),
        target=batch.target.to(device),
        past_future_covariates=batch.past_future_covariates.to(device),
        market=batch.market,
        node=batch.node,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output_features, input_features = base_head_state["weight"].shape
    base_head = nn.Linear(
        input_features,
        output_features,
        bias="bias" in base_head_state,
    ).to(device)
    base_head.load_state_dict(deepcopy(base_head_state))
    model.output_head = LoRALinear(base_head, rank=rank, alpha=alpha).to(device)
    model.train()
    adapter_parameters = _adapter_parameters(model)
    optimizer = torch.optim.AdamW(adapter_parameters, lr=learning_rate)
    backbone_snapshot = model.pre_transformer_resblock.hidden_layer.weight.detach().clone()

    with torch.no_grad():
        initial_prediction = _decode_with_grad(model, batch, HORIZONS[task])
        initial_loss = torch.nn.functional.smooth_l1_loss(
            initial_prediction, batch.target
        ).item()
    losses: list[float] = []
    gradient_norms: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = _decode_with_grad(model, batch, HORIZONS[task])
        loss = torch.nn.functional.smooth_l1_loss(prediction, batch.target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{task} loss became non-finite")
        loss.backward()
        gradient_norm = torch.sqrt(sum(
            parameter.grad.detach().float().square().sum()
            for parameter in adapter_parameters
            if parameter.grad is not None
        ))
        if not torch.isfinite(gradient_norm) or gradient_norm.item() <= 0:
            gradient_diagnostics = [
                {
                    "shape": list(parameter.shape),
                    "has_grad": parameter.grad is not None,
                    "finite": bool(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                    ),
                    "nan_count": int(
                        torch.isnan(parameter.grad).sum()
                        if parameter.grad is not None else 0
                    ),
                }
                for parameter in adapter_parameters
            ]
            raise FloatingPointError(
                f"{task} adapter gradient is invalid: {float(gradient_norm)}; "
                f"{gradient_diagnostics}"
            )
        torch.nn.utils.clip_grad_norm_(adapter_parameters, max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        gradient_norms.append(float(gradient_norm.detach()))

    model.eval()
    with torch.no_grad():
        final_prediction = _decode_with_grad(model, batch, HORIZONS[task])
        final_loss = torch.nn.functional.smooth_l1_loss(
            final_prediction, batch.target
        ).item()
    if not torch.equal(
        backbone_snapshot,
        model.pre_transformer_resblock.hidden_layer.weight.detach(),
    ):
        raise AssertionError(f"{task} frozen backbone changed")
    if not final_loss < initial_loss:
        raise AssertionError(
            f"{task} tiny-batch loss did not decrease: {initial_loss} -> {final_loss}"
        )

    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = task_dir / "adapter.pt"
    lora_head = model.output_head
    if not isinstance(lora_head, LoRALinear):
        raise TypeError("output head is not LoRALinear")
    payload = {
        "format": "timesfm3_output_head_lora_smoke_v1",
        "model_id": MODEL_ID,
        "task": task,
        "rank": rank,
        "alpha": alpha,
        "base_head_sha256": hashlib.sha256(
            b"".join(
                tensor.detach().cpu().numpy().tobytes()
                for tensor in base_head_state.values()
            )
        ).hexdigest(),
        "adapter": lora_head.adapter_state_dict(),
    }
    torch.save(payload, adapter_path)

    # Reload into a clean head and require bitwise-identical CPU predictions.
    clean_head = nn.Linear(
        base_head.in_features,
        base_head.out_features,
        bias=base_head.bias is not None,
    ).to(device)
    clean_head.load_state_dict(deepcopy(base_head_state))
    reloaded_head = LoRALinear(clean_head, rank=rank, alpha=alpha).to(device)
    saved = torch.load(adapter_path, map_location="cpu", weights_only=True)
    reloaded_head.load_adapter_state_dict(saved["adapter"])
    model.output_head = reloaded_head
    model.eval()
    with torch.no_grad():
        reloaded_prediction = _decode_with_grad(model, batch, HORIZONS[task])
    max_reload_difference = float(
        torch.max(torch.abs(final_prediction - reloaded_prediction))
    )
    if max_reload_difference != 0.0:
        raise AssertionError(
            f"{task} adapter reload changed predictions by {max_reload_difference}"
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in adapter_parameters)
    return {
        "task": task,
        "horizon": HORIZONS[task],
        "source_market": batch.market,
        "source_node": batch.node,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "relative_loss_reduction": (initial_loss - final_loss) / initial_loss,
        "first_step_loss": losses[0],
        "last_step_loss": losses[-1],
        "min_gradient_norm": min(gradient_norms),
        "max_gradient_norm": max(gradient_norms),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "backbone_unchanged": True,
        "reload_max_abs_difference": max_reload_difference,
        "adapter_path": adapter_path.relative_to(ROOT).as_posix(),
        "adapter_sha256": _sha256(adapter_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml",
    )
    parser.add_argument("--source-market", default="CAISO")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        default="data/results/timesfm3_finetune_smoke_v1",
    )
    args = parser.parse_args()
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not tasks or any(item not in TASKS for item in tasks):
        raise ValueError(f"tasks must be a subset of {TASKS}")
    if args.steps < 1:
        raise ValueError("steps must be positive")

    configure_runtime_environment(str(ROOT), "timesfm")
    _configure_timesfm_imports()
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    cfg_path = ROOT / args.config
    cfg = PilotConfig.from_yaml(cfg_path)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    evaluator = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=MODEL_ID,
            per_core_batch_size=1,
            device=str(device),
            cache_dir=os.environ.get("HF_HUB_CACHE"),
            local_files_only=True,
        )
    )
    # The iterative CPM/RevIN refinement is an inference-time fixed-point
    # correction.  Its backward pass is numerically undefined on constant
    # calendar variates, so it must be disabled while fitting an adapter.
    # Formal inference may re-enable it only after a dedicated parity check.
    evaluator.model.use_iterative_cpm_revin = False
    base_head_state = deepcopy(evaluator.model.output_head.state_dict())
    results = []
    for task in tasks:
        task_started = perf_counter()
        batch = build_smoke_batch(
            cfg, task, args.source_market, args.sample_index
        )
        result = run_task(
            evaluator,
            base_head_state,
            batch,
            task,
            output_dir,
            rank=args.rank,
            alpha=args.alpha,
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
        )
        result["wall_seconds"] = perf_counter() - task_started
        results.append(result)
        print(json.dumps({"event": "task_complete", **result}, ensure_ascii=False))

    summary = {
        "status": "passed",
        "scope": "trainability_smoke_only_not_paper_result",
        "model_id": MODEL_ID,
        "official_public_training_api": False,
        "training_path": "undecorated_pytorch_decode_plus_output_head_lora",
        "warning": (
            "TimesFM-3 public decode is inference-only; formal fine-tuning must "
            "freeze and disclose this experimental private training path."
        ),
        "config": cfg_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256(cfg_path),
        "source_market": args.source_market,
        "sample_index": args.sample_index,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "rank": args.rank,
        "alpha": args.alpha,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "tasks": results,
        "wall_seconds": perf_counter() - started,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
