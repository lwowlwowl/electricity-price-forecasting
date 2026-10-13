#!/usr/bin/env python
"""用正式Global三模型验证连续MPC零阶收益信号能否反传。"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.continuous_zo import (  # noqa: E402
    ContinuousMPCObjective,
    continuous_mpc_proxy_loss,
    estimate_continuous_mpc_gradients,
)
from decision_aware.contracts import FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    flatten_rt_delivery_day_batch,
    load_formal_entity_mappings,
)
from decision_aware.global_joint import build_global_joint_datasets  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402


DEFAULT_CHECKPOINTS = {
    "da": "data/checkpoints/global_ercot_huber_f1_stable_v2/da_inner_seed0/candidate_epoch_010.pt",
    "rt_at_da": "data/checkpoints/global_ercot_huber_f0_stable_v2/rt_at_da_inner_seed0/candidate_epoch_010.pt",
    "rt": "data/checkpoints/global_ercot_huber_f1_stable_v2/rt_inner_seed0/candidate_epoch_008.pt",
}
MODEL_CLASSES = {
    "da": DecisionAwareDAForecaster,
    "rt_at_da": DecisionAwareRTAtDAForecaster,
    "rt": DecisionAwareRTForecaster,
}
PREDICTION_KEYS = {"da": "p_da", "rt_at_da": "p_rt_at_da", "rt": "p_rt"}


def _config(checkpoint: Mapping) -> PilotConfig:
    allowed = {item.name for item in fields(PilotConfig)}
    return PilotConfig(**{
        key: value for key, value in checkpoint["config"].items() if key in allowed
    })


def _nested_equal(left, right) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    if (
        isinstance(left, Sequence) and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes)) and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_device(batch, device):
    if isinstance(batch, Mapping):
        return {key: _to_device(value, device) for key, value in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def _parameter_gradient_norm(model: torch.nn.Module) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in model.parameters() if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="LZ_LCRA")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--directions", type=int, default=2)
    parser.add_argument("--epsilon-spread", type=float, default=10.0)
    parser.add_argument("--epsilon-rt", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", default="data/results/global_joint_continuous_zo_smoke.json"
    )
    for task, default in DEFAULT_CHECKPOINTS.items():
        parser.add_argument(f"--{task.replace('_', '-')}-checkpoint", default=default)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.days < 1 or args.directions < 1:
        raise ValueError("days和directions必须至少为1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    checkpoint_paths = {
        "da": Path(args.da_checkpoint),
        "rt_at_da": Path(args.rt_at_da_checkpoint),
        "rt": Path(args.rt_checkpoint),
    }
    checkpoints = {
        task: torch.load(path, map_location="cpu", weights_only=False)
        for task, path in checkpoint_paths.items()
    }
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(["ERCOT"])
    configs = {}
    for task, checkpoint in checkpoints.items():
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(checkpoint)
        mappings.validate_checkpoint(checkpoint)
        if checkpoint.get("task") != task or checkpoint.get("phase") != "inner":
            raise ValueError(f"{task} checkpoint的任务或阶段不匹配")
        configs[task] = _config(checkpoint)
        FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(configs[task])

    entity = ("ERCOT", args.node)
    mappings.identity(*entity)
    train_dataset, validation_dataset = build_global_joint_datasets(
        configs, [entity], mappings
    )
    for task in ("da", "rt_at_da", "rt"):
        actual_stats = train_dataset.norm_stats_by_task_by_node[task][args.node]
        expected_stats = checkpoints[task]["norm_stats_by_node"][args.node]
        if not _nested_equal(actual_stats, expected_stats):
            raise ValueError(f"{task}逐节点scaler与checkpoint不一致")

    count = min(args.days, validation_dataset.entity_lengths[0])
    indices = list(range(validation_dataset.offsets[0], validation_dataset.offsets[0] + count))
    raw = next(iter(DataLoader(Subset(validation_dataset, indices), batch_size=count)))
    da_batch = _to_device(raw["da"], device)
    rt_da_batch = _to_device(raw["rt_at_da"], device)
    rt_batch = _to_device(flatten_rt_delivery_day_batch(raw["rt"]), device)

    models = {}
    outputs = {}
    for task, model_class in MODEL_CLASSES.items():
        model = model_class(configs[task]).to(device)
        model.load_state_dict(checkpoints[task]["model_state"], strict=True)
        model.eval()
        models[task] = model
        branch = {"da": da_batch, "rt_at_da": rt_da_batch, "rt": rt_batch}[task]
        outputs[task] = model(branch)[PREDICTION_KEYS[task]]
    p_da = outputs["da"]
    p_rt_at_da = outputs["rt_at_da"]
    p_rt = outputs["rt"].reshape(count, 24, -1)

    timestamps = tuple(validation_dataset.local_timestamps(index) for index in indices)
    objective = ContinuousMPCObjective(
        realized_da=raw["da"]["price_da_tgt"],
        realized_rt=raw["da"]["price_rt_tgt"],
        local_timestamps=timestamps,
        cfg=configs["da"],
    )
    baseline_revenue = objective(p_da, p_rt_at_da, p_rt)
    grad_spread, grad_rt, diagnostics = estimate_continuous_mpc_gradients(
        p_da,
        p_rt_at_da,
        p_rt,
        objective,
        epsilon_spread=args.epsilon_spread,
        epsilon_rt=args.epsilon_rt,
        directions_spread=args.directions,
        directions_rt=args.directions,
        direction_mode="orthogonal",
        feedback_mode="episode_scalar",
    )
    for model in models.values():
        model.zero_grad(set_to_none=True)
    proxy = continuous_mpc_proxy_loss(
        p_da, p_rt_at_da, p_rt, grad_spread, grad_rt,
        reduction="global_sum",
    )
    proxy.backward()

    parameter_ids = {
        task: {id(parameter) for parameter in model.parameters()}
        for task, model in models.items()
    }
    overlaps = {
        f"{left}__{right}": len(parameter_ids[left] & parameter_ids[right])
        for left, right in (("da", "rt_at_da"), ("da", "rt"), ("rt_at_da", "rt"))
    }
    parameter_gradient_norms = {
        task: _parameter_gradient_norm(model) for task, model in models.items()
    }
    passed = (
        all(diagnostics[name]["finite"] for name in ("spread", "rt"))
        and diagnostics["spread"]["gradient_norm"] > 0.0
        and diagnostics["rt"]["gradient_norm"] > 0.0
        and all(value > 0.0 and np.isfinite(value) for value in parameter_gradient_norms.values())
        and all(value == 0 for value in overlaps.values())
    )
    report = {
        "status": "passed" if passed else "failed",
        "purpose": "global three-model continuous penalty-aware MPC zero-order gradient smoke",
        "not_a_model_selection_result": True,
        "node": args.node,
        "days": count,
        "dates": [str(validation_dataset.sample_identity(index)["delivery_date"]) for index in indices],
        "device": str(device),
        "seed": args.seed,
        "contract": {
            "three_independent_models": True,
            "separate_parameter_sets": overlaps,
            "decisions": "forecast_only",
            "settlement": "realized prices after action",
            "deviation_penalty": True,
            "degradation_cost_usd_per_mwh": float(configs["da"].bess_kappa),
            "soc_reset": "per node or true time gap",
        },
        "checkpoints": {
            task: {"path": path.as_posix(), "sha256": _sha256(path)}
            for task, path in checkpoint_paths.items()
        },
        "baseline_daily_revenue": baseline_revenue.detach().cpu().tolist(),
        "baseline_mean_daily_revenue": float(baseline_revenue.mean().cpu()),
        "epsilon": {"spread": args.epsilon_spread, "rt": args.epsilon_rt},
        "directions": args.directions,
        "diagnostics": diagnostics,
        "parameter_gradient_norms": parameter_gradient_norms,
        "proxy_loss": float(proxy.detach().cpu()),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "status": report["status"],
        "output": output.relative_to(ROOT).as_posix(),
        "baseline_mean_daily_revenue": report["baseline_mean_daily_revenue"],
        "gradient_norms": {
            "spread": diagnostics["spread"]["gradient_norm"],
            "rt": diagnostics["rt"]["gradient_norm"],
        },
        "parameter_gradient_norms": parameter_gradient_norms,
    }, ensure_ascii=False), flush=True)
    if not passed:
        raise RuntimeError("连续MPC零阶信号烟雾试验未通过；详见输出JSON")


if __name__ == "__main__":
    main()
