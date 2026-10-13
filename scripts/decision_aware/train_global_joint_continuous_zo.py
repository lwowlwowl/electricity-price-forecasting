#!/usr/bin/env python
"""在2020--2022多节点数据上，用连续MPC零阶收益信号微调三个独立Global模型。

这是2023内部开发实验；不会读取2024或2025。每个batch严格限制在同一节点的连续交付日。
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


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
from decision_aware.global_joint import (  # noqa: E402
    NodeBalancedEpisodeBatchSampler,
    build_global_joint_datasets,
)
from decision_aware.joint_prediction import spread_auxiliary_losses  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402


TASKS = ("da", "rt_at_da", "rt")
MODEL_CLASSES = {
    "da": DecisionAwareDAForecaster,
    "rt_at_da": DecisionAwareRTAtDAForecaster,
    "rt": DecisionAwareRTForecaster,
}
DEFAULTS = {
    "da": "data/checkpoints/global_ercot_huber_f1_stable_v2/da_inner_seed0/candidate_epoch_010.pt",
    "rt_at_da": "data/checkpoints/global_ercot_huber_f0_stable_v2/rt_at_da_inner_seed0/candidate_epoch_010.pt",
    "rt": "data/checkpoints/global_ercot_huber_f1_stable_v2/rt_inner_seed0/candidate_epoch_008.pt",
}


def _config(checkpoint: dict) -> PilotConfig:
    allowed = {item.name for item in fields(PilotConfig)}
    return PilotConfig(**{
        key: value for key, value in checkpoint["config"].items() if key in allowed
    })


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _move(value, device):
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def _forward(models, batch, device, amp_enabled):
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=amp_enabled
    ):
        p_da = models["da"](batch["da"])["p_da"].float()
        p_rt_at_da = models["rt_at_da"](batch["rt_at_da"])["p_rt_at_da"].float()
        rt_flat = flatten_rt_delivery_day_batch(batch["rt"])
        p_rt = models["rt"](rt_flat)["p_rt"].float()
    return p_da, p_rt_at_da, p_rt.reshape(p_da.shape[0], 24, -1)


def _prediction_anchor(
    outputs,
    batch,
    scale: float,
    *,
    spread_weight: float = 0.0,
    direction_weight: float = 0.0,
    direction_threshold: float = 6.0,
    direction_temperature: float = 25.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = (
        batch["da"]["price_da_tgt"].float(),
        batch["rt_at_da"]["price_rt_at_da_tgt"].float(),
        batch["rt"]["price_rt_tgt"].float(),
    )
    base = sum(
        F.smooth_l1_loss(prediction, target, beta=25.0) / scale
        for prediction, target in zip(outputs, targets)
    ) / 3.0
    auxiliaries = spread_auxiliary_losses(
        outputs[0], outputs[1], targets[0], targets[1],
        batch["da"]["price_da_std"],
        batch["rt_at_da"]["price_rt_std"],
        direction_threshold=direction_threshold,
        direction_temperature=direction_temperature,
    )
    total = (
        base
        + spread_weight * auxiliaries["spread_magnitude"]
        + direction_weight * auxiliaries["spread_direction"]
    )
    return total, {"base_prediction": base, **auxiliaries}


def _set_trainable_scope(models, scope: str) -> list[torch.nn.Parameter]:
    if scope not in {"price_head", "decoder_head", "all"}:
        raise ValueError("scope必须是price_head、decoder_head或all")
    parameters = []
    for model in models.values():
        for name, parameter in model.named_parameters():
            trainable = (
                scope == "all"
                or name.startswith("price_head.")
                or (scope == "decoder_head" and name.startswith("decoder."))
            )
            parameter.requires_grad_(trainable)
            if trainable:
                parameters.append(parameter)
    if not parameters:
        raise ValueError("没有可训练参数")
    return parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    for task in TASKS:
        parser.add_argument(
            f"--{task.replace('_', '-')}-checkpoint", default=DEFAULTS[task]
        )
    parser.add_argument("--family", default="globalzo_b001")
    parser.add_argument("--output-dir", default="data/checkpoints/global_joint_continuous_zo_v1")
    parser.add_argument("--result", default="data/results/global_joint_continuous_zo_train_v1.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--episode-days", type=int, default=2)
    parser.add_argument("--episodes-per-node", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=15)
    parser.add_argument("--directions", type=int, default=1)
    parser.add_argument("--epsilon-spread", type=float, default=10.0)
    parser.add_argument("--epsilon-rt", type=float, default=15.0)
    parser.add_argument("--profit-weight", type=float, default=0.001)
    parser.add_argument("--proxy-scale", type=float, default=1000.0)
    parser.add_argument("--prediction-scale", type=float, default=25.0)
    parser.add_argument("--spread-weight", type=float, default=0.0)
    parser.add_argument("--direction-weight", type=float, default=0.0)
    parser.add_argument("--direction-threshold", type=float, default=6.0)
    parser.add_argument("--direction-temperature", type=float, default=25.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--scope", default="price_head")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.episode_days < 1 or args.episodes_per_node < 1:
        raise ValueError("epochs、episode-days和episodes-per-node必须为正")
    if not 0.0 < args.profit_weight <= 1.0:
        raise ValueError("profit-weight必须位于(0,1]")
    if args.spread_weight < 0 or args.direction_weight < 0:
        raise ValueError("spread-weight和direction-weight不能为负")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    amp_enabled = device.type == "cuda"
    paths = {
        task: ROOT / getattr(args, f"{task}_checkpoint") for task in TASKS
    }
    sources = {
        task: torch.load(path, map_location="cpu", weights_only=False)
        for task, path in paths.items()
    }
    configs = {task: _config(source) for task, source in sources.items()}
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(["ERCOT"])
    for task in TASKS:
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(sources[task])
        mappings.validate_checkpoint(sources[task])
        if sources[task].get("task") != task or sources[task].get("phase") != "inner":
            raise ValueError(f"{task}源checkpoint必须来自inner阶段")
    if not 1 <= args.max_nodes <= len(mappings.node_to_id):
        raise ValueError("max-nodes超出ERCOT正式节点数量")
    entities = [("ERCOT", node) for node in list(mappings.node_to_id)[:args.max_nodes]]
    train_dataset, _ = build_global_joint_datasets(
        configs, entities, mappings,
        train_stage="inner_train", validation_stage="inner_validation",
    )
    sampler = NodeBalancedEpisodeBatchSampler(
        train_dataset,
        episode_days=args.episode_days,
        episodes_per_node=args.episodes_per_node,
        seed=args.seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    models = {task: MODEL_CLASSES[task](configs[task]).to(device) for task in TASKS}
    for task in TASKS:
        models[task].load_state_dict(sources[task]["model_state"], strict=True)
        models[task].eval()
    parameters = _set_trainable_scope(models, args.scope)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    history = []
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        rows = []
        for batch_index, raw in enumerate(loader, start=1):
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward(models, batch, device, amp_enabled)
            p_da, p_rt_at_da, p_rt = outputs
            # batch_sampler guarantees one node and consecutive days. Timestamps are
            # recovered from the sampler's current yielded indices through ordinals.
            ordinals = raw["delivery_date_ordinal"].tolist()
            identity_node = int(raw["node_id"][0])
            candidate_indices = [
                index for index in range(len(train_dataset))
                if train_dataset.sample_identity(index)["node_id"] == identity_node
                and train_dataset.sample_identity(index)["delivery_date"].toordinal() in ordinals
            ]
            candidate_indices.sort(
                key=lambda index: train_dataset.sample_identity(index)["delivery_date"]
            )
            timestamps = tuple(train_dataset.local_timestamps(index) for index in candidate_indices)
            objective = ContinuousMPCObjective(
                realized_da=batch["da"]["price_da_tgt"].float(),
                realized_rt=batch["da"]["price_rt_tgt"].float(),
                local_timestamps=timestamps,
                cfg=configs["da"],
            )
            with torch.no_grad():
                base_revenue = objective(p_da, p_rt_at_da, p_rt)
                grad_spread, grad_rt, diagnostics = estimate_continuous_mpc_gradients(
                    p_da, p_rt_at_da, p_rt, objective,
                    epsilon_spread=args.epsilon_spread,
                    epsilon_rt=args.epsilon_rt,
                    directions_spread=args.directions,
                    directions_rt=args.directions,
                    pred_clamp=tuple(configs["da"].pred_clamp),
                    direction_mode="orthogonal",
                    feedback_mode="episode_scalar",
                )
            proxy = continuous_mpc_proxy_loss(
                p_da, p_rt_at_da, p_rt, grad_spread, grad_rt,
                scale=args.proxy_scale, reduction="global_sum",
            )
            prediction, prediction_parts = _prediction_anchor(
                outputs,
                batch,
                args.prediction_scale,
                spread_weight=args.spread_weight,
                direction_weight=args.direction_weight,
                direction_threshold=args.direction_threshold,
                direction_temperature=args.direction_temperature,
            )
            loss = (1.0 - args.profit_weight) * prediction + args.profit_weight * proxy
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            optimizer.step()
            rows.append({
                "batch": batch_index,
                "node_id": identity_node,
                "dates": [str(train_dataset.sample_identity(index)["delivery_date"]) for index in candidate_indices],
                "base_mean_revenue": float(base_revenue.mean().cpu()),
                "prediction_loss": float(prediction.detach().cpu()),
                "base_prediction_loss": float(
                    prediction_parts["base_prediction"].detach().cpu()
                ),
                "spread_magnitude_loss": float(
                    prediction_parts["spread_magnitude"].detach().cpu()
                ),
                "spread_direction_loss": float(
                    prediction_parts["spread_direction"].detach().cpu()
                ),
                "important_spread_fraction": float(
                    prediction_parts["important_spread_fraction"].detach().cpu()
                ),
                "proxy_loss": float(proxy.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
                "gradient_norm_before_clip": gradient_norm,
                "spread_gradient_norm": diagnostics["spread"]["gradient_norm"],
                "rt_gradient_norm": diagnostics["rt"]["gradient_norm"],
            })
            print(json.dumps({"event": "batch", "epoch": epoch, **rows[-1]}, ensure_ascii=False), flush=True)
        history.append({
            "epoch": epoch,
            "batches": len(rows),
            "mean_base_revenue": float(np.mean([row["base_mean_revenue"] for row in rows])),
            "mean_prediction_loss": float(np.mean([row["prediction_loss"] for row in rows])),
            "mean_proxy_loss": float(np.mean([row["proxy_loss"] for row in rows])),
            "rows": rows,
        })

    output_dir = ROOT / args.output_dir / args.family
    output_dir.mkdir(parents=True, exist_ok=True)
    families = {args.family: {}}
    source_contract = {
        task: {"path": paths[task].relative_to(ROOT).as_posix(), "sha256": _sha256(paths[task])}
        for task in TASKS
    }
    for task in TASKS:
        payload = dict(sources[task])
        for key in ("optimizer_state", "scheduler_state", "grad_scaler_state"):
            payload.pop(key, None)
        payload["model_state"] = models[task].state_dict()
        payload["decision_aware_finetune"] = {
            "method": "continuous_mpc_two_sided_zero_order_hybrid_loss",
            "source_checkpoints": source_contract,
            "profit_weight": args.profit_weight,
            "proxy_scale": args.proxy_scale,
            "episode_days": args.episode_days,
            "episodes_per_node": args.episodes_per_node,
            "directions": args.directions,
            "training_years": "2020-2022",
            "2024_used": False,
            "2025_used": False,
        }
        path = output_dir / f"{task}.pt"
        torch.save(payload, path)
        families[args.family][task] = path.relative_to(ROOT).as_posix()
    manifest = output_dir / "candidate_manifest.json"
    manifest.write_text(
        json.dumps({"families": families, "phase": "inner"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "experiment": "global_joint_continuous_zo_inner_development",
        "status": "trained_not_yet_selected",
        "family": args.family,
        "device": str(device),
        "source_checkpoints": source_contract,
        "parameters": vars(args),
        "history": history,
        "runtime_seconds": perf_counter() - started,
        "candidate_manifest": manifest.relative_to(ROOT).as_posix(),
        "2024_used": False,
        "2025_used": False,
    }
    result = ROOT / args.result
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "result": result.relative_to(ROOT).as_posix(), "manifest": manifest.relative_to(ROOT).as_posix()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
