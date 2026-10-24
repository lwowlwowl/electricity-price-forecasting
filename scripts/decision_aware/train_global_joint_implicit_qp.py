#!/usr/bin/env python
"""Fine-tune no-ID global forecasters through an implicit battery QP.

The default remains the source-only LOMO development protocol.  Final training
must be requested explicitly with ``--source-phase final --train-stage
final_train``; this uses the four-market 2020--2024 final checkpoints and never
loads the locked 2025 stage.
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
from decision_aware.contracts import (  # noqa: E402
    AnnualStage,
    FORMAL_EXPERIMENT_CONTRACT,
)
from decision_aware.global_data import (  # noqa: E402
    flatten_rt_delivery_day_batch,
    load_formal_entity_mappings,
)
from decision_aware.global_joint import (  # noqa: E402
    NodeBalancedEpisodeBatchSampler,
    build_global_joint_datasets,
)
from decision_aware.implicit_battery_qp import (  # noqa: E402
    BatteryQPConfig,
    ImplicitBatteryQPLayer,
    predicted_cost_vector,
    realized_two_settlement_revenue,
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


def _default_checkpoint(task: str, seed: int) -> str:
    if task == "rt_at_da":
        return (
            "data/checkpoints/universal_lomo_holdout_ercot_rt_at_da_v1/"
            f"rt_at_da_inner_seed{seed}/best.pt"
        )
    return (
        "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/"
        f"{task}_inner_seed{seed}/best.pt"
    )


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


def _prediction_anchor(outputs, batch, scale: float) -> torch.Tensor:
    targets = (
        batch["da"]["price_da_tgt"].float(),
        batch["rt_at_da"]["price_rt_at_da_tgt"].float(),
        batch["rt"]["price_rt_tgt"].float(),
    )
    return sum(
        F.smooth_l1_loss(prediction, target, beta=25.0) / scale
        for prediction, target in zip(outputs, targets)
    ) / 3.0


def _set_trainable_scope(models, scope: str) -> list[torch.nn.Parameter]:
    if scope not in {"price_head", "decoder_head", "all"}:
        raise ValueError("scope must be price_head, decoder_head, or all")
    parameters: list[torch.nn.Parameter] = []
    for model in models.values():
        for name, parameter in model.named_parameters():
            trainable = (
                scope == "all"
                or name.startswith("price_head.")
                or (
                    scope == "decoder_head"
                    and (name.startswith("decoder.") or name.startswith("price_head."))
                )
            )
            parameter.requires_grad_(trainable)
            if trainable:
                parameters.append(parameter)
    if not parameters:
        raise ValueError("no trainable parameters")
    return parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    for task in TASKS:
        parser.add_argument(f"--{task.replace('_', '-')}-checkpoint")
    parser.add_argument("--family", default="implicitqp_w020_seed0")
    parser.add_argument(
        "--output-dir", default="data/checkpoints/global_joint_implicit_qp_v1"
    )
    parser.add_argument(
        "--result", default="data/results/global_joint_implicit_qp_train_v1.json"
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--episode-days", type=int, default=1)
    parser.add_argument("--episodes-per-node", type=int, default=1)
    parser.add_argument("--max-source-nodes", type=int, default=0)
    parser.add_argument("--decision-weight", type=float, default=0.2)
    parser.add_argument("--revenue-scale", type=float, default=500.0)
    parser.add_argument("--prediction-scale", type=float, default=25.0)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scope", default="price_head")
    parser.add_argument("--quadratic-regularization", type=float, default=0.25)
    parser.add_argument("--barrier-temperature", type=float, default=0.25)
    parser.add_argument("--newton-steps", type=int, default=60)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--training-seed", type=int, default=20261023)
    parser.add_argument("--source-phase", choices=("inner", "final"), default="inner")
    parser.add_argument(
        "--train-stage",
        choices=(AnnualStage.INNER_TRAIN.value, AnnualStage.FINAL_TRAIN.value),
        default=AnnualStage.INNER_TRAIN.value,
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.episode_days < 1 or args.episodes_per_node < 1:
        raise ValueError("epochs and episode counts must be positive")
    if not 0.0 < args.decision_weight <= 1.0:
        raise ValueError("decision-weight must be in (0, 1]")
    if args.revenue_scale <= 0.0 or args.prediction_scale <= 0.0:
        raise ValueError("loss scales must be positive")

    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.training_seed)
    amp_enabled = device.type == "cuda"

    paths = {
        task: ROOT / (
            getattr(args, f"{task}_checkpoint") or _default_checkpoint(task, args.seed)
        )
        for task in TASKS
    }
    sources = {
        task: torch.load(path, map_location="cpu", weights_only=False)
        for task, path in paths.items()
    }
    configs = {task: _config(source) for task, source in sources.items()}
    training_markets = tuple(sources["da"].get("global_training_markets", ()))
    expected_markets = (
        {"CAISO", "NYISO", "PJM"}
        if args.source_phase == "inner"
        else {"CAISO", "ERCOT", "NYISO", "PJM"}
    )
    expected_stage = (
        AnnualStage.INNER_TRAIN.value
        if args.source_phase == "inner"
        else AnnualStage.FINAL_TRAIN.value
    )
    if args.train_stage != expected_stage:
        raise ValueError(
            f"source-phase={args.source_phase} requires train-stage={expected_stage}"
        )
    if set(training_markets) != expected_markets:
        raise ValueError(
            f"{args.source_phase} implicit-QP training requires markets "
            f"{sorted(expected_markets)}, got {sorted(training_markets)}"
        )
    full_mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    mappings = full_mappings.subset(training_markets)
    for task in TASKS:
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(sources[task])
        mappings.validate_checkpoint(sources[task])
        if (
            sources[task].get("task") != task
            or sources[task].get("phase") != args.source_phase
        ):
            raise ValueError(
                f"{task} source checkpoint must have phase={args.source_phase}"
            )
        if tuple(sources[task].get("global_training_markets", ())) != training_markets:
            raise ValueError("three source checkpoints disagree on training markets")
        if configs[task].global_use_entity_embeddings:
            raise ValueError("no-ID LOMO training forbids entity embeddings")

    entities = [(mappings.node_to_market[node], node) for node in mappings.node_to_id]
    if args.max_source_nodes > 0:
        entities = entities[: args.max_source_nodes]
    train_dataset, _ = build_global_joint_datasets(
        configs,
        entities,
        mappings,
        train_stage=args.train_stage,
        # The second dataset is not used for this fixed-epoch fine-tune.  Keep it
        # on the small, legal 2023 stage rather than duplicating five training
        # years in memory; no locked-2025 rows are loaded.
        validation_stage=AnnualStage.INNER_VALIDATION.value,
    )
    sampler = NodeBalancedEpisodeBatchSampler(
        train_dataset,
        episode_days=args.episode_days,
        episodes_per_node=args.episodes_per_node,
        seed=args.training_seed,
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
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    settlement = FORMAL_EXPERIMENT_CONTRACT.settlement
    qp_config = BatteryQPConfig(
        horizon=24,
        power_mwh=settlement.bess_power_mw,
        energy_mwh=settlement.bess_energy_mwh,
        soc_min_fraction=(settlement.soc_min_mwh / settlement.bess_energy_mwh),
        soc_max_fraction=(settlement.soc_max_mwh / settlement.bess_energy_mwh),
        initial_soc_fraction=(
            settlement.initial_soc_mwh / settlement.bess_energy_mwh
        ),
        efficiency=settlement.round_trip_component_efficiency,
        daily_discharge_limit_mwh=settlement.daily_discharge_cap_mwh,
        deviation_tolerance=settlement.deviation_tolerance_fraction_of_abs_da,
        quadratic_regularization=args.quadratic_regularization,
        barrier_temperature=args.barrier_temperature,
        newton_steps=args.newton_steps,
    )
    layer = ImplicitBatteryQPLayer(qp_config)
    kappa = settlement.degradation_cost_usd_per_mwh
    history = []
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        rows = []
        for batch_index, raw in enumerate(loader, start=1):
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            p_da, p_rt_at_da, p_rt = _forward(models, batch, device, amp_enabled)
            # Each rolling forecast row is issued immediately before its first
            # executable hour; using horizon 0 preserves the deployment cutoff.
            p_rt_executed = p_rt[:, :, 0]
            cost = predicted_cost_vector(
                p_da, p_rt_at_da, p_rt_executed, degradation_cost=kappa
            )
            solution = layer(cost)
            revenue, parts = realized_two_settlement_revenue(
                layer.split(solution),
                batch["da"]["price_da_tgt"].float(),
                batch["da"]["price_rt_tgt"].float(),
                degradation_cost=kappa,
                deviation_tolerance=qp_config.deviation_tolerance,
            )
            prediction = _prediction_anchor(
                (p_da, p_rt_at_da, p_rt), batch, args.prediction_scale
            )
            decision_loss = -revenue.mean() / args.revenue_scale
            loss = (
                (1.0 - args.decision_weight) * prediction
                + args.decision_weight * decision_loss
            )
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            optimizer.step()
            row = {
                "batch": batch_index,
                "epoch": epoch,
                "mean_revenue": float(revenue.detach().mean().cpu()),
                "prediction_loss": float(prediction.detach().cpu()),
                "decision_loss": float(decision_loss.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
                "gradient_norm_before_clip": gradient_norm,
                "mean_da_component": float(parts["da_component"].mean().detach().cpu()),
                "mean_rt_component": float(parts["rt_component"].mean().detach().cpu()),
                "mean_degradation": float(parts["degradation"].mean().detach().cpu()),
                "mean_deviation_penalty": float(
                    parts["deviation_penalty"].mean().detach().cpu()
                ),
                "qp_gradient_inf_max": max(
                    item["gradient_inf"] for item in layer.last_diagnostics
                ),
                "qp_minimum_slack_min": min(
                    item["minimum_slack"] for item in layer.last_diagnostics
                ),
            }
            rows.append(row)
            print(json.dumps({"event": "batch", **row}, ensure_ascii=False), flush=True)
        history.append({
            "epoch": epoch,
            "batches": len(rows),
            "mean_revenue": float(np.mean([row["mean_revenue"] for row in rows])),
            "mean_prediction_loss": float(
                np.mean([row["prediction_loss"] for row in rows])
            ),
            "mean_decision_loss": float(
                np.mean([row["decision_loss"] for row in rows])
            ),
        })

    output_dir = ROOT / args.output_dir / args.family
    output_dir.mkdir(parents=True, exist_ok=True)
    source_contract = {
        task: {
            "path": paths[task].relative_to(ROOT).as_posix(),
            "sha256": _sha256(paths[task]),
        }
        for task in TASKS
    }
    families = {args.family: {}}
    method = {
        "method": "strongly_convex_barrier_qp_kkt_implicit_differentiation",
        "source_checkpoints": source_contract,
        "training_markets": list(training_markets),
        "training_stage": args.train_stage,
        "training_years": (
            "2020-2022" if args.train_stage == "inner_train" else "2020-2024"
        ),
        "ercot_training_labels_used": "ERCOT" in training_markets,
        "2024_used": args.train_stage == "final_train",
        "2025_used": False,
        "decision_weight": args.decision_weight,
        "revenue_scale": args.revenue_scale,
        "quadratic_regularization": args.quadratic_regularization,
        "barrier_temperature": args.barrier_temperature,
        "economic_interpretation": (
            "hourly DA plan, RT recourse, SOC, degradation and deviation penalty"
        ),
    }
    for task in TASKS:
        payload = dict(sources[task])
        for key in ("optimizer_state", "scheduler_state", "grad_scaler_state"):
            payload.pop(key, None)
        payload["model_state"] = models[task].state_dict()
        payload["decision_aware_finetune"] = method
        path = output_dir / f"{task}.pt"
        torch.save(payload, path)
        families[args.family][task] = path.relative_to(ROOT).as_posix()
    manifest = output_dir / "candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {"families": families, "phase": args.source_phase},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "experiment": f"global_joint_implicit_qp_{args.source_phase}",
        "status": "trained_not_yet_evaluated",
        "family": args.family,
        "device": str(device),
        "source_checkpoints": source_contract,
        "parameters": vars(args),
        "trainable_parameter_count": int(sum(item.numel() for item in parameters)),
        "method": method,
        "history": history,
        "runtime_seconds": perf_counter() - started,
        "candidate_manifest": manifest.relative_to(ROOT).as_posix(),
    }
    result = ROOT / args.result
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "result": result.relative_to(ROOT).as_posix(),
        "manifest": manifest.relative_to(ROOT).as_posix(),
        "runtime_seconds": report["runtime_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

