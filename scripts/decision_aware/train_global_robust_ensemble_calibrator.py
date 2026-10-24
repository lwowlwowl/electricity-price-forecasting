#!/usr/bin/env python
"""Train a robust, low-parameter decision-aware ensemble calibrator.

The default is the original source-market development protocol.  Final mode is
explicit: four-market 2020--2024 data, disjoint fixed samples, a fixed epoch
count, and no access to the locked 2025 stage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import (  # noqa: E402
    AnnualStage,
    FORMAL_EXPERIMENT_CONTRACT,
)
from decision_aware.global_data import flatten_rt_delivery_day_batch, load_formal_entity_mappings  # noqa: E402
from decision_aware.global_joint import build_global_joint_datasets  # noqa: E402
from decision_aware.implicit_battery_qp import (  # noqa: E402
    BatteryQPConfig,
    ImplicitBatteryQPLayer,
    predicted_cost_vector,
    realized_two_settlement_revenue,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402
from decision_aware.robust_ensemble import (  # noqa: E402
    EconomicEnsembleCalibrator,
    EnsembleCalibratorConfig,
)


TASKS = ("da", "rt_at_da", "rt")
MODEL_CLASSES = {
    "da": DecisionAwareDAForecaster,
    "rt_at_da": DecisionAwareRTAtDAForecaster,
    "rt": DecisionAwareRTForecaster,
}


def _config(checkpoint: dict) -> PilotConfig:
    allowed = {item.name for item in fields(PilotConfig)}
    return PilotConfig(**{
        key: value for key, value in checkpoint["config"].items() if key in allowed
    })


def _move(value, device):
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True) if torch.is_tensor(value) else value


def _forward(models, batch, device):
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        da = models["da"](batch["da"])["p_da"].float()
        rt_at_da = models["rt_at_da"](batch["rt_at_da"])["p_rt_at_da"].float()
        rt_flat = flatten_rt_delivery_day_batch(batch["rt"])
        rt = models["rt"](rt_flat)["p_rt"].float()
    return da, rt_at_da, rt.reshape(da.shape[0], 24, -1)


def _balanced_indices(dataset, per_market: int, seed: int) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for entity_index, (market, _) in enumerate(dataset.entities):
        grouped[market].extend(range(dataset.offsets[entity_index], dataset.offsets[entity_index + 1]))
    generator = np.random.default_rng(seed)
    selected = []
    for market in sorted(grouped):
        values = np.asarray(grouped[market], dtype=np.int64)
        if len(values) < per_market:
            raise ValueError(f"{market} has only {len(values)} episodes, need {per_market}")
        selected.extend(generator.choice(values, size=per_market, replace=False).tolist())
    return selected


def _balanced_disjoint_indices(
    dataset, train_per_market: int, validation_per_market: int, seed: int
) -> tuple[list[int], list[int]]:
    """Draw deterministic, market-balanced, non-overlapping samples."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for entity_index, (market, _) in enumerate(dataset.entities):
        grouped[market].extend(
            range(dataset.offsets[entity_index], dataset.offsets[entity_index + 1])
        )
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    required = train_per_market + validation_per_market
    for market in sorted(grouped):
        values = np.asarray(grouped[market], dtype=np.int64)
        if len(values) < required:
            raise ValueError(
                f"{market} has only {len(values)} episodes, need {required}"
            )
        chosen = generator.choice(values, size=required, replace=False)
        train.extend(chosen[:train_per_market].tolist())
        validation.extend(chosen[train_per_market:].tolist())
    if set(train) & set(validation):
        raise AssertionError("train and validation samples overlap")
    return train, validation


def _collect(models_by_seed, dataset, indices, device, batch_size=12) -> dict[str, torch.Tensor]:
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    for raw in loader:
        batch = _move(raw, device)
        seed_outputs = [_forward(models, batch, device) for models in models_by_seed]
        rows["da_members"].append(torch.stack([item[0].cpu() for item in seed_outputs], dim=1))
        rows["rt_at_da_members"].append(torch.stack([item[1].cpu() for item in seed_outputs], dim=1))
        rows["rt_members"].append(torch.stack([item[2].cpu() for item in seed_outputs], dim=1))
        rows["target_da"].append(batch["da"]["price_da_tgt"].float().cpu())
        rows["target_rt"].append(batch["da"]["price_rt_tgt"].float().cpu())
        rows["target_rt_windows"].append(batch["rt"]["price_rt_tgt"].float().cpu())
        rows["market_id"].append(batch["market_id"].long().cpu())
    return {key: torch.cat(values, dim=0) for key, values in rows.items()}


def _qp_config(args) -> BatteryQPConfig:
    settlement = FORMAL_EXPERIMENT_CONTRACT.settlement
    return BatteryQPConfig(
        horizon=24,
        power_mwh=settlement.bess_power_mw,
        energy_mwh=settlement.bess_energy_mwh,
        soc_min_fraction=settlement.soc_min_mwh / settlement.bess_energy_mwh,
        soc_max_fraction=settlement.soc_max_mwh / settlement.bess_energy_mwh,
        initial_soc_fraction=settlement.initial_soc_mwh / settlement.bess_energy_mwh,
        efficiency=settlement.round_trip_component_efficiency,
        daily_discharge_limit_mwh=settlement.daily_discharge_cap_mwh,
        deviation_tolerance=settlement.deviation_tolerance_fraction_of_abs_da,
        quadratic_regularization=args.quadratic_regularization,
        barrier_temperature=args.barrier_temperature,
        newton_steps=args.newton_steps,
    )


def _revenue(layer, da, rt_at_da, rt, target_da, target_rt, degradation):
    cost = predicted_cost_vector(da, rt_at_da, rt[:, :, 0], degradation_cost=degradation)
    solution = layer(cost)
    revenue, _ = realized_two_settlement_revenue(
        layer.split(solution), target_da, target_rt,
        degradation_cost=degradation,
        deviation_tolerance=layer.config.deviation_tolerance,
    )
    return revenue


def _base_revenues(
    cache, layer, degradation, device, *, audit_members: bool = False
) -> tuple[torch.Tensor, dict]:
    member_count = cache["da_members"].shape[1]
    systems = ["equal_weight"]
    if audit_members:
        systems = [f"seed_{index}" for index in range(member_count)] + systems
    outputs = {name: [] for name in systems}
    batch_size = 12
    for start in range(0, len(cache["market_id"]), batch_size):
        stop = min(len(cache["market_id"]), start + batch_size)
        values = {key: item[start:stop].to(device) for key, item in cache.items()}
        if audit_members:
            for member in range(member_count):
                outputs[f"seed_{member}"].append(_revenue(
                    layer,
                    values["da_members"][:, member],
                    values["rt_at_da_members"][:, member],
                    values["rt_members"][:, member],
                    values["target_da"], values["target_rt"], degradation,
                ).detach().cpu())
        outputs["equal_weight"].append(_revenue(
            layer,
            values["da_members"].mean(dim=1),
            values["rt_at_da_members"].mean(dim=1),
            values["rt_members"].mean(dim=1),
            values["target_da"], values["target_rt"], degradation,
        ).detach().cpu())
    concatenated = {name: torch.cat(items) for name, items in outputs.items()}
    return concatenated["equal_weight"], {
        name: float(values.mean()) for name, values in concatenated.items()
    }


def _market_batches(market_ids: torch.Tensor, samples_per_market: int, seed: int):
    generator = np.random.default_rng(seed)
    groups = {
        int(market): np.flatnonzero(market_ids.numpy() == int(market))
        for market in torch.unique(market_ids).tolist()
    }
    batch_count = min(len(values) // samples_per_market for values in groups.values())
    shuffled = {market: generator.permutation(values) for market, values in groups.items()}
    for batch_index in range(batch_count):
        indices = []
        for market in sorted(shuffled):
            start = batch_index * samples_per_market
            indices.extend(shuffled[market][start:start + samples_per_market].tolist())
        yield torch.tensor(indices, dtype=torch.long)


@torch.no_grad()
def _validation_report(
    calibrator, cache, base_revenue, layer, degradation, market_names, device,
    *, tail_score_weight: float = 0.0,
):
    calibrated = []
    batch_size = 12
    for start in range(0, len(cache["market_id"]), batch_size):
        stop = min(len(cache["market_id"]), start + batch_size)
        values = {key: item[start:stop].to(device) for key, item in cache.items()}
        da, rt_at_da, rt = calibrator(
            values["da_members"], values["rt_at_da_members"], values["rt_members"]
        )
        calibrated.append(_revenue(
            layer, da, rt_at_da, rt, values["target_da"], values["target_rt"], degradation
        ).cpu())
    calibrated = torch.cat(calibrated)
    delta = calibrated - base_revenue
    markets = {}
    for market_id, market_name in market_names.items():
        mask = cache["market_id"] == market_id
        market_delta = delta[mask]
        tail_count = max(1, int(np.ceil(0.25 * int(mask.sum()))))
        markets[market_name] = {
            "samples": int(mask.sum()),
            "base_mean_revenue": float(base_revenue[mask].mean()),
            "calibrated_mean_revenue": float(calibrated[mask].mean()),
            "mean_improvement": float(market_delta.mean()),
            "median_improvement": float(market_delta.median()),
            "positive_fraction": float((market_delta > 0.0).float().mean()),
            "lower_quartile_mean_improvement": float(
                torch.topk(market_delta, tail_count, largest=False).values.mean()
            ),
        }
    improvements = np.asarray([item["mean_improvement"] for item in markets.values()])
    global_tail_count = max(1, int(np.ceil(0.25 * len(delta))))
    global_tail_mean = float(
        torch.topk(delta, global_tail_count, largest=False).values.mean()
    )
    score = float(
        improvements.min()
        + 0.25 * improvements.mean()
        + tail_score_weight * global_tail_mean
    )
    return {
        "markets": markets,
        "mean_improvement": float(improvements.mean()),
        "minimum_market_improvement": float(improvements.min()),
        "market_improvement_std": float(improvements.std(ddof=1)),
        "median_improvement": float(delta.median()),
        "positive_fraction": float((delta > 0.0).float().mean()),
        "lower_quartile_mean_improvement": global_tail_mean,
        "selection_score": score,
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", default="implicitqp_decoder_balanced_seed0,implicitqp_decoder_balanced_seed1,implicitqp_decoder_balanced_seed2")
    parser.add_argument("--checkpoint-root", default="data/checkpoints/global_joint_implicit_qp_v1")
    parser.add_argument("--output", default="data/checkpoints/global_robust_ensemble_v1/calibrator.pt")
    parser.add_argument("--report", default="data/results/global_robust_ensemble_train_v1.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-per-market", type=int, default=128)
    parser.add_argument("--validation-per-market", type=int, default=64)
    parser.add_argument("--samples-per-market-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--prediction-weight", type=float, default=0.05)
    parser.add_argument("--market-std-weight", type=float, default=0.50)
    parser.add_argument("--worst-market-weight", type=float, default=0.50)
    parser.add_argument("--regularization-weight", type=float, default=0.01)
    parser.add_argument("--downside-weight", type=float, default=0.0)
    parser.add_argument("--tail-weight", type=float, default=0.0)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--validation-tail-score-weight", type=float, default=0.0)
    parser.add_argument("--revenue-scale", type=float, default=500.0)
    parser.add_argument("--quadratic-regularization", type=float, default=0.25)
    parser.add_argument("--barrier-temperature", type=float, default=0.10)
    parser.add_argument("--newton-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--source-phase", choices=("inner", "final"), default="inner")
    parser.add_argument(
        "--train-stage",
        choices=(AnnualStage.INNER_TRAIN.value, AnnualStage.FINAL_TRAIN.value),
        default=AnnualStage.INNER_TRAIN.value,
    )
    parser.add_argument(
        "--validation-stage",
        choices=(AnnualStage.INNER_VALIDATION.value, AnnualStage.FINAL_TRAIN.value),
        default=AnnualStage.INNER_VALIDATION.value,
    )
    parser.add_argument(
        "--selection-mode",
        choices=("best_validation", "fixed_last"),
        default="best_validation",
    )
    parser.add_argument(
        "--source-cache",
        default="data/results/global_robust_ensemble_source_cache_v1.pt",
    )
    parser.add_argument(
        "--resume-state",
        default="",
        help="optional epoch-level resumable training state",
    )
    args = parser.parse_args()
    members = tuple(value.strip() for value in args.members.split(",") if value.strip())
    if len(members) != 3:
        raise ValueError("exactly three frozen members are required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)

    checkpoint_root = ROOT / args.checkpoint_root
    sources_by_seed = []
    models_by_seed = []
    paths_by_seed = []
    for family in members:
        paths = {task: checkpoint_root / family / f"{task}.pt" for task in TASKS}
        sources = {task: torch.load(path, map_location="cpu", weights_only=False) for task, path in paths.items()}
        configs = {task: _config(source) for task, source in sources.items()}
        models = {task: MODEL_CLASSES[task](configs[task]).to(device) for task in TASKS}
        for task in TASKS:
            models[task].load_state_dict(sources[task]["model_state"], strict=True)
            models[task].eval().requires_grad_(False)
        sources_by_seed.append(sources)
        models_by_seed.append(models)
        paths_by_seed.append(paths)

    training_markets = tuple(sources_by_seed[0]["da"].get("global_training_markets", ()))
    expected_markets = (
        {"CAISO", "NYISO", "PJM"}
        if args.source_phase == "inner"
        else {"CAISO", "ERCOT", "NYISO", "PJM"}
    )
    expected_train_stage = (
        AnnualStage.INNER_TRAIN.value
        if args.source_phase == "inner"
        else AnnualStage.FINAL_TRAIN.value
    )
    if set(training_markets) != expected_markets:
        raise ValueError(
            f"{args.source_phase} calibrator requires markets "
            f"{sorted(expected_markets)}, got {sorted(training_markets)}"
        )
    if args.train_stage != expected_train_stage:
        raise ValueError(
            f"source-phase={args.source_phase} requires train-stage={expected_train_stage}"
        )
    if args.source_phase == "final" and (
        args.validation_stage != AnnualStage.FINAL_TRAIN.value
        or args.selection_mode != "fixed_last"
    ):
        raise ValueError(
            "final calibrator requires validation-stage=final_train and "
            "selection-mode=fixed_last"
        )
    configs = {task: _config(sources_by_seed[0][task]) for task in TASKS}
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(training_markets)
    for sources in sources_by_seed:
        for task in TASKS:
            FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(sources[task])
            mappings.validate_checkpoint(sources[task])
            if sources[task].get("task") != task:
                raise ValueError(f"member checkpoint task mismatch for {task}")
            if sources[task].get("phase") != args.source_phase:
                raise ValueError(
                    f"member checkpoint for {task} must have phase={args.source_phase}"
                )
            if tuple(sources[task].get("global_training_markets", ())) != training_markets:
                raise ValueError("member checkpoints disagree on training markets")
    entities = [(mappings.node_to_market[node], node) for node in mappings.node_to_id]
    train_dataset, validation_dataset = build_global_joint_datasets(
        configs,
        entities,
        mappings,
        train_stage=args.train_stage,
        validation_stage=args.validation_stage,
    )
    if args.train_stage == args.validation_stage:
        train_indices, validation_indices = _balanced_disjoint_indices(
            train_dataset,
            args.train_per_market,
            args.validation_per_market,
            args.seed,
        )
    else:
        train_indices = _balanced_indices(
            train_dataset, args.train_per_market, args.seed
        )
        validation_indices = _balanced_indices(
            validation_dataset, args.validation_per_market, args.seed + 1
        )

    qp_config = _qp_config(args)
    layer = ImplicitBatteryQPLayer(qp_config)
    degradation = FORMAL_EXPERIMENT_CONTRACT.settlement.degradation_cost_usd_per_mwh
    checkpoint_contract = {
        family: {
            task: {"path": paths_by_seed[index][task].relative_to(ROOT).as_posix(), "sha256": _hash(paths_by_seed[index][task])}
            for task in TASKS
        }
        for index, family in enumerate(members)
    }
    cache_contract = {
        "source_checkpoints": checkpoint_contract,
        "training_markets": list(training_markets),
        "train_stage": args.train_stage,
        "validation_stage": args.validation_stage,
        "selection_mode": args.selection_mode,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "qp_config": asdict(qp_config),
        "degradation_cost": degradation,
    }
    started = perf_counter()
    cache_path = ROOT / args.source_cache
    cache_hit = False
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("contract") == cache_contract:
            train_cache = cached["train_cache"]
            validation_cache = cached["validation_cache"]
            train_base = cached["train_base"]
            validation_base = cached["validation_base"]
            train_audit = cached["train_audit"]
            validation_audit = cached["validation_audit"]
            cache_hit = True
    if not cache_hit:
        train_cache = _collect(models_by_seed, train_dataset, train_indices, device)
        validation_cache = _collect(models_by_seed, validation_dataset, validation_indices, device)
        train_base, train_audit = _base_revenues(
            train_cache, layer, degradation, device, audit_members=False
        )
        validation_base, validation_audit = _base_revenues(
            validation_cache, layer, degradation, device, audit_members=False
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "contract": cache_contract,
            "train_cache": train_cache,
            "validation_cache": validation_cache,
            "train_base": train_base,
            "validation_base": validation_base,
            "train_audit": train_audit,
            "validation_audit": validation_audit,
        }, cache_path)
    del models_by_seed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    calibrator = EconomicEnsembleCalibrator(EnsembleCalibratorConfig()).to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.lr)
    market_names = {value: key for key, value in mappings.market_to_id.items()}
    initial_validation = _validation_report(
        calibrator, validation_cache, validation_base, layer, degradation, market_names, device,
        tail_score_weight=args.validation_tail_score_weight,
    )
    best_score = initial_validation["selection_score"]
    best_state = {key: value.detach().cpu().clone() for key, value in calibrator.state_dict().items()}
    best_epoch = 0
    history = []
    start_epoch = 1
    resume_path = ROOT / args.resume_state if args.resume_state else None
    resume_contract = {
        "cache_contract": cache_contract,
        "epochs": args.epochs,
        "samples_per_market_batch": args.samples_per_market_batch,
        "lr": args.lr,
        "prediction_weight": args.prediction_weight,
        "market_std_weight": args.market_std_weight,
        "worst_market_weight": args.worst_market_weight,
        "regularization_weight": args.regularization_weight,
        "downside_weight": args.downside_weight,
        "tail_weight": args.tail_weight,
        "tail_fraction": args.tail_fraction,
        "validation_tail_score_weight": args.validation_tail_score_weight,
        "revenue_scale": args.revenue_scale,
        "seed": args.seed,
    }
    if resume_path is not None and resume_path.exists():
        resume = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if resume.get("contract") != resume_contract:
            raise ValueError("resume state contract does not match current run")
        calibrator.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        best_score = float(resume["best_score"])
        best_epoch = int(resume["best_epoch"])
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in resume["best_state"].items()
        }
        history = list(resume["history"])
        start_epoch = int(resume["completed_epoch"]) + 1
        print(json.dumps({
            "event": "resume",
            "completed_epoch": start_epoch - 1,
            "next_epoch": start_epoch,
            "path": resume_path.relative_to(ROOT).as_posix(),
        }, ensure_ascii=False), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        calibrator.train()
        rows = []
        for indices in _market_batches(
            train_cache["market_id"], args.samples_per_market_batch, args.seed + epoch
        ):
            values = {key: item[indices].to(device) for key, item in train_cache.items()}
            optimizer.zero_grad(set_to_none=True)
            da, rt_at_da, rt = calibrator(
                values["da_members"], values["rt_at_da_members"], values["rt_members"]
            )
            revenue = _revenue(
                layer, da, rt_at_da, rt, values["target_da"], values["target_rt"], degradation
            )
            delta = revenue - train_base[indices].to(device)
            market_means = torch.stack([
                delta[values["market_id"] == market].mean()
                for market in torch.unique(values["market_id"], sorted=True)
            ])
            robust = (
                -market_means.mean()
                + args.market_std_weight * market_means.std(unbiased=True)
                + args.worst_market_weight * torch.relu(-market_means.min())
            ) / args.revenue_scale
            tail_count = max(1, int(np.ceil(args.tail_fraction * len(delta))))
            lower_tail = torch.topk(delta, tail_count, largest=False).values.mean()
            robust = robust + (
                args.downside_weight * torch.relu(-delta).mean()
                + args.tail_weight * torch.relu(-lower_tail)
            ) / args.revenue_scale
            prediction = (
                F.smooth_l1_loss(da, values["target_da"], beta=25.0)
                + F.smooth_l1_loss(rt_at_da, values["target_rt"], beta=25.0)
                + F.smooth_l1_loss(rt, values["target_rt_windows"], beta=25.0)
            ) / (3.0 * 25.0)
            loss = (
                robust
                + args.prediction_weight * prediction
                + args.regularization_weight * calibrator.regularization()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 5.0)
            optimizer.step()
            rows.append((float(loss.detach()), float(delta.mean().detach())))

        calibrator.eval()
        validation = _validation_report(
            calibrator, validation_cache, validation_base, layer, degradation, market_names, device,
            tail_score_weight=args.validation_tail_score_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean([item[0] for item in rows])),
            "train_mean_improvement": float(np.mean([item[1] for item in rows])),
            "validation": validation,
        }
        history.append(row)
        print(json.dumps({"event": "epoch", **row}, ensure_ascii=False), flush=True)
        if args.selection_mode == "fixed_last":
            best_score = validation["selection_score"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in calibrator.state_dict().items()
            }
        elif validation["selection_score"] > best_score:
            best_score = validation["selection_score"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in calibrator.state_dict().items()}
        if resume_path is not None:
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = resume_path.with_suffix(resume_path.suffix + ".tmp")
            torch.save({
                "contract": resume_contract,
                "completed_epoch": epoch,
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in calibrator.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "history": history,
            }, temporary)
            temporary.replace(resume_path)

    calibrator.load_state_dict(best_state)
    calibrator.eval()
    final_validation = _validation_report(
        calibrator, validation_cache, validation_base, layer, degradation, market_names, device,
        tail_score_weight=args.validation_tail_score_weight,
    )
    improved = bool(
        best_epoch > 0
        and final_validation["minimum_market_improvement"] > 0.0
        and final_validation["mean_improvement"] > 0.0
    )
    deployment_approved = bool(
        best_epoch == args.epochs
        and args.selection_mode == "fixed_last"
        and np.isfinite(final_validation["selection_score"])
    ) or improved
    payload = {
        "model_state": best_state,
        "calibrator_config": calibrator.config.to_dict(),
        "members": list(members),
        "source_checkpoints": checkpoint_contract,
        "training_markets": list(training_markets),
        "training_stage": args.train_stage,
        "validation_stage": args.validation_stage,
        "ercot_training_labels_used": "ERCOT" in training_markets,
        "2024_used": args.train_stage == AnnualStage.FINAL_TRAIN.value,
        "2025_used": False,
        "parameters": vars(args),
        "best_epoch": best_epoch,
        "source_validation_improved": improved,
        "deployment_approved": deployment_approved,
        "source_validation": final_validation,
        "interpretation": calibrator.interpretation(),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        "experiment": f"global_robust_ensemble_calibrator_{args.source_phase}",
        "status": "pretest_calibration_complete_2025_unread",
        "members": list(members),
        "training_markets": list(training_markets),
        "ercot_read": "ERCOT" in training_markets,
        "2024_used": args.train_stage == AnnualStage.FINAL_TRAIN.value,
        "2025_used": False,
        "parameters": vars(args),
        "source_checkpoint_contract": checkpoint_contract,
        "source_proxy_revenue_audit": {"train": train_audit, "validation": validation_audit},
        "source_cache": {
            "path": cache_path.relative_to(ROOT).as_posix(),
            "hit": cache_hit,
            "contract": cache_contract,
        },
        "initial_validation": initial_validation,
        "history": history,
        "best_epoch": best_epoch,
        "source_validation_improved": improved,
        "deployment_approved": deployment_approved,
        "final_validation": final_validation,
        "interpretation": calibrator.interpretation(),
        "runtime_seconds": perf_counter() - started,
        "checkpoint": output.relative_to(ROOT).as_posix(),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if resume_path is not None and resume_path.exists():
        resume_path.unlink()
    print(json.dumps({
        "event": "complete",
        "source_validation_improved": improved,
        "deployment_approved": deployment_approved,
        "best_epoch": best_epoch,
        "selection_score": final_validation["selection_score"],
        "report": report_path.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
