#!/usr/bin/env python
"""Train and evaluate cutoff-safe all-market DLinear models for 2023/2024.

This is deliberately separate from the locked-2025 entry point.  It creates
strictly out-of-sample DLinear predictions that may be used to develop an
interpretable expert router without reading 2025.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.dlinear import DLinearForecaster  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    MARKET_ORDER,
    MarketNodeBalancedSampler,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.loader_v2 import market_timezone  # noqa: E402
from scripts.decision_aware.export_global_f0_f1_predictions import (  # noqa: E402
    _dates_and_timestamps,
)
from scripts.decision_aware.run_allmarket_final_dlinear import (  # noqa: E402
    TASKS,
    _batch,
    _loss,
    _protocol,
    _sha256,
    _validate_cfg,
)


PHASES = {
    "inner": {
        "train_stage": AnnualStage.INNER_TRAIN,
        "evaluation_stage": AnnualStage.INNER_VALIDATION,
        "training_years": [2020, 2021, 2022],
        "evaluation_year": 2023,
        "selection_allowed": True,
    },
    "outer": {
        "train_stage": AnnualStage.OUTER_TRAIN,
        "evaluation_stage": AnnualStage.OUTER_VALIDATION,
        "training_years": [2020, 2021, 2022, 2023],
        "evaluation_year": 2024,
        "selection_allowed": False,
    },
}


def _read_end_by_market(stage: AnnualStage) -> dict[str, str]:
    end = FORMAL_EXPERIMENT_CONTRACT.annual_split.range_for(stage).end
    return {
        market: (
            pd.Timestamp(end, tz=market_timezone(market)) + pd.Timedelta(days=1)
        ).tz_convert("UTC").isoformat()
        for market in MARKET_ORDER
    }


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    return torch.device(
        value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def _training_markets(holdout_market: str | None) -> list[str]:
    holdout = holdout_market.upper() if holdout_market else None
    if holdout is not None and holdout not in MARKET_ORDER:
        raise ValueError(f"未知留出市场: {holdout}")
    markets = [market for market in MARKET_ORDER if market != holdout]
    if not markets:
        raise ValueError("训练市场不能为空")
    return markets


def _train(args: argparse.Namespace, cfg: PilotConfig, protocol: dict) -> None:
    spec = PHASES[args.phase]
    checkpoint = ROOT / args.checkpoint
    if checkpoint.exists() and not args.force:
        raise FileExistsError(f"checkpoint已存在: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    section = protocol["dlinear"]
    seed = int(section["seed"])
    _set_seed(seed)
    device = _device(args.device)
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    training_markets = _training_markets(args.holdout_market)
    entities = [
        (mappings.node_to_market[node], node)
        for node in mappings.node_to_id
        if mappings.node_to_market[node] in training_markets
    ]
    if args.smoke:
        entities = [
            next(entity for entity in entities if entity[0] == market)
            for market in training_markets
        ]
    train, _ = build_global_task_datasets(
        cfg,
        args.task,
        entities,
        mappings,
        train_stage=spec["train_stage"],
        validation_stage=spec["evaluation_stage"],
        read_end_utc=_read_end_by_market(spec["evaluation_stage"]),
    )
    sampler = MarketNodeBalancedSampler(
        train,
        samples_per_market=8 if args.smoke else int(section["samples_per_market"]),
        seed=seed,
    )
    loader = DataLoader(
        train,
        batch_size=4 if args.smoke else int(section["batch_size"]),
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = DLinearForecaster(
        168, TASKS[args.task][-1], int(section["moving_average_kernel"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    epochs = 1 if args.smoke else int(section["fixed_epochs"])
    started = perf_counter()
    history: list[dict] = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for raw in loader:
            history_x, _, target_n, _, _ = _batch(raw, args.task, device)
            optimizer.zero_grad(set_to_none=True)
            prediction_n = model(history_x)
            loss = _loss(prediction_n, target_n, args.task)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps(record), flush=True)
    metadata = {
        "protocol": "all_market_dlinear_historical_train_v1",
        "task": args.task,
        "phase": args.phase,
        "seed": seed,
        "training_markets": training_markets,
        "holdout_market": args.holdout_market.upper() if args.holdout_market else None,
        "training_years": spec["training_years"],
        "evaluation_year": spec["evaluation_year"],
        "locked_2025_read": False,
        "selection_allowed": spec["selection_allowed"],
        "fixed_epochs": epochs,
        "input_features": [TASKS[args.task][0]],
        "uses_node_or_market_id_for_prediction": False,
        "kernel_size": int(section["moving_average_kernel"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "wall_seconds": float(perf_counter() - started),
        "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
        "config": Path(args.config).as_posix(),
        "smoke_only": bool(args.smoke),
    }
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, checkpoint)
    summary = {**metadata, "checkpoint": checkpoint.relative_to(ROOT).as_posix()}
    summary["checkpoint_sha256"] = _sha256(checkpoint)
    checkpoint.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


@torch.no_grad()
def _evaluate(args: argparse.Namespace, cfg: PilotConfig, protocol: dict) -> None:
    spec = PHASES[args.phase]
    checkpoint = ROOT / args.checkpoint
    output = ROOT / args.output
    if output.exists() and not args.force:
        raise FileExistsError(f"输出已存在: {output}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    frozen = payload.get("metadata", {})
    expected = {
        "protocol": "all_market_dlinear_historical_train_v1",
        "task": args.task,
        "phase": args.phase,
        "seed": int(protocol["dlinear"]["seed"]),
        "evaluation_year": spec["evaluation_year"],
        "locked_2025_read": False,
        "holdout_market": args.holdout_market.upper() if args.holdout_market else None,
    }
    mismatches = {
        key: (frozen.get(key), value)
        for key, value in expected.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise ValueError(f"DLinear历史checkpoint元数据不匹配: {mismatches}")
    device = _device(args.device)
    model = DLinearForecaster(
        168, TASKS[args.task][-1], int(protocol["dlinear"]["moving_average_kernel"])
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    market = args.market.upper()
    if args.holdout_market and market != args.holdout_market.upper():
        raise ValueError("LOMO DLinear只能评价其冻结留出市场")
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset([market])
    nodes = list(mappings.node_to_id)
    if args.smoke:
        nodes = nodes[:1]
    entities = [(market, node) for node in nodes]
    _, validation = build_global_task_datasets(
        replace(cfg, global_use_critical_event_weighting=True),
        args.task,
        entities,
        mappings,
        train_stage=spec["train_stage"],
        validation_stage=spec["evaluation_stage"],
        read_end_utc=_read_end_by_market(spec["evaluation_stage"])[market],
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    targets_da: list[np.ndarray] = []
    common_dates: np.ndarray | None = None
    common_timestamps: np.ndarray | None = None
    per_node: dict[str, dict[str, float]] = {}
    inference_started = perf_counter()
    for entity_index, ((_, node), node_dataset) in enumerate(
        zip(validation.entities, validation.datasets)
    ):
        start, stop = validation.offsets[entity_index : entity_index + 2]
        indices = list(range(start, stop))
        if args.smoke:
            indices = indices[:2]
        loader = DataLoader(Subset(validation, indices), batch_size=64, shuffle=False)
        node_predictions: list[np.ndarray] = []
        node_targets: list[np.ndarray] = []
        node_da: list[np.ndarray] = []
        for raw in loader:
            history_x, target, _, mean, std = _batch(raw, args.task, device)
            prediction = model(history_x) * std + mean
            lo, hi = cfg.pred_clamp
            prediction = prediction.clamp(float(lo), float(hi))
            if args.task == "rt":
                prediction = prediction.reshape(-1, 24, prediction.shape[-1])
                target = target.reshape(-1, 24, target.shape[-1])
            node_predictions.append(prediction.cpu().numpy())
            node_targets.append(target.cpu().numpy())
            if args.task == "rt_at_da":
                node_da.append(raw["price_da_tgt_weighting_only"].numpy())
        prediction_np = np.concatenate(node_predictions).astype(np.float32)
        target_np = np.concatenate(node_targets).astype(np.float32)
        dates, timestamps = _dates_and_timestamps(node_dataset, args.task)
        if args.smoke:
            dates = dates[:2]
            if timestamps is not None:
                timestamps = timestamps[:2]
        if common_dates is None:
            common_dates, common_timestamps = dates, timestamps
        elif not np.array_equal(common_dates, dates):
            raise ValueError(f"节点日期不一致: {node}")
        error = prediction_np.astype(np.float64) - target_np.astype(np.float64)
        per_node[node] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "bias": float(np.mean(error)),
        }
        predictions.append(prediction_np)
        targets.append(target_np)
        if node_da:
            targets_da.append(np.concatenate(node_da).astype(np.float32))
    inference_seconds = float(perf_counter() - inference_started)
    summary = {
        "protocol": "all_market_dlinear_historical_evaluation_v1",
        "task": args.task,
        "phase": args.phase,
        "evaluation_market": market,
        "evaluation_year": spec["evaluation_year"],
        "selection_allowed": spec["selection_allowed"],
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "nodes": len(nodes),
        "node_macro_mae": float(np.mean([value["mae"] for value in per_node.values()])),
        "node_macro_rmse": float(np.mean([value["rmse"] for value in per_node.values()])),
        "node_macro_bias": float(np.mean([value["bias"] for value in per_node.values()])),
        "per_node": per_node,
        "inference_seconds": inference_seconds,
        "smoke_only": bool(args.smoke),
        "locked_2025_read": False,
    }
    arrays: dict[str, np.ndarray] = {
        "nodes": np.asarray(nodes),
        "dates": common_dates,
        "prediction": np.stack(predictions),
        "target": np.stack(targets),
        "metadata": np.asarray(json.dumps(summary, ensure_ascii=False, sort_keys=True)),
    }
    if targets_da:
        arrays["target_da"] = np.stack(targets_da)
    if args.task == "rt":
        arrays["executed_timestamps"] = common_timestamps
        arrays["timezone"] = np.asarray(market_timezone(market))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("train", "evaluate"))
    parser.add_argument("--phase", required=True, choices=tuple(PHASES))
    parser.add_argument(
        "--protocol-config",
        default="configs/decision_aware/allmarket_final_baselines_v1.yaml",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=tuple(TASKS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--market", choices=tuple(MARKET_ORDER))
    parser.add_argument(
        "--holdout-market",
        choices=tuple(MARKET_ORDER),
        help="可选；训练时完全排除该市场，评价时只允许评价该市场。",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = _protocol(ROOT / args.protocol_config, args.task)
    cfg = PilotConfig.from_yaml(ROOT / args.config)
    _validate_cfg(cfg)
    if args.mode == "train":
        if args.market or args.output:
            raise ValueError("train模式不接受--market/--output")
        _train(args, cfg, protocol)
    else:
        if not args.market or not args.output:
            raise ValueError("evaluate必须提供--market/--output")
        _evaluate(args, cfg, protocol)


if __name__ == "__main__":
    main()
