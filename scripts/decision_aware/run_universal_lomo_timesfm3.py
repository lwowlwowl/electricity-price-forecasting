#!/usr/bin/env python
"""Frozen TimesFM-3在当前四市场三任务合同下的正式推理入口。

主比较采用逐节点单变量模式：目标价格使用原始168小时历史；DA和RT-at-DA
只增加历史+未来均确定的8维日历协变量；滚动RT还可使用起报时已经公布的
目标时段DA价格。未来真实RT、负荷、风电和光伏不会进入请求。
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "models" / "workers"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    RTDeliveryDayDataset,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.loader_v2 import market_timezone  # noqa: E402
from scripts.decision_aware.export_global_f0_f1_predictions import (  # noqa: E402
    _dates_and_timestamps,
)
from foundation import Task, TimesFMForecaster, VENV_PYTHON, WORKER_SCRIPT  # noqa: E402
from worker_timesfm import MODEL_ID  # noqa: E402


MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")
TASKS = ("da", "rt_at_da", "rt")
EXPECTED_HORIZON = {"da": 24, "rt_at_da": 24, "rt": 4}
CALENDAR_NAMES = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend", "is_holiday",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(
    cfg: PilotConfig,
    task: str,
    holdout_market: str,
    *,
    locked_2025: bool = False,
) -> None:
    if task not in TASKS:
        raise ValueError(f"未知任务: {task}")
    if holdout_market not in MARKETS:
        raise ValueError(f"未知市场: {holdout_market}")
    if cfg.context_len != 168:
        raise ValueError("TimesFM-3正式对照要求168小时历史")
    horizon = cfg.horizon_rt if task == "rt" else cfg.horizon_da
    if horizon != EXPECTED_HORIZON[task]:
        raise ValueError(f"{task}预测长度必须为{EXPECTED_HORIZON[task]}")
    if cfg.global_use_entity_embeddings:
        raise ValueError("TimesFM-3零样本对照禁止项目node/market embedding")
    if str(cfg.global_normalization_mode).lower() != "context_robust":
        raise ValueError("TimesFM-3对照必须复用无ID模型的context_robust窗口合同")
    training_markets = {
        str(item).upper() for item in cfg.global_training_markets
    }
    if locked_2025:
        if training_markets != set(MARKETS):
            raise ValueError("锁定2025对照必须使用四市场最终配置")
    elif holdout_market in training_markets:
        raise ValueError("留出市场出现在LOMO配置训练市场中")


def _window_arrays(
    dataset: Any,
    task: str,
    window_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """返回target context、目标位置、历史/未来合法协变量和协变量名。"""
    window = dataset.windows[window_index]
    if task == "rt":
        context_end = int(window.issue_position)
        target_price = dataset.price_rt
    else:
        context_end = int(window.context_end)
        target_price = dataset.price_da if task == "da" else dataset.price_rt
    context_start = int(window.context_start)
    target_positions = np.asarray(window.target_positions, dtype=np.int64)
    if context_end - context_start != 168:
        raise ValueError("TimesFM-3样本历史长度不是168")
    if len(target_positions) != EXPECTED_HORIZON[task]:
        raise ValueError("TimesFM-3样本预测长度不符合任务合同")
    if np.any(target_positions < context_end):
        raise ValueError("TimesFM-3目标窗口与历史窗口重叠")

    context = target_price[context_start:context_end].astype(np.float32, copy=True)
    calendar = dataset.streams["cal"]
    hist_cov = calendar[context_start:context_end].astype(np.float32, copy=True)
    future_cov = calendar[target_positions].astype(np.float32, copy=True)
    covariate_names = CALENDAR_NAMES
    if task == "rt":
        # RT起报时目标时段DA价格已经公布；它是合法future-known协变量。
        hist_cov = np.column_stack([
            hist_cov,
            dataset.price_da[context_start:context_end],
        ]).astype(np.float32)
        future_cov = np.column_stack([
            future_cov,
            dataset.price_da[target_positions],
        ]).astype(np.float32)
        covariate_names = (*CALENDAR_NAMES, "published_da_price")
    for label, values in {
        "context": context,
        "hist_cov": hist_cov,
        "future_cov": future_cov,
    }.items():
        if not np.isfinite(values).all():
            raise ValueError(f"TimesFM-3 {label}包含非有限值")
    return context, target_positions, hist_cov, future_cov, covariate_names


def make_timesfm_task(
    dataset: Any,
    task: str,
    window_index: int,
    node: str,
) -> Task:
    context, target_positions, hist_cov, future_cov, _ = _window_arrays(
        dataset, task, window_index
    )
    window = dataset.windows[window_index]
    return Task(
        context=context[:, None],
        index=pd.DatetimeIndex(dataset.index[target_positions]),
        series_names=[node],
        hist_cov=hist_cov,
        future_cov=future_cov,
        start_ts=int(dataset.index[int(window.context_start)].timestamp()),
    )


def _forecast_arrays(forecasts: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not forecasts:
        raise ValueError("TimesFM-3没有收到预测任务")
    mean = np.stack([np.asarray(item.mean, dtype=np.float32) for item in forecasts])
    q10 = np.stack([np.asarray(item.q10, dtype=np.float32) for item in forecasts])
    q90 = np.stack([np.asarray(item.q90, dtype=np.float32) for item in forecasts])
    if mean.shape != q10.shape or mean.shape != q90.shape:
        raise ValueError("TimesFM-3点预测和分位数形状不一致")
    if not (np.isfinite(mean).all() and np.isfinite(q10).all() and np.isfinite(q90).all()):
        raise ValueError("TimesFM-3输出包含非有限值")
    if np.any(q10 > q90):
        raise ValueError("TimesFM-3输出存在q10>q90")
    return mean, q10, q90


def predict_node(
    model: TimesFMForecaster,
    node_dataset: Any,
    task: str,
    node: str,
    *,
    max_days: int | None = None,
) -> dict[str, np.ndarray]:
    horizon = EXPECTED_HORIZON[task]
    if task == "rt":
        if not isinstance(node_dataset, RTDeliveryDayDataset):
            raise TypeError("滚动RT TimesFM-3评价要求RTDeliveryDayDataset")
        days = node_dataset.days[:max_days] if max_days else node_dataset.days
        hourly = node_dataset.hourly_dataset
        window_indices = [index for _, indices in days for index in indices]
        tasks = [make_timesfm_task(hourly, task, index, node) for index in window_indices]
        forecasts = model.predict_tasks(tasks, horizon=horizon, multivariate=False)
        mean, q10, q90 = _forecast_arrays(forecasts)
        shape = (len(days), 24, horizon)
        mean, q10, q90 = mean.reshape(shape), q10.reshape(shape), q90.reshape(shape)
        target = np.stack([
            np.stack([
                hourly.price_rt[hourly.windows[index].target_positions]
                for index in indices
            ])
            for _, indices in days
        ]).astype(np.float32)
        dates = np.asarray([str(day) for day, _ in days])
        target_da = None
    else:
        count = min(len(node_dataset), max_days) if max_days else len(node_dataset)
        window_indices = list(range(count))
        tasks = [
            make_timesfm_task(node_dataset, task, index, node)
            for index in window_indices
        ]
        forecasts = model.predict_tasks(tasks, horizon=horizon, multivariate=False)
        mean, q10, q90 = _forecast_arrays(forecasts)
        price = node_dataset.price_da if task == "da" else node_dataset.price_rt
        target = np.stack([
            price[node_dataset.windows[index].target_positions]
            for index in window_indices
        ]).astype(np.float32)
        dates = np.asarray([
            str(node_dataset.sample_time_contract(index)["delivery_date"])
            for index in window_indices
        ])
        target_da = (
            np.stack([
                node_dataset.price_da[node_dataset.windows[index].target_positions]
                for index in window_indices
            ]).astype(np.float32)
            if task == "rt_at_da" else None
        )
    model._cache.clear()
    result = {
        "prediction": mean.astype(np.float32),
        "q10": q10.astype(np.float32),
        "q90": q90.astype(np.float32),
        "target": target,
        "dates": dates,
    }
    if target_da is not None:
        result["target_da"] = target_da
    return result


def _node_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def _default_output(market: str, task: str, stage: str) -> Path:
    suffix = (
        "_locked_2025" if stage == "locked_2025"
        else "_outer_frozen" if stage == "outer_frozen"
        else ""
    )
    return ROOT / "data" / "results" / f"universal_lomo_timesfm3{suffix}_v1" / (
        f"{market.lower()}_{task}_v1.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--holdout-market", required=True, choices=MARKETS)
    parser.add_argument(
        "--evaluation-stage",
        choices=("inner", "outer_frozen", "locked_2025"),
        default="inner",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = PilotConfig.from_yaml(cfg_path)
    locked_2025 = args.evaluation_stage == "locked_2025"
    validate_protocol(
        cfg,
        args.task,
        args.holdout_market,
        locked_2025=locked_2025,
    )
    python_path = Path(VENV_PYTHON["timesfm"])
    worker_path = Path(WORKER_SCRIPT["timesfm"])
    if not python_path.exists() or not worker_path.exists():
        raise FileNotFoundError("TimesFM-3独立环境或worker不存在")
    preflight = {
        "model": "TimesFM-3 frozen",
        "model_id": MODEL_ID,
        "task": args.task,
        "holdout_market": args.holdout_market,
        "context_hours": cfg.context_len,
        "horizon": EXPECTED_HORIZON[args.task],
        "target_mode": "per_node_univariate",
        "future_covariates": list(CALENDAR_NAMES) + (
            ["published_da_price"] if args.task == "rt" else []
        ),
        "project_supervised_training": False,
        "test_2025": locked_2025,
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    output = ROOT / args.output if args.output else _default_output(
        args.holdout_market, args.task, args.evaluation_stage
    )
    if output.exists() and not args.force:
        raise FileExistsError(f"结果已存在；如需覆盖请加--force: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    full_mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    mappings = full_mappings.subset([args.holdout_market])
    nodes = list(mappings.node_to_id)
    max_nodes = 1 if args.smoke else args.max_nodes
    max_days = 2 if args.smoke else args.max_days
    if max_nodes:
        nodes = nodes[:max_nodes]
    entities = [(args.holdout_market, node) for node in nodes]
    validation_stage = (
        AnnualStage.LOCKED_RETROSPECTIVE_TEST
        if locked_2025 else
        AnnualStage.OUTER_VALIDATION
        if args.evaluation_stage == "outer_frozen"
        else AnnualStage.INNER_VALIDATION
    )
    train_stage = (
        AnnualStage.FINAL_TRAIN if locked_2025 else AnnualStage.INNER_TRAIN
    )
    evaluation_end = FORMAL_EXPERIMENT_CONTRACT.annual_split.range_for(
        validation_stage
    ).end
    read_end_utc = (
        pd.Timestamp(evaluation_end, tz=market_timezone(args.holdout_market))
        + pd.Timedelta(days=1)
    ).tz_convert("UTC").isoformat()
    evaluation_cfg = replace(cfg, global_use_critical_event_weighting=True)
    _, validation = build_global_task_datasets(
        evaluation_cfg,
        args.task,
        entities,
        mappings,
        train_stage=train_stage,
        validation_stage=validation_stage,
        read_end_utc=read_end_utc,
    )

    started = perf_counter()
    model = TimesFMForecaster()
    predictions: list[np.ndarray] = []
    q10_values: list[np.ndarray] = []
    q90_values: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_da_values: list[np.ndarray] = []
    common_dates: np.ndarray | None = None
    common_timestamps: np.ndarray | None = None
    per_node: dict[str, dict[str, float]] = {}
    for (market, node), node_dataset in zip(validation.entities, validation.datasets):
        result = predict_node(
            model,
            node_dataset,
            args.task,
            node,
            max_days=max_days or None,
        )
        _, timestamps = _dates_and_timestamps(node_dataset, args.task)
        if max_days and timestamps is not None:
            timestamps = timestamps[:max_days]
        if common_dates is None:
            common_dates = result["dates"]
            common_timestamps = timestamps
        elif not np.array_equal(common_dates, result["dates"]):
            raise ValueError(f"留出市场各节点日期不一致，首次冲突节点={node}")
        predictions.append(result["prediction"])
        q10_values.append(result["q10"])
        q90_values.append(result["q90"])
        targets.append(result["target"])
        if "target_da" in result:
            target_da_values.append(result["target_da"])
        per_node[node] = _node_metrics(result["target"], result["prediction"])
        print(json.dumps({
            "event": "timesfm3_node_complete",
            "market": market,
            "node": node,
            **per_node[node],
        }, ensure_ascii=False), flush=True)

    prediction = np.stack(predictions)
    target = np.stack(targets)
    q10 = np.stack(q10_values)
    q90 = np.stack(q90_values)
    wall_seconds = float(perf_counter() - started)
    prediction_elements = int(prediction.size)
    summary = {
        **preflight,
        "protocol": (
            "all_market_locked_2025_frozen_foundation_model"
            if locked_2025 else
            "frozen_foundation_model_same_information_contract"
        ),
        "evaluation_stage": args.evaluation_stage,
        "selection_allowed": args.evaluation_stage == "inner",
        "holdout_market": args.holdout_market,
        "project_training_markets": [],
        "external_pretraining_data_disclosure": "unknown_beyond_model_card",
        "uses_node_id_for_prediction": False,
        "uses_market_id_for_prediction": False,
        "uses_future_true_price": False,
        "uses_future_true_load_wind_solar": False,
        "uses_published_future_da_for_rt": args.task == "rt",
        "config": cfg_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256(cfg_path),
        "worker_sha256": _sha256(worker_path),
        "data_read_end_utc_exclusive": read_end_utc,
        "nodes": len(nodes),
        "days": int(prediction.shape[1]),
        "node_macro_mae": float(np.mean([item["mae"] for item in per_node.values()])),
        "node_macro_rmse": float(np.mean([item["rmse"] for item in per_node.values()])),
        "node_macro_bias": float(np.mean([item["bias"] for item in per_node.values()])),
        "central_80_coverage": float(np.mean((target >= q10) & (target <= q90))),
        "per_node": per_node,
        "wall_seconds": wall_seconds,
        "prediction_elements": prediction_elements,
        "seconds_per_prediction_element": wall_seconds / prediction_elements,
        "smoke": bool(args.smoke),
        "final_profit_claim_allowed": False,
        "note": "预测层结果；ERCOT盈利需将三个任务共同送入同一LP/MPC。",
    }
    arrays: dict[str, Any] = {
        "nodes": np.asarray(nodes),
        "dates": common_dates,
        "prediction": prediction.astype(np.float32),
        "q10": q10.astype(np.float32),
        "q90": q90.astype(np.float32),
        "target": target.astype(np.float32),
        "metadata": json.dumps(summary, ensure_ascii=False, sort_keys=True),
    }
    if target_da_values:
        arrays["target_da"] = np.stack(target_da_values).astype(np.float32)
    if args.task == "rt":
        arrays["executed_timestamps"] = common_timestamps
        arrays["timezone"] = np.asarray(market_timezone(args.holdout_market))
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

