#!/usr/bin/env python
"""训练并评价与无ID Transformer 同年份的 pooled LOMO XGBoost。

每一折只使用三个源市场的2020--2022样本训练，在完整留出市场的2023
（或冻结2024）上评价。市场/节点ID只用于模型外分组和等权采样，从不进入
特征。价格监督先按当前起报点可见的168小时历史中位数/MAD归一化，预测后
再用同一历史窗口还原，因而不需要留出市场训练期scaler。
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.baselines_global import (  # noqa: E402
    fit_xgboost_multioutput,
    portable_xgboost_target,
    prediction_metrics,
    restore_portable_xgboost_prediction,
    xgboost_feature,
)
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    MARKET_ORDER,
    MarketNodeBalancedSampler,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.loader_v2 import market_timezone  # noqa: E402


OBJECTIVES = {
    "da": "reg:absoluteerror",
    "rt_at_da": "reg:pseudohubererror",
    "rt": "reg:absoluteerror",
}
TARGET_KEYS = {
    "da": "price_da_tgt",
    "rt_at_da": "price_rt_at_da_tgt",
    "rt": "price_rt_tgt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hour_from_rt_day(day_sample: dict[str, Any], hour: int) -> dict[str, Any]:
    """从RT交付日抽一个起报小时，并丢弃模型不需要的身份字段。"""
    if not 0 <= hour < 24:
        raise ValueError("RT小时必须位于[0, 23]")
    hourly: dict[str, Any] = {}
    for key, value in day_sample.items():
        if key in {"node_id", "market_id"}:
            continue
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 24:
            hourly[key] = value[hour]
        else:
            hourly[key] = value
    return hourly


def _training_samples(
    dataset,
    task: str,
    *,
    samples_per_market: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """按市场、节点等权抽样；RT预算单位为单个起报小时。"""
    sampler = MarketNodeBalancedSampler(
        dataset, samples_per_market=samples_per_market, seed=seed
    )
    hour_generator = np.random.default_rng(seed + 71_003)
    for index in sampler:
        sample = dataset[index]
        if task == "rt":
            yield _hour_from_rt_day(
                sample, int(hour_generator.integers(0, 24))
            )
        else:
            # ID可以供sampler分组，但不能进入xgboost_feature。
            yield {
                key: value for key, value in sample.items()
                if key not in {"node_id", "market_id"}
            }


def _arrays(samples: Iterator[dict[str, Any]], task: str):
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    centers: list[float] = []
    scales: list[float] = []
    for sample in samples:
        features.append(xgboost_feature(sample, task))
        target, center, scale = portable_xgboost_target(sample, task)
        targets.append(target)
        centers.append(center)
        scales.append(scale)
    if not features:
        raise ValueError("pooled LOMO XGBoost没有可用样本")
    return (
        np.stack(features).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


def _node_samples(node_dataset, task: str) -> Iterator[dict[str, Any]]:
    if task != "rt":
        for index in range(len(node_dataset)):
            yield node_dataset[index]
        return
    for day_index in range(len(node_dataset)):
        day = node_dataset[day_index]
        for hour in range(24):
            yield _hour_from_rt_day(day, hour)


def _dates(node_dataset, task: str) -> np.ndarray:
    if task == "rt":
        return np.asarray([str(day) for day, _ in node_dataset.days])
    return np.asarray([
        str(node_dataset.sample_time_contract(index)["delivery_date"])
        for index in range(len(node_dataset))
    ])


def _default_output(holdout_market: str, task: str, stage: str, seed: int) -> Path:
    return (
        ROOT / "data" / "results" / "universal_lomo_xgboost_v1"
        / f"holdout_{holdout_market.lower()}_{task}_{stage}_seed{seed}.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=tuple(TARGET_KEYS))
    parser.add_argument(
        "--holdout-market", required=True, choices=tuple(MARKET_ORDER)
    )
    parser.add_argument(
        "--evaluation-stage", choices=("inner", "outer_frozen"), default="inner"
    )
    parser.add_argument("--samples-per-market", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-jobs", type=int, default=1)
    parser.add_argument("--output", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(ROOT / args.config)
    training_markets = tuple(str(item).upper() for item in cfg.global_training_markets)
    if args.holdout_market in training_markets:
        raise ValueError("留出市场不能出现在pooled XGBoost训练市场中")
    expected_training = tuple(
        market for market in MARKET_ORDER if market != args.holdout_market
    )
    if set(training_markets) != set(expected_training):
        raise ValueError(
            f"LOMO训练市场必须恰好是{expected_training}，实际为{training_markets}"
        )
    if cfg.global_use_entity_embeddings:
        raise ValueError("公平pooled XGBoost配置必须关闭实体embedding")
    if str(cfg.global_normalization_mode).lower() != "context_robust":
        raise ValueError("公平pooled XGBoost要求context_robust归一化")
    if args.samples_per_market < 0:
        raise ValueError("--samples-per-market不能为负")
    if args.n_estimators < 1:
        raise ValueError("--n-estimators必须至少为1")

    output = (
        ROOT / args.output if args.output
        else _default_output(
            args.holdout_market, args.task, args.evaluation_stage, args.seed
        )
    )
    if output.exists() and not args.force:
        raise FileExistsError(f"结果已存在；如需覆盖请加--force: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    samples_per_market = (
        8 if args.smoke
        else args.samples_per_market or int(cfg.global_samples_per_market)
    )
    n_estimators = 3 if args.smoke else args.n_estimators
    if samples_per_market < 1:
        raise ValueError("正式配置必须冻结正数global_samples_per_market")

    full_mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    source_mappings = full_mappings.subset(training_markets)
    source_entities = [
        (source_mappings.node_to_market[node], node)
        for node in source_mappings.node_to_id
    ]
    if args.smoke:
        source_entities = [
            entity
            for market in training_markets
            for entity in [
                item for item in source_entities if item[0] == market
            ][:2]
        ]
    source_train, _ = build_global_task_datasets(
        cfg,
        args.task,
        source_entities,
        source_mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
    )

    holdout_mappings = full_mappings.subset([args.holdout_market])
    holdout_nodes = list(holdout_mappings.node_to_id)
    if args.smoke:
        holdout_nodes = holdout_nodes[:2]
    holdout_entities = [(args.holdout_market, node) for node in holdout_nodes]
    validation_stage = (
        AnnualStage.OUTER_VALIDATION
        if args.evaluation_stage == "outer_frozen"
        else AnnualStage.INNER_VALIDATION
    )
    evaluation_end = FORMAL_EXPERIMENT_CONTRACT.annual_split.range_for(
        validation_stage
    ).end
    read_end_utc = (
        pd.Timestamp(evaluation_end, tz=market_timezone(args.holdout_market))
        + pd.Timedelta(days=1)
    ).tz_convert("UTC").isoformat()
    evaluation_cfg = replace(cfg, global_use_critical_event_weighting=True)
    _, holdout_validation = build_global_task_datasets(
        evaluation_cfg,
        args.task,
        holdout_entities,
        holdout_mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=validation_stage,
        read_end_utc=read_end_utc,
    )

    started = perf_counter()
    train_x, train_y, _, _ = _arrays(
        _training_samples(
            source_train,
            args.task,
            samples_per_market=samples_per_market,
            seed=args.seed,
        ),
        args.task,
    )
    objective = OBJECTIVES[args.task]
    estimator = fit_xgboost_multioutput(
        train_x,
        train_y,
        seed=args.seed,
        n_estimators=n_estimators,
        objective=objective,
        device=args.device,
        n_jobs=args.n_jobs,
        output_jobs=args.output_jobs,
    )
    actual_devices = sorted({
        json.loads(item.get_booster().save_config())["learner"]["generic_param"][
            "device"
        ]
        for item in estimator.estimators_
    })
    if args.device == "cuda" and not any(
        device.startswith("cuda") for device in actual_devices
    ):
        raise RuntimeError(
            "请求了CUDA XGBoost，但当前xgboost构建实际回退到CPU；"
            "请改用--device cpu或安装GPU构建"
        )
    fit_seconds = perf_counter() - started

    node_predictions: list[np.ndarray] = []
    node_targets: list[np.ndarray] = []
    node_target_da: list[np.ndarray] = []
    per_node_mae: dict[str, float] = {}
    per_node_rmse: dict[str, float] = {}
    common_dates: np.ndarray | None = None
    for (_, node), node_dataset in zip(
        holdout_validation.entities, holdout_validation.datasets
    ):
        validation_x, validation_y, centers, scales = _arrays(
            _node_samples(node_dataset, args.task), args.task
        )
        if args.smoke:
            limit = 48 if args.task == "rt" else 3
            validation_x = validation_x[:limit]
            validation_y = validation_y[:limit]
            centers = centers[:limit]
            scales = scales[:limit]
        normalized_prediction = estimator.predict(validation_x).astype(np.float32)
        prediction = restore_portable_xgboost_prediction(
            normalized_prediction, centers, scales
        )
        target = restore_portable_xgboost_prediction(
            validation_y, centers, scales
        )
        dates = _dates(node_dataset, args.task)
        if args.smoke:
            dates = dates[:2] if args.task == "rt" else dates[:3]
        if args.task == "rt":
            prediction = prediction.reshape(-1, 24, prediction.shape[-1])
            target = target.reshape(-1, 24, target.shape[-1])
        if common_dates is None:
            common_dates = dates
        elif not np.array_equal(common_dates, dates):
            raise ValueError(f"留出市场各节点日期不一致，首次冲突节点={node}")
        metrics = prediction_metrics(target, prediction)
        per_node_mae[node] = metrics.mae
        per_node_rmse[node] = metrics.rmse
        node_predictions.append(prediction)
        node_targets.append(target)
        if args.task == "rt_at_da":
            da_values = np.stack([
                node_dataset[index]["price_da_tgt_weighting_only"].numpy()
                for index in range(len(dates))
            ]).astype(np.float32)
            node_target_da.append(da_values)
        print(json.dumps({
            "event": "holdout_node_complete",
            "market": args.holdout_market,
            "node": node,
            "mae": metrics.mae,
        }, ensure_ascii=False), flush=True)

    summary = {
        "protocol": "pooled_lomo_xgboost_equal_training_years",
        "model": "pooled_lomo_xgboost",
        "task": args.task,
        "holdout_market": args.holdout_market,
        "training_markets": list(training_markets),
        "train_stage": AnnualStage.INNER_TRAIN.value,
        "evaluation_stage": args.evaluation_stage,
        "data_read_end_utc_exclusive": read_end_utc,
        "uses_target_market_training_labels": False,
        "uses_node_id_for_prediction": False,
        "uses_market_id_for_prediction": False,
        "identity_used_only_for_balanced_sampling": True,
        "normalization_mode": "context_robust",
        "target_parameterization": "history_centered_scaled_then_inverse",
        "history_steps_in_features": 24,
        "samples_per_source_market": samples_per_market,
        "rt_sampling_unit": "one_random_issue_hour_per_balanced_draw",
        "objective": objective,
        "n_estimators": n_estimators,
        "device_requested": args.device,
        "devices_actual": actual_devices,
        "n_jobs": args.n_jobs,
        "output_jobs": args.output_jobs,
        "seed": args.seed,
        "feature_count": int(train_x.shape[1]),
        "training_sample_count": int(train_x.shape[0]),
        "fit_seconds": fit_seconds,
        "node_macro_mae": float(np.mean(list(per_node_mae.values()))),
        "node_macro_rmse": float(np.mean(list(per_node_rmse.values()))),
        "per_node_mae": per_node_mae,
        "per_node_rmse": per_node_rmse,
        "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
        "source_entity_mapping_hash": source_mappings.mapping_hash,
        "holdout_entity_mapping_hash": holdout_mappings.mapping_hash,
        "config": Path(args.config).as_posix(),
        "smoke_only": args.smoke,
        "selection_allowed": args.evaluation_stage == "inner",
        "final_profit_claim_allowed": False,
    }
    checkpoint = output.with_suffix(".joblib")
    joblib.dump({"estimator": estimator, "metadata": summary}, checkpoint)
    summary["checkpoint"] = checkpoint.relative_to(ROOT).as_posix()
    summary["checkpoint_sha256"] = _sha256(checkpoint)
    arrays: dict[str, Any] = {
        "nodes": np.asarray(holdout_nodes),
        "dates": common_dates,
        "prediction": np.stack(node_predictions),
        "target": np.stack(node_targets),
        "metadata": json.dumps(summary, ensure_ascii=False, sort_keys=True),
    }
    if node_target_da:
        arrays["target_da"] = np.stack(node_target_da)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
