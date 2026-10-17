#!/usr/bin/env python
"""四市场最终 pooled XGBoost：训练与锁定2025评价严格分离。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.baselines_global import (  # noqa: E402
    fit_xgboost_multioutput,
    prediction_metrics,
    restore_portable_xgboost_prediction,
)
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    MARKET_ORDER,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.loader_v2 import market_timezone  # noqa: E402
from scripts.decision_aware.export_global_f0_f1_predictions import (  # noqa: E402
    _dates_and_timestamps,
)
from scripts.decision_aware.run_universal_lomo_xgboost import (  # noqa: E402
    _arrays,
    _node_samples,
    _training_samples,
)


TASKS = ("da", "rt_at_da", "rt")
OBJECTIVES = {
    "da": "reg:absoluteerror",
    "rt_at_da": "reg:pseudohubererror",
    "rt": "reg:absoluteerror",
}
FINAL_READ_END_UTC = {
    "CAISO": "2025-01-01T08:00:00+00:00",
    "ERCOT": "2025-01-01T06:00:00+00:00",
    "NYISO": "2025-01-01T05:00:00+00:00",
    "PJM": "2025-01-01T05:00:00+00:00",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(cfg: PilotConfig) -> None:
    markets = tuple(str(item).upper() for item in cfg.global_training_markets)
    if set(markets) != set(MARKET_ORDER):
        raise ValueError("最终pooled XGBoost必须使用全部四个市场")
    if cfg.global_use_entity_embeddings:
        raise ValueError("pooled XGBoost禁止实体ID")
    if str(cfg.global_normalization_mode).lower() != "context_robust":
        raise ValueError("pooled XGBoost必须使用context_robust窗口合同")


def _train(args: argparse.Namespace, cfg: PilotConfig) -> None:
    checkpoint = ROOT / args.checkpoint
    if checkpoint.exists() and not args.force:
        raise FileExistsError(f"checkpoint已存在: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    entities = [
        (mappings.node_to_market[node], node) for node in mappings.node_to_id
    ]
    if args.smoke:
        entities = [
            next(entity for entity in entities if entity[0] == market)
            for market in MARKET_ORDER
        ]
    train, _ = build_global_task_datasets(
        cfg,
        args.task,
        entities,
        mappings,
        train_stage=AnnualStage.FINAL_TRAIN,
        validation_stage=AnnualStage.OUTER_VALIDATION,
        read_end_utc=FINAL_READ_END_UTC,
    )
    samples_per_market = 8 if args.smoke else (
        args.samples_per_market or int(cfg.global_samples_per_market)
    )
    if samples_per_market < 1:
        raise ValueError("samples_per_market必须为正")
    train_x, train_y, _, _ = _arrays(
        _training_samples(
            train,
            args.task,
            samples_per_market=samples_per_market,
            seed=args.seed,
        ),
        args.task,
    )
    started = perf_counter()
    estimator = fit_xgboost_multioutput(
        train_x,
        train_y,
        seed=args.seed,
        n_estimators=3 if args.smoke else args.n_estimators,
        objective=OBJECTIVES[args.task],
        device=args.device,
        n_jobs=args.n_jobs,
        output_jobs=args.output_jobs,
    )
    devices = sorted({
        json.loads(item.get_booster().save_config())["learner"]["generic_param"]["device"]
        for item in estimator.estimators_
    })
    if args.device == "cuda" and not any(item.startswith("cuda") for item in devices):
        raise RuntimeError("请求CUDA XGBoost但实际设备不是CUDA")
    metadata = {
        "protocol": "all_market_pooled_xgboost_final_train",
        "model": "pooled_xgboost",
        "task": args.task,
        "phase": "final",
        "seed": args.seed,
        "training_markets": list(MARKET_ORDER),
        "training_years": [2020, 2021, 2022, 2023, 2024],
        "locked_2025_read": False,
        "read_end_utc_exclusive_by_market": FINAL_READ_END_UTC,
        "selection_allowed": False,
        "uses_node_id_for_prediction": False,
        "uses_market_id_for_prediction": False,
        "identity_used_only_for_balanced_sampling": True,
        "normalization_mode": "context_robust",
        "target_parameterization": "history_centered_scaled_then_inverse",
        "objective": OBJECTIVES[args.task],
        "n_estimators": 3 if args.smoke else args.n_estimators,
        "samples_per_market": samples_per_market,
        "training_sample_count": int(len(train_x)),
        "feature_count": int(train_x.shape[1]),
        "device_requested": args.device,
        "devices_actual": devices,
        "fit_seconds": float(perf_counter() - started),
        "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
        "entity_mapping_hash": mappings.mapping_hash,
        "config": Path(args.config).as_posix(),
        "smoke_only": bool(args.smoke),
    }
    joblib.dump({"estimator": estimator, "metadata": metadata}, checkpoint)
    summary = {**metadata, "checkpoint": checkpoint.relative_to(ROOT).as_posix()}
    summary["checkpoint_sha256"] = _sha256(checkpoint)
    checkpoint.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _evaluate(args: argparse.Namespace, cfg: PilotConfig) -> None:
    checkpoint = ROOT / args.checkpoint
    output = ROOT / args.output
    if output.exists() and not args.force:
        raise FileExistsError(f"输出已存在: {output}")
    payload = joblib.load(checkpoint)
    if not isinstance(payload, dict) or "estimator" not in payload:
        raise ValueError("最终pooled XGBoost checkpoint格式错误")
    frozen = payload.get("metadata", {})
    expected = {
        "protocol": "all_market_pooled_xgboost_final_train",
        "task": args.task,
        "phase": "final",
        "seed": args.seed,
        "locked_2025_read": False,
        "selection_allowed": False,
    }
    mismatches = {
        key: (frozen.get(key), value)
        for key, value in expected.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise ValueError(f"pooled XGBoost冻结元数据不匹配: {mismatches}")
    market = args.market.upper()
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset([market])
    nodes = list(mappings.node_to_id)
    if args.smoke:
        nodes = nodes[:1]
    entities = [(market, node) for node in nodes]
    end = FORMAL_EXPERIMENT_CONTRACT.annual_split.range_for(
        AnnualStage.LOCKED_RETROSPECTIVE_TEST
    ).end
    read_end_utc = (
        pd.Timestamp(end, tz=market_timezone(market)) + pd.Timedelta(days=1)
    ).tz_convert("UTC").isoformat()
    _, validation = build_global_task_datasets(
        replace(cfg, global_use_critical_event_weighting=True),
        args.task,
        entities,
        mappings,
        train_stage=AnnualStage.FINAL_TRAIN,
        validation_stage=AnnualStage.LOCKED_RETROSPECTIVE_TEST,
        read_end_utc=read_end_utc,
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    targets_da: list[np.ndarray] = []
    common_dates: np.ndarray | None = None
    common_timestamps: np.ndarray | None = None
    per_node: dict[str, dict[str, float]] = {}
    inference_started = perf_counter()
    for (_, node), node_dataset in zip(validation.entities, validation.datasets):
        x, y, centers, scales = _arrays(_node_samples(node_dataset, args.task), args.task)
        if args.smoke:
            limit = 24 if args.task == "rt" else 2
            x, y, centers, scales = x[:limit], y[:limit], centers[:limit], scales[:limit]
        normalized = payload["estimator"].predict(x).astype(np.float32)
        prediction = restore_portable_xgboost_prediction(normalized, centers, scales)
        target = restore_portable_xgboost_prediction(y, centers, scales)
        if args.task == "rt":
            prediction = prediction.reshape(-1, 24, prediction.shape[-1])
            target = target.reshape(-1, 24, target.shape[-1])
        dates, timestamps = _dates_and_timestamps(node_dataset, args.task)
        if args.smoke:
            dates = dates[:1] if args.task == "rt" else dates[:2]
            if timestamps is not None:
                timestamps = timestamps[:1] if args.task == "rt" else timestamps[:2]
        if common_dates is None:
            common_dates, common_timestamps = dates, timestamps
        elif not np.array_equal(common_dates, dates):
            raise ValueError(f"节点日期不一致: {node}")
        metrics = prediction_metrics(target, prediction)
        per_node[node] = {
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "samples": metrics.samples,
        }
        predictions.append(prediction)
        targets.append(target)
        if args.task == "rt_at_da":
            targets_da.append(np.stack([
                node_dataset[index]["price_da_tgt_weighting_only"].numpy()
                for index in range(len(dates))
            ]).astype(np.float32))
    inference_seconds = float(perf_counter() - inference_started)
    prediction_elements = int(sum(item.size for item in predictions))
    summary = {
        "protocol": "all_market_pooled_xgboost_locked_2025",
        "model": "pooled_xgboost",
        "task": args.task,
        "evaluation_market": market,
        "evaluation_stage": "locked_2025",
        "selection_allowed": False,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "data_read_end_utc_exclusive": read_end_utc,
        "nodes": len(nodes),
        "node_macro_mae": float(np.mean([value["mae"] for value in per_node.values()])),
        "node_macro_rmse": float(np.mean([value["rmse"] for value in per_node.values()])),
        "per_node": per_node,
        "inference_seconds": inference_seconds,
        "prediction_elements": prediction_elements,
        "seconds_per_prediction_element": inference_seconds / prediction_elements,
        "smoke_only": bool(args.smoke),
        "final_profit_claim_allowed": False,
    }
    arrays = {
        "nodes": np.asarray(nodes),
        "dates": common_dates,
        "prediction": np.stack(predictions).astype(np.float32),
        "target": np.stack(targets).astype(np.float32),
        "metadata": json.dumps(summary, ensure_ascii=False, sort_keys=True),
    }
    if targets_da:
        arrays["target_da"] = np.stack(targets_da).astype(np.float32)
    if args.task == "rt":
        arrays["executed_timestamps"] = common_timestamps
        arrays["timezone"] = np.asarray(market_timezone(market))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("train", "locked_2025"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--market", choices=tuple(MARKET_ORDER))
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-market", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = PilotConfig.from_yaml(ROOT / args.config)
    _validate_config(cfg)
    if args.n_estimators < 1:
        raise ValueError("--n-estimators必须至少为1")
    if args.mode == "train":
        if args.market or args.output:
            raise ValueError("训练模式不接受--market/--output")
        _train(args, cfg)
    else:
        if not args.market or not args.output:
            raise ValueError("locked_2025必须提供--market和--output")
        _evaluate(args, cfg)


if __name__ == "__main__":
    main()
