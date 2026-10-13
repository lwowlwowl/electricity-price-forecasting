#!/usr/bin/env python
"""按市场专属配置与正式年度协议运行逐节点Seasonal/LEAR/XGBoost基线。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.baselines_global import (  # noqa: E402
    fit_xgboost_multioutput,
    node_datasets,
    prediction_metrics,
    rolling_cutoff_safe_lear_predictions,
    weekly_seasonal_node_predictions,
    xgboost_arrays,
)
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    load_formal_entity_mappings,
)


PHASES = {
    "inner": (AnnualStage.INNER_TRAIN, AnnualStage.INNER_VALIDATION),
    "outer": (AnnualStage.OUTER_TRAIN, AnnualStage.OUTER_VALIDATION),
}


_UNSAFE_FILENAME_CHARS = frozenset('<>:"/\\|?*%')
_WINDOWS_RESERVED_STEMS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _portable_node_key(node: str) -> str:
    """把市场节点名转换成可跨Windows/Linux安全复制的文件名片段。"""
    if not node:
        raise ValueError("节点名不能为空")
    parts: list[str] = []
    for character in node:
        if character in _UNSAFE_FILENAME_CHARS or ord(character) < 32:
            parts.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
        else:
            parts.append(character)
    key = "".join(parts)
    stem = key.split(".", 1)[0].upper()
    if key in {".", ".."} or key.endswith((" ", ".")) or stem in _WINDOWS_RESERVED_STEMS:
        digest = hashlib.sha256(node.encode("utf-8")).hexdigest()[:16]
        key = f"node_{digest}"
    return key


def _macro(node_reports: dict[str, dict]) -> dict:
    return {
        "macro_mae": float(np.mean([item["mae"] for item in node_reports.values()])),
        "macro_rmse": float(np.mean([item["rmse"] for item in node_reports.values()])),
        "node_count": len(node_reports),
    }


def _cached_predictions(
    path: Path,
    *,
    model_name: str,
    node: str,
    args: argparse.Namespace,
    cfg: PilotConfig,
    mappings,
) -> tuple[np.ndarray, np.ndarray] | None:
    """读取一个可确认属于当前实验的节点缓存；不确定时重新计算。"""
    if args.force or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            prediction = archive["prediction"]
            target = archive["target"]
            if prediction.shape != target.shape or prediction.size == 0:
                return None
            if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
                return None
            required = {
                "model": model_name,
                "market": args.market,
                "task": args.task,
                "phase": args.phase,
                "node": node,
                "seed": str(args.seed),
                "cache_schema_version": "2",
                "da_issue_hour_local": str(cfg.da_issue_hour_local),
                "context_len": str(cfg.context_len),
                "horizon_da": str(cfg.horizon_da),
                "horizon_rt": str(cfg.horizon_rt),
                "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
                "entity_mapping_hash": mappings.mapping_hash,
            }
            present = set(archive.files)
            if not set(required).issubset(present):
                return None
            for key, expected in required.items():
                actual = str(np.asarray(archive[key]).reshape(()).item())
                if actual != str(expected):
                    return None
            return prediction, target
    except (OSError, ValueError, KeyError):
        return None


def _save_predictions(
    path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    model_name: str,
    node: str,
    args: argparse.Namespace,
    cfg: PilotConfig,
    mappings,
) -> None:
    np.savez_compressed(
        path,
        prediction=prediction,
        target=target,
        model=model_name,
        market=args.market,
        task=args.task,
        phase=args.phase,
        node=node,
        seed=args.seed,
        cache_schema_version=2,
        da_issue_hour_local=cfg.da_issue_hour_local,
        context_len=cfg.context_len,
        horizon_da=cfg.horizon_da,
        horizon_rt=cfg.horizon_rt,
        contract_hash=FORMAL_EXPERIMENT_CONTRACT.contract_hash,
        entity_mapping_hash=mappings.mapping_hash,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_aware/global_ercot_huber_v1.yaml")
    parser.add_argument(
        "--market",
        default="",
        help="市场名；空值使用配置中的market。其他市场须使用已冻结起报时刻的专用配置。",
    )
    parser.add_argument("--task", required=True, choices=("da", "rt_at_da", "rt"))
    parser.add_argument("--phase", default="inner", choices=sorted(PHASES))
    parser.add_argument("--models", default="seasonal,lear,xgboost")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lear-calibration-days", type=int, default=1092)
    parser.add_argument("--lear-refit-every-days", type=int, default=1)
    parser.add_argument(
        "--nodes",
        default="",
        help="可选：逗号分隔的节点名；空值表示全部节点",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行处理的节点数；每个XGBoost估计器内部仍使用单线程",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有的逐节点缓存并重新计算",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers必须至少为1")

    requested = tuple(item.strip().lower() for item in args.models.split(",") if item.strip())
    allowed = {
        "seasonal",
        "lear",
        "xgboost",
        "xgboost_absolute",
        "xgboost_pseudohuber",
    }
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"未知基线: {unknown}")
    if args.task == "rt" and "lear" in requested:
        raise ValueError("LEAR是24小时日前模型，不用于4小时滚动RT；请从--models移除lear")

    cfg = PilotConfig.from_yaml(args.config)
    configured_market = cfg.market.upper()
    args.market = (args.market or configured_market).upper()
    if args.market != configured_market and not args.smoke:
        raise ValueError(
            "正式其他市场基线必须使用market匹配、起报时刻已冻结的专用配置；"
            "跨市场覆盖只允许--smoke"
        )
    cfg = replace(cfg, market=args.market)
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(
        [args.market]
    )
    all_entities = [(args.market, node) for node in mappings.node_to_id]
    requested_nodes = [item.strip() for item in args.nodes.split(",") if item.strip()]
    unknown_nodes = sorted(set(requested_nodes) - set(mappings.node_to_id))
    if unknown_nodes:
        raise ValueError(f"未知节点: {unknown_nodes}")
    if args.smoke and requested_nodes:
        raise ValueError("--smoke与--nodes不能同时使用")
    entities = (
        all_entities[:2]
        if args.smoke
        else [(args.market, node) for node in requested_nodes]
        if requested_nodes
        else all_entities
    )
    train_stage, validation_stage = PHASES[args.phase]
    train_dataset, validation_dataset = build_global_task_datasets(
        cfg,
        args.task,
        entities,
        mappings,
        train_stage=train_stage,
        validation_stage=validation_stage,
    )
    train_by_node = {
        node: dataset for _, node, dataset in node_datasets(train_dataset)
    }
    validation_by_node = {
        node: dataset for _, node, dataset in node_datasets(validation_dataset)
    }

    suffix = "_smoke" if args.smoke else ""
    market_subdir = Path() if args.market == "ERCOT" else Path(args.market.lower())
    output_dir = (
        ROOT / "data" / "results" / "global_baselines_v1" / market_subdir
        / f"{args.task}_{args.phase}{suffix}"
    )
    checkpoint_dir = (
        ROOT / "data" / "checkpoints" / "global_baselines_v1" / market_subdir
        / f"{args.task}_{args.phase}{suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report = {}
    if summary_path.exists() and not args.force:
        try:
            report = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            report = {}
    common_metadata_matches = (
        report.get("task") == args.task
        and report.get("market", "ERCOT") == args.market
        and report.get("phase") == args.phase
        and report.get("smoke_only") == args.smoke
        and report.get("configured_market") == configured_market
        and report.get("da_issue_hour_local") == cfg.da_issue_hour_local
        and report.get("contract_hash") == FORMAL_EXPERIMENT_CONTRACT.contract_hash
        and report.get("entity_mapping_hash") == mappings.mapping_hash
        and report.get("nodes") == [node for _, node in entities]
    )
    if not common_metadata_matches:
        report = {
            "task": args.task,
            "market": args.market,
            "configured_market": configured_market,
            "da_issue_hour_local": cfg.da_issue_hour_local,
            "market_contract_smoke_only": (
                args.smoke and args.market != configured_market
            ),
            "phase": args.phase,
            "smoke_only": args.smoke,
            "nodes": [node for _, node in entities],
            "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
            "entity_mapping_hash": mappings.mapping_hash,
            "models": {},
        }

    for model_name in requested:
        started = perf_counter()
        previous_model_report = report.get("models", {}).get(model_name, {})
        node_reports: dict[str, dict] = dict(previous_model_report.get("nodes", {}))
        reused_nodes: list[str] = []
        computed_nodes: list[str] = []

        def run_node(node: str) -> tuple[str, dict, bool]:
            train_node = train_by_node[node]
            validation_node = validation_by_node[node]
            node_key = _portable_node_key(node)
            result_path = output_dir / f"{model_name}_{node_key}_seed{args.seed}.npz"
            cached = _cached_predictions(
                result_path,
                model_name=model_name,
                node=node,
                args=args,
                cfg=cfg,
                mappings=mappings,
            )
            if cached is not None:
                prediction, target = cached
                reused = True
            elif model_name == "seasonal":
                prediction, target = weekly_seasonal_node_predictions(
                    validation_node, args.task
                )
                if args.smoke:
                    prediction, target = prediction[:3], target[:3]
                reused = False
            elif model_name == "lear":
                prediction, target = rolling_cutoff_safe_lear_predictions(
                    train_node,
                    validation_node,
                    args.task,
                    calibration_window=args.lear_calibration_days,
                    refit_every_days=args.lear_refit_every_days,
                    max_validation_days=1 if args.smoke else None,
                )
                reused = False
            else:
                train_x, train_y = xgboost_arrays(train_node, args.task)
                validation_x, validation_y = xgboost_arrays(
                    validation_node, args.task
                )
                if args.smoke:
                    train_x, train_y = train_x[:200], train_y[:200]
                    validation_x, validation_y = validation_x[:4], validation_y[:4]
                objective = {
                    "xgboost_absolute": "reg:absoluteerror",
                    "xgboost_pseudohuber": "reg:pseudohubererror",
                }.get(model_name, "reg:squarederror")
                estimator = fit_xgboost_multioutput(
                    train_x,
                    train_y,
                    seed=args.seed,
                    n_estimators=5 if args.smoke else 180,
                    objective=objective,
                )
                prediction = estimator.predict(validation_x).astype(np.float32)
                target = validation_y
                joblib.dump(
                    estimator,
                    checkpoint_dir / f"{model_name}_{node_key}_seed{args.seed}.joblib",
                )
                reused = False

            metrics = prediction_metrics(target, prediction)
            node_report = {
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "samples": metrics.samples,
            }
            if not reused:
                _save_predictions(
                    result_path,
                    prediction,
                    target,
                    model_name=model_name,
                    node=node,
                    args=args,
                    cfg=cfg,
                    mappings=mappings,
                )
            return node, node_report, reused

        nodes_to_run = [node for _, node in entities]
        if args.workers == 1:
            completed = (run_node(node) for node in nodes_to_run)
            for node, node_report, reused in completed:
                node_reports[node] = node_report
                (reused_nodes if reused else computed_nodes).append(node)
                print(f"{model_name}: {node} ({'reused' if reused else 'computed'})", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run_node, node): node for node in nodes_to_run}
                for future in as_completed(futures):
                    node, node_report, reused = future.result()
                    node_reports[node] = node_report
                    (reused_nodes if reused else computed_nodes).append(node)
                    print(f"{model_name}: {node} ({'reused' if reused else 'computed'})", flush=True)

        expected_nodes = [node for _, node in entities]
        ordered_node_reports = {
            node: node_reports[node] for node in expected_nodes if node in node_reports
        }
        previous_elapsed = float(previous_model_report.get("elapsed_seconds", 0.0))
        last_run_elapsed = perf_counter() - started
        report["models"][model_name] = {
            **_macro(ordered_node_reports),
            "nodes": ordered_node_reports,
            "complete": len(ordered_node_reports) == len(expected_nodes),
            "elapsed_seconds": previous_elapsed + last_run_elapsed if computed_nodes else previous_elapsed,
            "last_run_elapsed_seconds": last_run_elapsed,
            "last_run_computed_nodes": computed_nodes,
            "last_run_reused_nodes": reused_nodes,
            "protocol": (
                "weekly_lag_168"
                if model_name == "seasonal"
                else "cutoff_safe_lear_daily_recalibration"
                if model_name == "lear"
                else "local_per_node_xgboost_pseudohuber"
                if model_name == "xgboost_pseudohuber"
                else "local_per_node_xgboost_absolute_error"
                if model_name == "xgboost_absolute"
                else "local_per_node_xgboost_squared_error"
            ),
        }
        print(json.dumps({model_name: report["models"][model_name]}, ensure_ascii=False))
        summary_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
