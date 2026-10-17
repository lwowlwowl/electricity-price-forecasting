#!/usr/bin/env python
"""汇总一次性2025预测、尾部与价差指标；结果只报告，不选模。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.tail_diagnostics import (  # noqa: E402
    fit_tail_thresholds,
    price_slice_report,
    spread_slice_report,
)
from scripts.decision_aware.diagnose_tail_and_profit_instability import _target_history  # noqa: E402
from scripts.decision_aware.evaluate_global_continuous_mpc import _block_bootstrap_ci  # noqa: E402


MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")
TASKS = ("da", "rt_at_da", "rt")
LOCAL_MODEL = {"da": "xgboost_absolute", "rt_at_da": "xgboost_absolute", "rt": "xgboost_absolute"}


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _same_numeric(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.allclose(
        left, right, rtol=0.0, atol=1e-5, equal_nan=True
    ):
        raise ValueError(f"{label}不一致")


def _validate_archive_alignment(
    reference: dict, item: dict, *, system: str, market: str, task: str
) -> None:
    label = f"{system}/{market}/{task}"
    required = {"nodes", "dates", "prediction", "target"}
    missing = sorted(required.difference(item))
    if missing:
        raise ValueError(f"{label}缺少字段: {missing}")
    if item["prediction"].shape != item["target"].shape:
        raise ValueError(f"{label}预测与真值形状不一致")
    if not np.array_equal(reference["nodes"], item["nodes"]):
        raise ValueError(f"{label}节点不一致")
    if not np.array_equal(reference["dates"], item["dates"]):
        raise ValueError(f"{label}日期不一致")
    _same_numeric(reference["target"], item["target"], f"{label}真值")
    for key in ("target_da", "executed_timestamps", "timezone"):
        if key not in reference:
            continue
        if key not in item:
            raise ValueError(f"{label}缺少对齐字段{key}")
        if key == "target_da":
            _same_numeric(reference[key], item[key], f"{label}/{key}")
        elif not np.array_equal(reference[key], item[key]):
            raise ValueError(f"{label}/{key}不一致")


def source_path(args, system: str, market: str, task: str) -> Path:
    lower = market.lower()
    if system.startswith("transformer_seed"):
        seed = int(system.removeprefix("transformer_seed"))
        return ROOT / args.transformer_dir / f"{lower}_{task}_seed{seed}.npz"
    if system == "transformer_ensemble":
        return ROOT / args.ensemble_dir / f"{lower}_{task}_ensemble.npz"
    if system in {"timesfm3", "timesfm25", "pooled_xgboost", "dlinear"}:
        return ROOT / args.baseline_dir / system / f"{lower}_{task}.npz"
    market_subdir = Path() if market == "ERCOT" else Path(lower)
    model = LOCAL_MODEL[task] if system == "per_node_xgboost" else system
    return (ROOT / "data/results/global_baselines_v1" / market_subdir /
            f"{task}_locked_2025" / f"{model}_{lower}_{task}_locked_2025.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-dir", default="data/results/allmarket_locked_2025_predictions_v1")
    parser.add_argument("--ensemble-dir", default="data/results/allmarket_locked_2025_ensemble_v1")
    parser.add_argument("--baseline-dir", default="data/results/allmarket_locked_2025_baselines_v1")
    parser.add_argument("--settlement", default="data/results/allmarket_locked_2025_all_continuous_mpc_v2.json")
    parser.add_argument("--output", default="data/results/allmarket_locked_2025_summary_v2.json")
    args = parser.parse_args()
    systems = (
        "transformer_seed0", "transformer_seed1", "transformer_seed2",
        "transformer_ensemble", "timesfm3", "timesfm25", "pooled_xgboost",
        "per_node_xgboost", "dlinear", "seasonal", "lear",
    )
    report = {
        "protocol": "allmarket_locked_2025_report_only_v1",
        "year": 2025,
        "selection_allowed": False,
        "threshold_source": "market_2020_2024_truth_only",
        "markets": {},
    }
    for market in MARKETS:
        references = {
            task: load(source_path(args, "transformer_ensemble", market, task))
            for task in TASKS
        }
        for task, reference in references.items():
            _validate_archive_alignment(
                reference, reference, system="transformer_ensemble", market=market, task=task
            )
        train_da, train_rt = _target_history(
            market, references["da"]["nodes"].astype(str), 2024
        )
        thresholds = fit_tail_thresholds(train_da, train_rt)
        market_report = {"thresholds": thresholds.to_dict(), "systems": {}}
        for system in systems:
            task_items = {}
            for task in TASKS:
                if system == "lear" and task == "rt":
                    continue
                path = source_path(args, system, market, task)
                if not path.is_file():
                    raise FileNotFoundError(path)
                item = load(path)
                _validate_archive_alignment(
                    references[task], item, system=system, market=market, task=task
                )
                lower = thresholds.da_lower_1pct if task == "da" else thresholds.rt_lower_1pct
                upper = thresholds.da_upper_1pct if task == "da" else thresholds.rt_upper_1pct
                task_items[task] = item
                market_report["systems"].setdefault(system, {})[task] = price_slice_report(
                    item["prediction"], item["target"], lower_1pct=lower, upper_1pct=upper
                )
            if "da" in task_items and "rt_at_da" in task_items:
                market_report["systems"][system]["da_rt_at_da_spread"] = spread_slice_report(
                    task_items["da"]["prediction"], task_items["rt_at_da"]["prediction"],
                    task_items["da"]["target"], task_items["rt_at_da"]["target"], thresholds,
                )
        report["markets"][market] = market_report
    settlement_path = ROOT / args.settlement
    settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
    system_results = settlement["systems"]
    oracle_label = "da_oracle__rt_at_da_oracle__rt_oracle"
    transformer_label = "da_ensemble__rt_at_da_ensemble__rt_ensemble"
    oracle_revenue = float(system_results[oracle_label]["macro_mean_daily_revenue"])
    transformer_daily = system_results[transformer_label]["macro_daily_net_revenue"]
    derived = {}
    for label, value in system_results.items():
        daily_map = value["macro_daily_net_revenue"]
        ordered_days = sorted(daily_map)
        daily = np.asarray([daily_map[day] for day in ordered_days], dtype=np.float64)
        sorted_daily = np.sort(daily)
        cumulative = np.cumsum(daily)
        peaks = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
        item = {
            "macro_mean_daily_revenue": float(daily.mean()),
            "fraction_of_oracle": (
                float(daily.mean() / oracle_revenue) if abs(oracle_revenue) > 1e-12 else None
            ),
            "macro_maximum_drawdown_usd": float(np.max(peaks - cumulative)),
            "macro_mean_daily_revenue_without_top_5_days": (
                float(np.mean(sorted_daily[:-5])) if len(sorted_daily) > 5 else None
            ),
            "macro_mean_daily_revenue_without_top_10_days": (
                float(np.mean(sorted_daily[:-10])) if len(sorted_daily) > 10 else None
            ),
        }
        if label != transformer_label:
            if sorted(transformer_daily) != ordered_days:
                raise ValueError(f"{label}与Transformer的ERCOT结算日期不一致")
            difference = np.asarray([
                transformer_daily[day] - daily_map[day] for day in ordered_days
            ], dtype=np.float64)
            item["transformer_ensemble_minus_system"] = {
                "mean_daily_difference": float(difference.mean()),
                "paired_7day_block_bootstrap_95ci": _block_bootstrap_ci(difference),
            }
        derived[label] = item
    seed_labels = [
        "da_seed0__rt_at_da_seed0__rt_seed0",
        "da_seed1__rt_at_da_seed1__rt_seed1",
        "da_seed2__rt_at_da_seed2__rt_seed2",
    ]
    seed_revenue = np.asarray([
        system_results[label]["macro_mean_daily_revenue"] for label in seed_labels
    ], dtype=np.float64)
    report["ercot_transformer_seed_stability"] = {
        "seed_values": dict(zip(("seed0", "seed1", "seed2"), seed_revenue.tolist())),
        "mean": float(seed_revenue.mean()),
        "sample_std": float(seed_revenue.std(ddof=1)),
        "minimum": float(seed_revenue.min()),
        "maximum": float(seed_revenue.max()),
        "fixed_ensemble": float(system_results[transformer_label]["macro_mean_daily_revenue"]),
        "selection_allowed": False,
    }
    report["ercot_settlement_derived"] = derived
    report["ercot_settlement"] = settlement
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "locked_2025_summary_complete", "output": output.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()
