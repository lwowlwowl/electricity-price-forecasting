#!/usr/bin/env python
"""Attribute economic-spread-coordination profit to interpretable tail slices."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.audit_peak_dynamic_actions import (  # noqa: E402
    _node_input,
    _worker,
)
from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    _thresholds,
    _target_history,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import _load_archive  # noqa: E402
from decision_aware.tail_diagnostics import fit_tail_thresholds  # noqa: E402


TASKS = ("da", "rt_at_da", "rt")
FIELDS = ("da_leg", "rt_deviation_leg", "degradation_cost", "deviation_penalty", "net_revenue")


def _rows_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    keys = (
        "realized_da_price", "realized_rt_price", "predicted_da_price",
        "predicted_rt_at_da_price", "da_plan_mwh", "actual_dispatch_mwh",
        *FIELDS,
    )
    return {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in keys}


def _exclusive_masks(hourly: dict[str, np.ndarray], thresholds) -> dict[str, np.ndarray]:
    da = hourly["realized_da_price"]
    rt = hourly["realized_rt_price"]
    absolute_spread = np.abs(da - rt)
    remaining = np.isfinite(da) & np.isfinite(rt)
    masks = {}
    masks["negative_price"] = remaining & ((da < 0.0) | (rt < 0.0))
    remaining &= ~masks["negative_price"]
    masks["positive_spike"] = remaining & (
        (da >= thresholds.da_upper_1pct) | (rt >= thresholds.rt_upper_1pct)
    )
    remaining &= ~masks["positive_spike"]
    masks["absolute_spread_top_1pct"] = remaining & (
        absolute_spread >= thresholds.abs_spread_upper_1pct
    )
    remaining &= ~masks["absolute_spread_top_1pct"]
    masks["absolute_spread_top_5pct_remainder"] = remaining & (
        absolute_spread >= thresholds.abs_spread_upper_5pct
    )
    remaining &= ~masks["absolute_spread_top_5pct_remainder"]
    masks["ordinary"] = remaining
    return masks


def _slice_report(
    baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray], masks: dict[str, np.ndarray],
    node_days: int,
) -> dict:
    result = {}
    baseline_state = np.sign(baseline["da_plan_mwh"])
    candidate_state = np.sign(candidate["da_plan_mwh"])
    changed = (baseline_state != candidate_state) | (
        np.abs(baseline["da_plan_mwh"] - candidate["da_plan_mwh"]) > 1.0e-6
    )
    for name, mask in masks.items():
        record = {
            "hours": int(mask.sum()),
            "hour_share": float(mask.mean()),
            "changed_da_plan_hours": int(np.sum(mask & changed)),
        }
        for field in FIELDS:
            baseline_value = float(np.sum(baseline[field][mask]) / node_days)
            candidate_value = float(np.sum(candidate[field][mask]) / node_days)
            record[field] = {
                "baseline_per_node_day": baseline_value,
                "candidate_per_node_day": candidate_value,
                "difference_per_node_day": candidate_value - baseline_value,
            }
        result[name] = record
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--phase", choices=("inner", "outer", "final"), required=True)
    parser.add_argument("--threshold-phase", choices=("inner", "outer"))
    parser.add_argument("--threshold-end-year", type=int)
    parser.add_argument("--baseline-family", default="current")
    parser.add_argument("--candidate-family", default="econcoord")
    parser.add_argument("--config", default="configs/decision_aware/global_ercot_huber_v1.yaml")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    directory = ROOT / args.prediction_dir
    archives = {
        (family, task): _load_archive(directory, family, task, args.phase)
        for family in (args.baseline_family, args.candidate_family)
        for task in TASKS
    }
    nodes = archives[(args.baseline_family, "da")]["nodes"]
    reports = {args.baseline_family: {}, args.candidate_family: {}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for index, node in enumerate(nodes):
            inputs = {
                family: _node_input(archives, family, index)
                for family in reports
            }
            jobs.append(pool.submit(_worker, args.config, str(node), inputs))
        for completed, future in enumerate(as_completed(jobs), start=1):
            node, node_reports = future.result()
            for family, report in node_reports.items():
                reports[family][node] = report
            print(json.dumps({"event": "node_complete", "completed": completed, "total": len(jobs)}))

    all_rows = {}
    for family in reports:
        rows = []
        for node in sorted(reports[family]):
            for row in reports[family][node]["hourly"]:
                rows.append({"node": node, **row})
        all_rows[family] = rows
    baseline_rows = all_rows[args.baseline_family]
    candidate_rows = all_rows[args.candidate_family]
    baseline_keys = [(row["node"], row["timestamp_local"]) for row in baseline_rows]
    candidate_keys = [(row["node"], row["timestamp_local"]) for row in candidate_rows]
    if baseline_keys != candidate_keys:
        raise ValueError("baseline and candidate hourly rows are not aligned")
    baseline = _rows_to_arrays(baseline_rows)
    candidate = _rows_to_arrays(candidate_rows)
    if (args.threshold_phase is None) == (args.threshold_end_year is None):
        raise ValueError("set exactly one of --threshold-phase or --threshold-end-year")
    if args.threshold_end_year is not None:
        historical_da, historical_rt = _target_history(
            "ERCOT", nodes, args.threshold_end_year
        )
        thresholds = fit_tail_thresholds(historical_da, historical_rt)
        threshold_source = f"2020-{args.threshold_end_year}"
    else:
        thresholds = _thresholds("ERCOT", nodes, args.threshold_phase)
        threshold_source = args.threshold_phase
    masks = _exclusive_masks(baseline, thresholds)
    days = sum(report["days"] for report in reports[args.baseline_family].values())
    slices = _slice_report(baseline, candidate, masks, days)
    total_delta = sum(item["net_revenue"]["difference_per_node_day"] for item in slices.values())
    result = {
        "experiment": "economic_spread_coordination_tail_profit_attribution_v1",
        "selection_allowed": False,
        "phase": args.phase,
        "threshold_training_phase": threshold_source,
        "thresholds": thresholds.to_dict(),
        "exclusive_priority": [
            "negative_price", "positive_spike", "absolute_spread_top_1pct",
            "absolute_spread_top_5pct_remainder", "ordinary",
        ],
        "node_count": len(nodes),
        "summed_node_days": days,
        "slices": slices,
        "reconstructed_total_net_difference_per_node_day": float(total_delta),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
