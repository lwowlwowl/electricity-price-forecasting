#!/usr/bin/env python
"""Summarize one frozen continuous-settlement candidate against one baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _label(family: str) -> str:
    return f"da_{family}__rt_at_da_{family}__rt_{family}"


def _block_bootstrap_ci(
    values: np.ndarray, *, block: int = 7, draws: int = 5000, seed: int = 20261024
) -> list[float]:
    generator = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - block + 1))
    blocks = int(np.ceil(len(values) / block))
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        chosen = generator.choice(starts, size=blocks, replace=True)
        sample = np.concatenate([values[start:start + block] for start in chosen])
        means[index] = sample[:len(values)].mean()
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(report: dict, candidate_family: str, baseline_family: str) -> dict:
    candidate = report["systems"][_label(candidate_family)]
    baseline = report["systems"][_label(baseline_family)]
    dates = sorted(set(candidate["macro_daily_net_revenue"]) & set(
        baseline["macro_daily_net_revenue"]
    ))
    difference = np.asarray([
        candidate["macro_daily_net_revenue"][day]
        - baseline["macro_daily_net_revenue"][day]
        for day in dates
    ], dtype=np.float64)
    node_differences = {
        node: (
            candidate["nodes"][node]["mean_daily_revenue"]
            - baseline["nodes"][node]["mean_daily_revenue"]
        )
        for node in sorted(candidate["nodes"])
    }
    keep = np.ones(len(difference), dtype=bool)
    keep[np.argsort(difference)[-min(5, len(difference)):]] = False
    worst_count = max(1, int(np.ceil(0.10 * len(difference))))
    baseline_daily = np.asarray([
        baseline["macro_daily_net_revenue"][day] for day in dates
    ])
    baseline_worst = np.argsort(baseline_daily)[:worst_count]
    interval = _block_bootstrap_ci(difference)
    return {
        "candidate_family": candidate_family,
        "baseline_family": baseline_family,
        "days": len(dates),
        "nodes": len(node_differences),
        "candidate_macro_mean_daily_revenue": candidate["macro_mean_daily_revenue"],
        "baseline_macro_mean_daily_revenue": baseline["macro_mean_daily_revenue"],
        "mean_daily_difference": float(difference.mean()),
        "relative_improvement_fraction": float(
            difference.mean() / baseline["macro_mean_daily_revenue"]
        ),
        "paired_7day_block_bootstrap_95ci": interval,
        "improved_nodes": int(sum(value > 0.0 for value in node_differences.values())),
        "node_differences": node_differences,
        "difference_after_removing_5_best_delta_days": float(difference[keep].mean()),
        "difference_on_baseline_worst_10pct_days": float(difference[baseline_worst].mean()),
        "passes_point_estimate": bool(difference.mean() > 0.0),
        "passes_paired_interval": bool(interval[0] > 0.0),
        "passes_remove_top5": bool(difference[keep].mean() > 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate-family", required=True)
    parser.add_argument("--baseline-family", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = ROOT / args.input
    report = json.loads(input_path.read_text(encoding="utf-8"))
    result = summarize(report, args.candidate_family, args.baseline_family)
    result.update({
        "year": args.year,
        "selection_allowed": bool(report.get("selection_allowed", False)),
        "source": input_path.relative_to(ROOT).as_posix(),
    })
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "paired_summary_saved", "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()

