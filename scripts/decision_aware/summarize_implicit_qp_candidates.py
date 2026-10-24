#!/usr/bin/env python
"""Summarize implicit-QP candidates against the frozen pooled LOMO XGBoost."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _block_bootstrap_ci(
    values: np.ndarray,
    *,
    block: int = 7,
    draws: int = 5000,
    seed: int = 20261023,
) -> list[float]:
    generator = np.random.default_rng(seed)
    n = len(values)
    starts = np.arange(max(1, n - block + 1))
    means = np.empty(draws, dtype=np.float64)
    blocks_needed = int(np.ceil(n / block))
    for index in range(draws):
        selected = generator.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block] for start in selected])[:n]
        means[index] = sample.mean()
    return np.quantile(means, [0.025, 0.975]).tolist()


def _pure_label(family: str) -> str:
    return f"da_{family}__rt_at_da_{family}__rt_{family}"


def summarize(paths: list[Path]) -> dict:
    rows = []
    baseline_label = _pure_label("xgbpooled")
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        family = path.stem.replace("_continuous_mpc", "")
        candidate_label = _pure_label(family)
        systems = report["systems"]
        candidate = systems[candidate_label]
        baseline = systems[baseline_label]
        dates = sorted(set(candidate["macro_daily_net_revenue"]) & set(
            baseline["macro_daily_net_revenue"]
        ))
        difference = np.asarray([
            candidate["macro_daily_net_revenue"][day]
            - baseline["macro_daily_net_revenue"][day]
            for day in dates
        ], dtype=np.float64)
        candidate_nodes = candidate["nodes"]
        baseline_nodes = baseline["nodes"]
        node_differences = {
            node: (
                candidate_nodes[node]["mean_daily_revenue"]
                - baseline_nodes[node]["mean_daily_revenue"]
            )
            for node in sorted(candidate_nodes)
        }
        top_five = np.argsort(difference)[-5:]
        keep = np.ones(len(difference), dtype=bool)
        keep[top_five] = False
        worst_count = max(1, int(np.ceil(0.10 * len(difference))))
        baseline_daily = np.asarray([
            baseline["macro_daily_net_revenue"][day] for day in dates
        ])
        baseline_worst = np.argsort(baseline_daily)[:worst_count]
        rows.append({
            "family": family,
            "candidate_macro_mean_daily_revenue": candidate[
                "macro_mean_daily_revenue"
            ],
            "pooled_lomo_xgboost_macro_mean_daily_revenue": baseline[
                "macro_mean_daily_revenue"
            ],
            "mean_daily_difference": float(difference.mean()),
            "paired_7day_block_bootstrap_95ci": _block_bootstrap_ci(difference),
            "improved_nodes": int(sum(value > 0.0 for value in node_differences.values())),
            "node_count": len(node_differences),
            "node_differences": node_differences,
            "difference_after_removing_5_best_delta_days": float(
                difference[keep].mean()
            ),
            "difference_on_baseline_worst_10pct_days": float(
                difference[baseline_worst].mean()
            ),
            "passes_point_estimate": bool(difference.mean() > 0.0),
            "passes_paired_interval": bool(
                _block_bootstrap_ci(difference)[0] > 0.0
            ),
            "passes_remove_top5": bool(difference[keep].mean() > 0.0),
            "source": path.relative_to(ROOT).as_posix(),
        })
    rows.sort(key=lambda item: item["mean_daily_difference"], reverse=True)
    return {
        "experiment": "global_implicit_qp_candidate_summary_v1",
        "baseline": "pooled_lomo_xgboost",
        "baseline_registry_stage": "development_2023",
        "selection_year": 2023,
        "2024_used": False,
        "2025_used": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="data/results/global_implicit_qp_settlement_v1"
    )
    parser.add_argument(
        "--output", default="data/results/global_implicit_qp_summary_v1.json"
    )
    args = parser.parse_args()
    paths = sorted((ROOT / args.input_dir).glob("*_continuous_mpc.json"))
    if not paths:
        raise FileNotFoundError("no implicit-QP settlement reports found")
    report = summarize(paths)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "implicit_qp_summary_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "best": report["rows"][0],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

