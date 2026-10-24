#!/usr/bin/env python
"""Summarize the frozen economic spread coordinator across 2023 and 2024."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _block_bootstrap_ci,
)


BASELINE = "da_current__rt_at_da_current__rt_current"
CANDIDATE = "da_econcoord__rt_at_da_econcoord__rt_econcoord"
COMPONENTS = ("da_leg", "rt_deviation_leg", "degradation_cost", "deviation_penalty")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _system_summary(system: dict) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    nodes = sorted(system["nodes"])
    dates = sorted(system["nodes"][nodes[0]]["daily_net_revenue"])
    matrix = np.asarray(
        [
            [system["nodes"][node]["daily_net_revenue"][date] for date in dates]
            for node in nodes
        ],
        dtype=np.float64,
    )
    macro_daily = matrix.mean(axis=0)
    components = {
        component: float(
            np.mean(
                [
                    system["nodes"][node]["revenue_components"][component]
                    / system["nodes"][node]["days"]
                    for node in nodes
                ]
            )
        )
        for component in COMPONENTS
    }
    throughput = float(
        np.mean(
            [
                system["nodes"][node]["actual_throughput_mwh"]
                / system["nodes"][node]["days"]
                for node in nodes
            ]
        )
    )
    cumulative = np.cumsum(macro_daily)
    running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    return (
        {
            "mean_net_revenue": float(macro_daily.mean()),
            "components_per_node_day": components,
            "daily_throughput_mwh": throughput,
            "daily_equivalent_cycles": throughput / 8.0,
            "maximum_drawdown_usd": float(np.max(running_peak - cumulative)),
            "positive_day_rate": float(np.mean(macro_daily > 0.0)),
            "worst_node_mean_revenue": float(matrix.mean(axis=1).min()),
            "best_node_mean_revenue": float(matrix.mean(axis=1).max()),
        },
        dates,
        matrix,
        macro_daily,
    )


def _comparison(baseline: dict, candidate: dict) -> dict:
    b_summary, b_dates, b_matrix, b_daily = _system_summary(baseline)
    c_summary, c_dates, c_matrix, c_daily = _system_summary(candidate)
    if b_dates != c_dates or b_matrix.shape != c_matrix.shape:
        raise ValueError("baseline and candidate settlement grids differ")
    difference = c_daily - b_daily
    ordered_difference = np.sort(difference)
    node_difference = (c_matrix - b_matrix).mean(axis=1)
    # A second trimming view removes the candidate's highest-revenue dates,
    # rather than the dates with the largest candidate-baseline advantage.
    candidate_order = np.argsort(c_daily)

    def without_top_difference(count: int) -> float:
        return float(ordered_difference[:-count].mean())

    def without_top_candidate_revenue(count: int) -> float:
        keep = candidate_order[:-count]
        return float(difference[keep].mean())

    return {
        "baseline": b_summary,
        "candidate": c_summary,
        "mean_daily_difference": float(difference.mean()),
        "7day_block_bootstrap_95ci": _block_bootstrap_ci(
            difference, seed=20261020, draws=5000
        ),
        "positive_difference_day_rate": float(np.mean(difference > 0.0)),
        "positive_difference_node_rate": float(np.mean(node_difference > 0.0)),
        "node_difference_min": float(node_difference.min()),
        "node_difference_max": float(node_difference.max()),
        "node_differences": dict(zip(sorted(baseline["nodes"]), node_difference.tolist())),
        "without_top_difference_days": {
            str(count): without_top_difference(count) for count in (1, 5, 10)
        },
        "without_top_candidate_revenue_days": {
            str(count): without_top_candidate_revenue(count) for count in (1, 5, 10)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inner",
        default="data/results/universal_lomo_ercot_economic_coordination_inner_mpc_v1.json",
    )
    parser.add_argument(
        "--outer",
        default="data/results/universal_lomo_ercot_economic_coordination_outer_mpc_v1.json",
    )
    parser.add_argument(
        "--diagnostics",
        default=(
            "data/results/universal_lomo_ercot_economic_coordination_v1/"
            "economic_spread_coordination_model_and_diagnostics.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/results/universal_lomo_ercot_economic_coordination_summary_v1.json",
    )
    args = parser.parse_args()

    diagnostics = _load(ROOT / args.diagnostics)
    result = {
        "experiment": "frozen_economic_spread_coordination_v1",
        "status": "complete_closed_after_frozen_2024_confirmation",
        "selection_allowed": {"2023": True, "2024": False, "2025": False},
        "years": {},
        "forecast_diagnostics": diagnostics["diagnostics"],
        "model": diagnostics["model"],
    }
    for year, path in (("2023", args.inner), ("2024", args.outer)):
        report = _load(ROOT / path)
        result["years"][year] = _comparison(
            report["systems"][BASELINE], report["systems"][CANDIDATE]
        )

    outer = result["years"]["2024"]
    ci = outer["7day_block_bootstrap_95ci"]
    robust = (
        outer["mean_daily_difference"] > 0.0
        and ci[0] > 0.0
        and outer["positive_difference_node_rate"] >= 0.8
        and outer["without_top_difference_days"]["10"] > 0.0
    )
    result["conclusion"] = {
        "2024_profit_confirmation_passed": bool(robust),
        "route_status": (
            "promising_frozen_post_training_mechanism"
            if robust
            else "closed_no_robust_profit_improvement"
        ),
        "important_limitation": (
            "The mechanism raises economic-threshold crossings by taking much larger "
            "spread positions and worsens spread MAE; it is a decision calibration layer, "
            "not a better probabilistic price forecaster."
        ),
        "no_parameter_search_after_2024": True,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "passed": robust}))


if __name__ == "__main__":
    main()
