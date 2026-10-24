#!/usr/bin/env python
"""Combine the fixed 2025 coordination follow-up with frozen baseline results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.summarize_economic_spread_coordination import (  # noqa: E402
    _comparison,
)


BASELINE = "da_ensemble__rt_at_da_ensemble__rt_ensemble"
CANDIDATE = "da_econcoord__rt_at_da_econcoord__rt_econcoord"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--followup",
        default="data/results/allmarket_locked_2025_econcoord_followup_mpc_v1.json",
    )
    parser.add_argument(
        "--original",
        default="data/results/allmarket_locked_2025_all_continuous_mpc_v2.json",
    )
    parser.add_argument(
        "--tail",
        default="data/results/allmarket_locked_2025_econcoord_tail_profit_v1.json",
    )
    parser.add_argument(
        "--application",
        default="data/results/allmarket_locked_2025_econcoord_followup_v1/application_report.json",
    )
    parser.add_argument(
        "--output",
        default="data/results/allmarket_locked_2025_econcoord_followup_summary_v1.json",
    )
    args = parser.parse_args()

    followup = _load(ROOT / args.followup)
    original = _load(ROOT / args.original)
    tail = _load(ROOT / args.tail)
    application = _load(ROOT / args.application)
    comparison = _comparison(
        followup["systems"][BASELINE], followup["systems"][CANDIDATE]
    )
    frozen_baselines = {
        row["triplet"]: row["macro_mean_daily_revenue"]
        for row in original["ranking"]
    }
    selected = {
        "pooled_xgboost": frozen_baselines[
            "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost"
        ],
        "per_node_xgboost": frozen_baselines[
            "da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost"
        ],
        "dlinear": frozen_baselines[
            "da_dlinear__rt_at_da_dlinear__rt_dlinear"
        ],
        "timesfm25": frozen_baselines[
            "da_timesfm25__rt_at_da_timesfm25__rt_timesfm25"
        ],
        "timesfm3": frozen_baselines[
            "da_timesfm3__rt_at_da_timesfm3__rt_timesfm3"
        ],
    }
    candidate_profit = comparison["candidate"]["mean_net_revenue"]
    result = {
        "experiment": "post_original_2025_fixed_economic_coordination_followup_v1",
        "status": "complete_route_frozen",
        "selection_allowed": False,
        "comparison": comparison,
        "frozen_baseline_profit": selected,
        "candidate_minus_frozen_baseline": {
            name: candidate_profit - value for name, value in selected.items()
        },
        "tail_profit_attribution": tail,
        "application_compatibility": application,
        "conclusion": {
            "profit_improved_over_original_transformer": (
                comparison["mean_daily_difference"] > 0
                and comparison["7day_block_bootstrap_95ci"][0] > 0
            ),
            "beats_pooled_xgboost": candidate_profit > selected["pooled_xgboost"],
            "beats_dlinear": candidate_profit > selected["dlinear"],
            "tail_prediction_solved": False,
            "route_status": "retain_as_interpretable_profit_calibration_not_forecast_head",
            "further_2025_tuning_allowed": False,
        },
    }
    output = ROOT / args.output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
