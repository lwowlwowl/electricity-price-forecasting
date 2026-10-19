#!/usr/bin/env python
"""Summarize the frozen 2024 confirmation of the DA tail-weighting pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.tail_diagnostics import price_slice_report  # noqa: E402
from scripts.decision_aware.summarize_universal_lomo_multiseed_profit import (  # noqa: E402
    circular_block_ci,
)


CURRENT = "da_universal__rt_at_da_universal__rt_universal"
CANDIDATE = "da_tailpriceda__rt_at_da_tailpriceda__rt_tailpriceda"


def _archive(path: Path, *, target_key: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "nodes": archive["nodes"].astype(str),
            "dates": archive["dates"].astype(str),
            "prediction": archive["prediction"].astype(np.float32),
            "target": archive[target_key].astype(np.float32),
        }


def _paired(candidate: dict, current: dict) -> dict:
    left = candidate["macro_daily_net_revenue"]
    right = current["macro_daily_net_revenue"]
    dates = sorted(right)
    if dates != sorted(left):
        raise ValueError("候选与当前系统的日期不一致")
    difference = np.asarray([left[day] - right[day] for day in dates])
    return {
        "mean_daily_difference": float(np.mean(difference)),
        "median_daily_difference": float(np.median(difference)),
        "positive_difference_day_rate": float(np.mean(difference > 0.0)),
        "seven_day_block_bootstrap_95ci": circular_block_ci(
            difference, block=7, draws=5000, seed=20261019
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-da", required=True)
    parser.add_argument("--candidate-da", required=True)
    parser.add_argument("--development-summary", required=True)
    parser.add_argument("--settlement", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    current = _archive(ROOT / args.current_da, target_key="target_da")
    candidate = _archive(ROOT / args.candidate_da, target_key="target")
    for key in ("nodes", "dates", "target"):
        if key in ("nodes", "dates"):
            equal = np.array_equal(current[key], candidate[key])
        else:
            equal = np.allclose(current[key], candidate[key], rtol=0.0, atol=1e-5)
        if not equal:
            raise ValueError(f"2024当前与候选DA的{key}不一致")

    development = json.loads((ROOT / args.development_summary).read_text(encoding="utf-8"))
    thresholds = development["thresholds"]
    current_forecast = price_slice_report(
        current["prediction"], current["target"],
        lower_1pct=float(thresholds["da_lower_1pct"]),
        upper_1pct=float(thresholds["da_upper_1pct"]),
    )
    candidate_forecast = price_slice_report(
        candidate["prediction"], candidate["target"],
        lower_1pct=float(thresholds["da_lower_1pct"]),
        upper_1pct=float(thresholds["da_upper_1pct"]),
    )

    settlement = json.loads((ROOT / args.settlement).read_text(encoding="utf-8"))
    if settlement.get("split") != "2024 outer validation only":
        raise ValueError("输入不是冻结2024 outer结算")
    if settlement.get("selection_allowed") is not False:
        raise ValueError("2024 outer结果不得用于调参")
    systems = settlement["systems"]
    current_profit = systems[CURRENT]
    candidate_profit = systems[CANDIDATE]
    paired = _paired(candidate_profit, current_profit)
    report = {
        "experiment": "universal_lomo_ercot_da_tailprice_outer_confirmation_v1",
        "status": "2024_outer_frozen_confirmation_not_for_selection",
        "selection_allowed": False,
        "test_2025_read": False,
        "candidate_frozen_from_2023": True,
        "single_changed_mechanism": development["single_changed_mechanism"],
        "thresholds_frozen_from_2020_2022": thresholds,
        "forecast": {"current": current_forecast, "tail_price_da": candidate_forecast},
        "profit": {
            "current": current_profit,
            "tail_price_da": candidate_profit,
            "paired_vs_current": paired,
        },
        "confirmation": {
            "mae_improves": candidate_forecast["all"]["mae"] < current_forecast["all"]["mae"],
            "profit_improves": (
                candidate_profit["macro_mean_daily_revenue"]
                > current_profit["macro_mean_daily_revenue"]
            ),
            "paired_interval_excludes_zero": paired["seven_day_block_bootstrap_95ci"][0] > 0.0,
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "tailprice_da_outer_confirmation_complete",
        "current_da_mae": current_forecast["all"]["mae"],
        "candidate_da_mae": candidate_forecast["all"]["mae"],
        "current_profit": current_profit["macro_mean_daily_revenue"],
        "candidate_profit": candidate_profit["macro_mean_daily_revenue"],
        "paired_difference": paired,
        "confirmation": report["confirmation"],
        "output": output.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
