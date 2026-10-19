#!/usr/bin/env python
"""Summarize the post-2025 DA tail-weighting follow-up without model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.tail_diagnostics import (  # noqa: E402
    fit_tail_thresholds,
    price_slice_report,
    spread_slice_report,
)
from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    _target_history,
)
from scripts.decision_aware.summarize_universal_lomo_multiseed_profit import (  # noqa: E402
    circular_block_ci,
)


MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")
BASE_SYSTEM = "da_ensemble__rt_at_da_ensemble__rt_ensemble"
CANDIDATE_SYSTEM = BASE_SYSTEM


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _paired(candidate: dict, current: dict) -> dict:
    left = candidate["macro_daily_net_revenue"]
    right = current["macro_daily_net_revenue"]
    dates = sorted(right)
    if dates != sorted(left):
        raise ValueError("当前与候选系统的2025日期不一致")
    difference = np.asarray([left[day] - right[day] for day in dates])
    return {
        "mean_daily_difference": float(np.mean(difference)),
        "median_daily_difference": float(np.median(difference)),
        "positive_difference_day_rate": float(np.mean(difference > 0.0)),
        "seven_day_block_bootstrap_95ci": circular_block_ci(
            difference, block=7, draws=5000, seed=20261020
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-ensemble-dir", required=True)
    parser.add_argument("--candidate-ensemble-dir", required=True)
    parser.add_argument("--current-settlement", required=True)
    parser.add_argument("--candidate-settlement", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    current_dir = ROOT / args.current_ensemble_dir
    candidate_dir = ROOT / args.candidate_ensemble_dir
    forecast = {}
    for market in MARKETS:
        current_da = _load(current_dir / f"{market.lower()}_da_ensemble.npz")
        candidate_da = _load(candidate_dir / f"{market.lower()}_da_ensemble.npz")
        for key in ("nodes", "dates", "target"):
            if key in ("nodes", "dates"):
                equal = np.array_equal(current_da[key], candidate_da[key])
            else:
                equal = np.allclose(current_da[key], candidate_da[key], rtol=0.0, atol=1e-5)
            if not equal:
                raise ValueError(f"{market}当前与候选DA的{key}不一致")
        history_da, history_rt = _target_history(market, current_da["nodes"].astype(str), 2024)
        thresholds = fit_tail_thresholds(history_da, history_rt)
        market_report = {
            "thresholds_from_2020_2024": thresholds.to_dict(),
            "current_da": price_slice_report(
                current_da["prediction"], current_da["target"],
                lower_1pct=thresholds.da_lower_1pct,
                upper_1pct=thresholds.da_upper_1pct,
            ),
            "tail_price_da": price_slice_report(
                candidate_da["prediction"], candidate_da["target"],
                lower_1pct=thresholds.da_lower_1pct,
                upper_1pct=thresholds.da_upper_1pct,
            ),
        }
        if market == "ERCOT":
            rt_at_da = _load(current_dir / "ercot_rt_at_da_ensemble.npz")
            market_report["current_spread"] = spread_slice_report(
                current_da["prediction"], rt_at_da["prediction"],
                current_da["target"], rt_at_da["target"], thresholds,
            )
            market_report["tail_price_da_spread"] = spread_slice_report(
                candidate_da["prediction"], rt_at_da["prediction"],
                candidate_da["target"], rt_at_da["target"], thresholds,
            )
        forecast[market] = market_report

    current_settlement = json.loads((ROOT / args.current_settlement).read_text(encoding="utf-8"))
    candidate_settlement = json.loads((ROOT / args.candidate_settlement).read_text(encoding="utf-8"))
    current_profit = current_settlement["systems"][BASE_SYSTEM]
    candidate_profit = candidate_settlement["systems"][CANDIDATE_SYSTEM]
    paired = _paired(candidate_profit, current_profit)
    report = {
        "experiment": "allmarket_da_tailprice_posthoc_followup_2025_v2",
        "status": "post_original_2025_followup_no_model_selection_allowed",
        "selection_allowed": False,
        "original_2025_results_were_known_before_followup": True,
        "claim_boundary": (
            "The mechanism and weight were selected on 2023 and confirmed unchanged "
            "on 2024, but this is a post-original-test follow-up rather than a pristine "
            "first-look 2025 confirmation."
        ),
        "changed_component": "DA training loss weights only; RT-at-DA, rolling RT, LP/MPC and settlement unchanged",
        "forecast": forecast,
        "ercot_profit": {
            "current": current_profit,
            "tail_price_da": candidate_profit,
            "paired_vs_current": paired,
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "allmarket_tailprice_followup_summary_complete",
        "current_ercot_profit": current_profit["macro_mean_daily_revenue"],
        "candidate_ercot_profit": candidate_profit["macro_mean_daily_revenue"],
        "paired_difference": paired,
        "output": output.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
