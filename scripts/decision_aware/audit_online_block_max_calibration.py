#!/usr/bin/env python
"""Prequential 2023 audit of rolling conformal block-max calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.block_max_head import (  # noqa: E402
    event_probability_from_scalar_quantiles,
    online_conformalize_block_quantiles,
)
from scripts.decision_aware.smoke_chronos2_block_max_head import (  # noqa: E402
    _adjust_hourly_quantiles,
    _block_metrics,
)
from scripts.decision_aware.smoke_chronos2_tail_posttraining import evaluate  # noqa: E402


SOURCES = {
    "CAISO": "data/results/chronos2_blockmax_expected_oof_caiso3_seed0_v1",
    "ERCOT": "data/results/chronos2_blockmax_expected_oof_ercot3_seed0_v1",
    "NYISO": "data/results/chronos2_blockmax_expected_oof_nyiso3_seed0_v1",
    "PJM": "data/results/chronos2_blockmax_expected_oof_pjm3_seed0_v1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--minimum-history-days", type=int, default=30)
    parser.add_argument("--event-temperature", type=float, default=5.0)
    parser.add_argument(
        "--output-dir", default="data/results/chronos2_online_blockmax_calibration_v1"
    )
    args = parser.parse_args()
    markets: dict[str, object] = {}
    passed = 0
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for market, relative in SOURCES.items():
        source = ROOT / relative
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        archive = np.load(source / "predictions.npz")
        target = archive["target"]
        quantiles = archive["quantiles"]
        frozen = archive["frozen"]
        block = archive["block_max"]
        threshold = float(summary["positive_threshold"])
        target_max = np.max(target, axis=1)
        calibrated_block = online_conformalize_block_quantiles(
            block,
            target_max,
            quantiles,
            num_entities=3,
            window_days=args.window_days,
            minimum_history_days=args.minimum_history_days,
        )
        probabilities = event_probability_from_scalar_quantiles(
            torch.from_numpy(calibrated_block),
            torch.from_numpy(quantiles),
            threshold,
            temperature=args.event_temperature,
        ).numpy()
        baseline_block = np.max(frozen, axis=2)
        baseline_block_metrics = _block_metrics(
            baseline_block,
            target,
            quantiles,
            threshold,
            args.event_temperature,
        )
        calibrated_block_metrics = _block_metrics(
            calibrated_block,
            target,
            quantiles,
            threshold,
            args.event_temperature,
            event_probabilities=probabilities,
        )
        adjusted = _adjust_hourly_quantiles(
            frozen,
            calibrated_block,
            probabilities,
            quantiles,
            threshold,
            args.event_temperature,
        )
        frozen_metrics = evaluate(
            frozen,
            target,
            quantiles,
            positive_threshold=threshold,
            event_temperature=args.event_temperature,
        )
        adjusted_metrics = evaluate(
            adjusted,
            target,
            quantiles,
            positive_threshold=threshold,
            event_temperature=args.event_temperature,
        )
        baseline_block_metrics.pop("probabilities")
        calibrated_block_metrics.pop("probabilities")
        brier_improvement = (
            baseline_block_metrics["event_brier"]
            - calibrated_block_metrics["event_brier"]
        ) / baseline_block_metrics["event_brier"]
        block_conditional_improvement = (
            baseline_block_metrics["event_conditional_mae"]
            - calibrated_block_metrics["event_conditional_mae"]
        ) / baseline_block_metrics["event_conditional_mae"]
        hourly_conditional_improvement = (
            frozen_metrics["positive"]["conditional_mae"]
            - adjusted_metrics["positive"]["conditional_mae"]
        ) / frozen_metrics["positive"]["conditional_mae"]
        gate = {
            "block_event_brier_relative_improvement": brier_improvement,
            "block_event_brier_improves_1pct": brier_improvement >= 0.01,
            "block_event_conditional_mae_relative_improvement": block_conditional_improvement,
            "block_event_conditional_mae_improves_0_5pct": block_conditional_improvement >= 0.005,
            "hourly_event_conditional_mae_relative_improvement": hourly_conditional_improvement,
            "hourly_event_conditional_mae_improves_0_5pct": hourly_conditional_improvement >= 0.005,
            "ordinary_mae_within_1pct": adjusted_metrics["ordinary_mae"]
            <= frozen_metrics["ordinary_mae"] * 1.01,
            "ramp_mae_within_2pct": adjusted_metrics["ramp_mae"]
            <= frozen_metrics["ramp_mae"] * 1.02,
        }
        gate["passed"] = bool(all(value for key, value in gate.items() if key.endswith(("1pct", "0_5pct", "2pct"))))
        passed += int(gate["passed"])
        markets[market] = {
            "source": relative,
            "threshold": threshold,
            "screen_days": int(len(target)),
            "event_days": int(np.sum(target_max > threshold)),
            "frozen_hourly": frozen_metrics,
            "adjusted_hourly": adjusted_metrics,
            "baseline_block_max": baseline_block_metrics,
            "calibrated_block_max": calibrated_block_metrics,
            "screen_gate": gate,
        }
        np.savez_compressed(
            output_dir / f"{market.lower()}_predictions.npz",
            target=target,
            quantiles=quantiles,
            frozen=frozen,
            calibrated_block_max=calibrated_block,
            event_probability=probabilities,
            adjusted=adjusted,
        )
    report = {
        "status": "passed_screen" if passed == len(SOURCES) else "failed_screen",
        "scope": "strict_prequential_2023_screen_not_paper_result",
        "method": "per_node_rolling_conformal_block_max_quantile_calibration",
        "future_targets_read_by_current_prediction": False,
        "window_days": args.window_days,
        "minimum_history_days": args.minimum_history_days,
        "markets_passed": passed,
        "markets_total": len(SOURCES),
        "markets": markets,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
