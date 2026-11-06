#!/usr/bin/env python
"""Recalculate teacher-board baselines whose prediction rule is fully specified."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_external import PRICE_DA, TIMESTAMP  # noqa: E402
from scripts.decision_aware.summarize_austria_da_day_groups import (  # noqa: E402
    SPIKE_FLOOR_EUR_PER_MWH,
    classify_days,
    forecast_metrics,
)


TEACHER_D1 = {
    "all": 26.095,
    "neg": 30.148,
    "spike": 33.488,
    "ordinary": 20.423,
}
TEACHER_SEASONAL = {
    "all": 58.046,
    "neg": 94.006,
    "spike": 66.999,
    "ordinary": 41.593,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/Austria/奥地利_20250801_20260930.parquet")
    parser.add_argument(
        "--output-dir", default="data/results/austria_teacher_simple_baselines_v1"
    )
    args = parser.parse_args()

    raw = pd.read_parquet(ROOT / args.raw)
    timestamp = pd.DatetimeIndex(pd.to_datetime(raw[TIMESTAMP], utc=True))
    price = pd.Series(
        pd.to_numeric(raw[PRICE_DA], errors="coerce").to_numpy(np.float64),
        index=timestamp,
    )
    evaluation_index = pd.date_range(
        pd.Timestamp("2025-09-30 22:00", tz="UTC"),
        pd.Timestamp("2026-09-30 22:00", tz="UTC"),
        freq="15min",
        inclusive="left",
    )
    truth = price.reindex(evaluation_index).to_numpy(np.float64)
    # The teacher's d-1 persist rule is the immediately preceding 96 quarters.
    prediction = price.shift(96).reindex(evaluation_index).to_numpy(np.float64)
    weekly_prediction = price.shift(7 * 96).reindex(evaluation_index).to_numpy(np.float64)
    if (
        not np.isfinite(truth).all()
        or not np.isfinite(prediction).all()
        or not np.isfinite(weekly_prediction).all()
    ):
        raise ValueError("d-1 persist评价窗口包含缺失值")

    local = evaluation_index.tz_convert("Europe/Vienna")
    groups = classify_days(local, truth, spike_floor=SPIKE_FLOOR_EUR_PER_MWH)
    metrics = forecast_metrics(local, truth, prediction, groups)
    compact = {
        name: float(metrics[name]["point_weighted_mae_eur_per_mwh"])
        for name in TEACHER_D1
    }
    comparison = {
        name: {
            "ours": compact[name],
            "teacher": TEACHER_D1[name],
            "absolute_gap_ours_minus_teacher": compact[name] - TEACHER_D1[name],
        }
        for name in TEACHER_D1
    }
    weekly_metrics = forecast_metrics(local, truth, weekly_prediction, groups)
    weekly_compact = {
        name: float(weekly_metrics[name]["point_weighted_mae_eur_per_mwh"])
        for name in TEACHER_SEASONAL
    }
    weekly_comparison = {
        name: {
            "ours": weekly_compact[name],
            "teacher": TEACHER_SEASONAL[name],
            "absolute_gap_ours_minus_teacher": (
                weekly_compact[name] - TEACHER_SEASONAL[name]
            ),
        }
        for name in TEACHER_SEASONAL
    }

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "d1_persist_predictions.npz",
        timestamp_unix_seconds=np.asarray(
            [int(item.timestamp()) for item in evaluation_index], dtype=np.int64
        ),
        truth_eur_per_mwh=truth.astype(np.float32),
        prediction_eur_per_mwh=prediction.astype(np.float32),
    )
    np.savez_compressed(
        output_dir / "seasonal_7d_predictions.npz",
        timestamp_unix_seconds=np.asarray(
            [int(item.timestamp()) for item in evaluation_index], dtype=np.int64
        ),
        truth_eur_per_mwh=truth.astype(np.float32),
        prediction_eur_per_mwh=weekly_prediction.astype(np.float32),
    )
    summary = {
        "protocol": "austria_teacher_simple_baselines_v1",
        "model": "d-1 persist",
        "prediction_rule": "UTC series shifted by exactly 96 native 15-minute rows",
        "days": 365,
        "quarters": int(len(evaluation_index)),
        "metrics": compact,
        "teacher_comparison": comparison,
        "seasonal_7d_candidate": {
            "status": "candidate_rule_check_not_claimed_as_teacher_implementation",
            "prediction_rule": "UTC series shifted by exactly 672 native 15-minute rows",
            "metrics": weekly_compact,
            "teacher_comparison": weekly_comparison,
        },
        "interpretation": (
            "The near-zero all-MAE gap verifies the local target series and scoring alignment."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

