from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.decision_aware.summarize_austria_da_day_groups import (
    classify_days,
    forecast_metrics,
    group_counts,
)


def test_day_groups_are_mutually_exclusive_with_negative_priority():
    local = pd.DatetimeIndex(
        [
            "2026-01-01 00:00+01:00",
            "2026-01-01 00:15+01:00",
            "2026-01-02 00:00+01:00",
            "2026-01-02 00:15+01:00",
            "2026-01-03 00:00+01:00",
            "2026-01-03 00:15+01:00",
        ]
    )
    # Day 1 has both a negative and a spike and must remain in the negative group.
    realized = np.asarray([-1.0, 300.0, 10.0, 250.0, 20.0, 30.0])
    daily = classify_days(local, realized)
    assert daily["day_group"].tolist() == ["neg", "spike", "ordinary"]
    assert group_counts(daily) == {"neg": 1, "spike": 1, "ordinary": 1}


def test_forecast_metrics_report_point_and_daily_macro_mae():
    local = pd.DatetimeIndex(
        [
            "2026-01-01 00:00+01:00",
            "2026-01-01 00:15+01:00",
            "2026-01-02 00:00+01:00",
            "2026-01-02 00:15+01:00",
        ]
    )
    realized = np.asarray([-1.0, 1.0, 10.0, 220.0])
    predicted = np.asarray([0.0, 0.0, 20.0, 200.0])
    daily = classify_days(local, realized)
    metrics = forecast_metrics(local, realized, predicted, daily)
    assert metrics["all"]["days"] == 2
    assert metrics["all"]["quarters"] == 4
    assert metrics["all"]["point_weighted_mae_eur_per_mwh"] == 8.0
    assert metrics["neg"]["point_weighted_mae_eur_per_mwh"] == 1.0
    assert metrics["spike"]["point_weighted_mae_eur_per_mwh"] == 15.0

