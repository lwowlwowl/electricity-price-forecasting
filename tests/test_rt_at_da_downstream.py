import pytest

from scripts.decision_aware.evaluate_rt_at_da_downstream import (
    _average_rolling_rt_reports,
)


def _report(day_values, digest):
    daily = {
        date: {
            "da_leg": value + 1.0,
            "rt_deviation_leg": value + 2.0,
            "degradation_cost": 1.0,
            "deviation_penalty": 0.0,
            "net_revenue": value,
        }
        for date, value in day_values.items()
    }
    return {
        "daily": daily,
        "segments": 1,
        "mean_daily_revenue": sum(day_values.values()) / len(day_values),
        "plan_feasibility": {"reference_count": 9},
        "actual_state": {"action_digest": digest},
    }


def test_average_rolling_rt_reports_allows_different_rt_actions():
    result = _average_rolling_rt_reports([
        ("rt0", _report({"2025-01-01": 10.0, "2025-01-02": 20.0}, "a")),
        ("rt1", _report({"2025-01-01": 14.0, "2025-01-02": 18.0}, "b")),
    ])
    assert result["mean_daily_revenue"] == pytest.approx(15.5)
    assert result["daily"]["2025-01-01"]["net_revenue"] == pytest.approx(12.0)
    assert result["actual_state"]["per_rolling_rt_action_digest"] == {
        "rt0": "a", "rt1": "b"
    }


def test_average_rolling_rt_reports_rejects_date_mismatch():
    with pytest.raises(ValueError, match="日期不一致"):
        _average_rolling_rt_reports([
            ("rt0", _report({"2025-01-01": 10.0}, "a")),
            ("rt1", _report({"2025-01-02": 10.0}, "b")),
        ])
