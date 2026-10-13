import numpy as np
import pytest

from scripts.decision_aware.summarize_global_taskloss_v1 import (
    _best_candidate_label,
    _maximum_drawdown,
    _parse_triplet,
    _reshape_daily_prices,
)


def test_parse_triplet_preserves_rt_at_da_name():
    label = "da_taskloss0__rt_at_da_taskloss1__rt_taskloss0"
    assert _parse_triplet(label) == {
        "da": "taskloss0", "rt_at_da": "taskloss1", "rt": "taskloss0"
    }


def test_best_candidate_label_skips_pure_baseline():
    ranking = [
        {
            "triplet": (
                "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
            )
        },
        {"triplet": "da_taskloss1__rt_at_da_taskloss0__rt_taskloss1"},
    ]
    assert _best_candidate_label(ranking, "taskloss") == ranking[1]["triplet"]


def test_maximum_drawdown_uses_peak_to_later_trough():
    daily = {"d1": 100.0, "d2": 200.0, "d3": -400.0, "d4": 50.0}
    assert _maximum_drawdown(daily) == 400.0


def test_hourly_aligned_truth_is_restored_to_day_by_24():
    restored = _reshape_daily_prices(np.arange(48), 2, "DA真值")
    assert restored.shape == (2, 24)
    assert restored[1, 0] == 24


def test_daily_price_reshape_rejects_inconsistent_day_count():
    with pytest.raises(ValueError, match="day_count\\*24"):
        _reshape_daily_prices(np.arange(47), 2, "RT真值")
