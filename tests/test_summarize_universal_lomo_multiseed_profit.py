import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.summarize_universal_lomo_multiseed_profit import (
    BASELINE,
    ENSEMBLE,
    circular_block_ci,
    summarize,
)


def _system(value):
    daily = {f"2023-01-{day:02d}": value + day for day in range(1, 15)}
    return {
        "macro_mean_daily_revenue": float(np.mean(list(daily.values()))),
        "macro_median_daily_revenue": float(np.median(list(daily.values()))),
        "macro_positive_day_rate": 1.0,
        "macro_mean_daily_revenue_without_top_5_days": value + 5.0,
        "top_5_days_total_revenue_share": 0.5,
        "top_10_days_total_revenue_share": 0.8,
        "worst_node_mean_daily_revenue": value - 1.0,
        "best_node_mean_daily_revenue": value + 1.0,
        "macro_mean_daily_revenue_7day_block_bootstrap_95ci": [value, value + 15.0],
        "macro_daily_net_revenue": daily,
    }


def test_circular_block_ci_is_deterministic():
    values = np.arange(30.0)
    assert circular_block_ci(values, draws=100, seed=7) == circular_block_ci(
        values, draws=100, seed=7
    )


def test_summarize_uses_all_three_seeds_without_selecting_best():
    systems = {
        f"da_seed{seed}__rt_at_da_seed{seed}__rt_seed{seed}": _system(10.0 + seed)
        for seed in range(3)
    }
    systems[ENSEMBLE] = _system(12.0)
    systems[BASELINE] = _system(8.0)
    result = summarize({"split": "2023 inner validation only", "systems": systems})
    assert result["seed_selection_by_profit"] is False
    expected = np.mean([
        systems[f"da_seed{seed}__rt_at_da_seed{seed}__rt_seed{seed}"][
            "macro_mean_daily_revenue"
        ]
        for seed in range(3)
    ])
    assert result["across_seed_mean_daily_revenue"] == pytest.approx(expected)
    assert result["paired_vs_xgboost"]["fixed_equal_weight_ensemble"][
        "mean_daily_difference"
    ] == pytest.approx(4.0)


def test_summarize_rejects_missing_seed():
    systems = {
        "da_seed0__rt_at_da_seed0__rt_seed0": _system(10.0),
        "da_seed2__rt_at_da_seed2__rt_seed2": _system(12.0),
        ENSEMBLE: _system(11.0),
        BASELINE: _system(8.0),
    }
    with pytest.raises(ValueError, match="seed 0、1、2"):
        summarize({"split": "2023 inner validation only", "systems": systems})
