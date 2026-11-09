from __future__ import annotations

import numpy as np

from scripts.decision_aware.summarize_austria_controlled_da_robustness import (
    circular_block_bootstrap,
    paired_bootstrap,
)


def test_paired_bootstrap_preserves_constant_advantage() -> None:
    result = paired_bootstrap(
        np.full(20, 3.5),
        samples=1_000,
        rng=np.random.default_rng(0),
    )
    assert result["mean_v4_advantage_eur_per_day"] == 3.5
    assert result["ci95_low"] == 3.5
    assert result["ci95_high"] == 3.5
    assert result["bootstrap_probability_v4_advantage_positive"] == 1.0
    assert result["v4_daily_win_rate"] == 1.0


def test_paired_bootstrap_reports_mixed_daily_wins() -> None:
    result = paired_bootstrap(
        np.asarray([-2.0, 1.0, 3.0, 4.0]),
        samples=2_000,
        rng=np.random.default_rng(7),
    )
    assert result["mean_v4_advantage_eur_per_day"] == 1.5
    assert result["v4_daily_win_rate"] == 0.75
    assert result["ci95_low"] < result["ci95_high"]


def test_circular_block_bootstrap_preserves_constant_advantage() -> None:
    result = circular_block_bootstrap(
        np.full(30, 2.0),
        samples=1_000,
        block_length=7,
        rng=np.random.default_rng(3),
    )
    assert result["block_length_days"] == 7
    assert result["ci95_low"] == 2.0
    assert result["ci95_high"] == 2.0
    assert result["bootstrap_probability_v4_advantage_positive"] == 1.0
