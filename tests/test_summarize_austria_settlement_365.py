import numpy as np

from scripts.decision_aware.summarize_austria_settlement_365 import (
    block_bootstrap_ci,
    component_means,
)


def test_block_bootstrap_ci_is_deterministic_and_contains_constant_mean() -> None:
    values = np.full(20, 3.5, dtype=np.float64)
    first = block_bootstrap_ci(values, draws=50, seed=7)
    second = block_bootstrap_ci(values, draws=50, seed=7)
    assert first == second == [3.5, 3.5]


def test_component_means_uses_all_dates() -> None:
    daily = {
        "2026-01-01": {
            "da_leg": 2.0,
            "rt_deviation_leg": -1.0,
            "degradation_cost": 0.5,
            "deviation_penalty": 0.25,
            "throughput_mwh": 1.0,
        },
        "2026-01-02": {
            "da_leg": 4.0,
            "rt_deviation_leg": 1.0,
            "degradation_cost": 1.5,
            "deviation_penalty": 0.75,
            "throughput_mwh": 3.0,
        },
    }
    result = component_means(daily, sorted(daily))
    assert result == {
        "da_leg": 3.0,
        "rt_deviation_leg": 0.0,
        "degradation_cost": 1.0,
        "deviation_penalty": 0.5,
        "throughput_mwh": 2.0,
    }
