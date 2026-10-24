from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.negative_price_adapter import FEATURE_NAMES
from decision_aware.positive_spike_day_audit import (
    DAY_FEATURE_NAMES,
    binary_metrics,
    bootstrap_pr_auc_difference,
    make_daily_bundle,
    market_equal_weights,
    node_event_thresholds,
    quantile_calibration,
    select_macro_f1_threshold,
)


def _synthetic_hourly_bundle() -> dict:
    dates = np.repeat(np.asarray(["2020-01-01", "2020-01-02"]), 24)
    hours = np.tile(np.arange(24), 2)
    x = np.zeros((48, len(FEATURE_NAMES)), dtype=np.float64)
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    for name, value in (
        ("da_ctx_median", 20.0), ("da_ctx_mad", 5.0),
        ("da_ctx_max", 80.0), ("da_ctx_last", 25.0),
        ("rt_ctx_median", 18.0), ("rt_ctx_mad", 6.0),
        ("rt_ctx_max", 75.0), ("rt_ctx_last", 20.0),
        ("load_ctx_median", 100.0), ("load_ctx_mad", 10.0),
        ("net_load_ctx_median", 80.0), ("net_load_ctx_mad", 8.0),
        ("wind_ctx_median", 10.0), ("wind_ctx_mad", 2.0),
        ("solar_ctx_median", 5.0), ("solar_ctx_mad", 1.0),
        ("dow_sin", 0.2), ("dow_cos", 0.8),
        ("month_sin", 0.5), ("month_cos", 0.5),
    ):
        x[:, index[name]] = value
    x[:, index["da_lag168"]] = 20.0 + hours
    x[:, index["rt_lag168"]] = 18.0 + hours
    x[:, index["load_lag168"]] = 100.0 + hours
    x[:, index["net_load_lag168"]] = 80.0 + hours
    x[:, index["wind_lag168"]] = 10.0 + 0.1 * hours
    x[:, index["solar_lag168"]] = 5.0 + 0.1 * hours
    target = np.concatenate((np.linspace(10, 50, 24), np.linspace(10, 150, 24)))
    return {
        "x": x, "target": target, "dates": dates, "hours": hours,
        "node_id": np.zeros(48, dtype=np.int16), "nodes": np.asarray(["NODE"]),
    }


def test_daily_feature_contract_and_event_label() -> None:
    bundle = _synthetic_hourly_bundle()
    thresholds = np.asarray([100.0])
    daily = make_daily_bundle(bundle, thresholds, years=(2020,))
    assert daily.x.shape == (2, len(DAY_FEATURE_NAMES))
    assert daily.y.tolist() == [False, True]
    assert np.isfinite(daily.x).all()


def test_event_threshold_uses_only_requested_historical_years() -> None:
    bundle = _synthetic_hourly_bundle()
    threshold = node_event_thresholds(bundle, end_year=2020, quantile=0.99)
    assert threshold.shape == (1,)
    assert threshold[0] > 100.0


def test_market_weights_equalize_market_mass_without_class_balance() -> None:
    markets = np.asarray(["A", "A", "A", "B"])
    weights = market_equal_weights(markets)
    assert np.isclose(weights[markets == "A"].sum(), weights[markets == "B"].sum())
    assert np.isclose(weights.mean(), 1.0)


def test_threshold_tie_prefers_higher_threshold() -> None:
    probability = np.asarray([0.9, 0.8, 0.2, 0.1])
    truth = np.asarray([True, True, False, False])
    markets = np.asarray(["A", "A", "B", "B"])
    selected, _ = select_macro_f1_threshold(
        probability, truth, markets, candidates=(0.3, 0.5, 0.7)
    )
    assert selected == 0.7


def test_metrics_calibration_and_date_bootstrap() -> None:
    probability = np.asarray([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])
    truth = np.asarray([True, True, False, False, True, False])
    dates = np.asarray(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "2023-01-03"])
    metrics = binary_metrics(probability, truth, 0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["pr_auc"] > metrics["event_rate"]
    calibration = quantile_calibration(probability, truth, bins=3)
    assert sum(row["samples"] for row in calibration) == len(truth)
    bootstrap = bootstrap_pr_auc_difference(
        probability, truth, dates, repeats=50, seed=1
    )
    assert bootstrap["repeats"] == 50
    assert len(bootstrap["ci95"]) == 2
