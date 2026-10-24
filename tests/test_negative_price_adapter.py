from types import SimpleNamespace

import numpy as np
import pandas as pd

from decision_aware.negative_price_adapter import (
    FEATURE_NAMES,
    SEASONAL_EVENT_FEATURE_INDICES,
    apply_negative_price_adapter,
    balanced_event_weights,
    event_metrics,
    extract_da_adapter_features,
    select_market_macro_threshold,
    select_source_fpr_threshold,
)


def _dataset():
    index = pd.date_range("2023-01-01", periods=240, freq="h", tz="UTC")
    positions = np.arange(192, 216)
    window = SimpleNamespace(
        context_start=24,
        context_end=192,
        target_positions=positions,
        delivery_date=pd.Timestamp("2023-01-09").date(),
    )
    values = np.arange(240, dtype=np.float32)
    return SimpleNamespace(
        windows=[window],
        index=index,
        price_da=values - 100,
        price_rt=values - 90,
        streams={
            "load": (1000 + values)[:, None],
            "system": np.column_stack((100 + values, 50 + values)),
            "cal": np.zeros((240, 8), dtype=np.float32),
        },
    )


def test_feature_extraction_obeys_shape_and_lag_contract():
    bundle = extract_da_adapter_features(_dataset())
    assert bundle.x.shape == (24, len(FEATURE_NAMES))
    assert np.array_equal(bundle.x[:, 0], np.arange(24, 48) - 100)
    assert np.array_equal(bundle.target, np.arange(192, 216) - 100)
    assert set(bundle.delivery_dates) == {"2023-01-09"}


def test_event_weights_equalize_market_and_class_mass():
    labels = np.array([0, 0, 1, 0, 1, 1, 1], dtype=bool)
    markets = np.array(["A", "A", "A", "B", "B", "B", "B"])
    weights = balanced_event_weights(labels, markets)
    for market in ("A", "B"):
        mask = markets == market
        assert np.isclose(weights[mask & labels].sum(), weights[mask & ~labels].sum())
    assert np.isclose(weights[markets == "A"].sum(), weights[markets == "B"].sum())


def test_threshold_selection_uses_market_macro_f1():
    probability = np.array([0.9, 0.4, 0.8, 0.6])
    target = np.array([-1.0, 2.0, -2.0, 3.0])
    markets = np.array(["A", "A", "B", "B"])
    threshold, rows = select_market_macro_threshold(
        probability, target, markets, candidates=(0.5, 0.7)
    )
    assert threshold == 0.7
    assert len(rows) == 2


def test_adapter_only_changes_triggered_hours():
    base = np.array([5.0, -3.0, 7.0])
    probability = np.array([0.9, 0.8, 0.1])
    amplitude = np.array([-20.0, -2.0, -100.0])
    event, full, mask = apply_negative_price_adapter(
        base, probability, amplitude, threshold=0.5
    )
    assert np.array_equal(mask, [True, True, False])
    assert np.array_equal(event, [-1.0, -3.0, 7.0])
    assert np.array_equal(full, [-20.0, -2.0, 7.0])
    metrics = event_metrics(probability, np.array([-1.0, 1.0, 1.0]), 0.5)
    assert metrics["tp"] == 1 and metrics["fp"] == 1


def test_source_fpr_threshold_uses_most_conservative_market():
    probability = np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.9])
    target = np.ones(6)
    markets = np.array(["A", "A", "A", "B", "B", "B"])
    threshold, by_market = select_source_fpr_threshold(
        probability, target, markets, maximum_false_positive_rate=0.34
    )
    assert threshold == by_market["B"]
    assert threshold > 0.9
    assert len(SEASONAL_EVENT_FEATURE_INDICES) == 12

