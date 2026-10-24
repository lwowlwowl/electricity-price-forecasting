from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.decision_aware.smoke_chronos2_tail_posttraining import (
    _delivery_offset,
    _required_output_patches,
    _slice_delivery_predictions,
    screen_gate,
)


def test_delivery_offset_uses_timestamp_gap_from_forecast_origin() -> None:
    index = pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC")
    window = SimpleNamespace(context_end=10, target_positions=np.arange(24, 48))
    dataset = SimpleNamespace(index=index)
    assert _delivery_offset(window, dataset) == 14


def test_delivery_offset_rejects_missing_hour_alignment() -> None:
    index = pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC").delete(20)
    window = SimpleNamespace(context_end=10, target_positions=np.arange(24, 48))
    dataset = SimpleNamespace(index=index)
    with pytest.raises(ValueError, match="row offset"):
        _delivery_offset(window, dataset)


def test_delivery_slice_is_per_sample_and_requires_three_patches_for_14_hour_gap() -> None:
    offsets = np.asarray([14, 14], dtype=np.int64)
    assert _required_output_patches(offsets, 16) == 3
    predictions = torch.arange(2 * 1 * 48, dtype=torch.float32).reshape(2, 1, 48)
    selected = _slice_delivery_predictions(predictions, torch.from_numpy(offsets))
    assert selected.shape == (2, 1, 24)
    assert selected[0, 0, 0].item() == 14.0
    assert selected[0, 0, -1].item() == 37.0


def test_screen_gate_rejects_numerical_noise_as_tail_improvement() -> None:
    ordinary = {
        "ordinary_mae": 10.0,
        "ramp_mae": 4.0,
        "positive": {"conditional_mae": 500.0, "daily_peak_amplitude_ratio": 0.2},
        "negative": {"brier": 0.01},
    }
    proposed = {
        "ordinary_mae": 10.0,
        "ramp_mae": 4.0,
        "positive": {"conditional_mae": 499.99, "daily_peak_amplitude_ratio": 0.20001},
        "negative": {"brier": 0.0099999},
    }
    gate = screen_gate({}, ordinary, proposed)
    assert gate["positive_tail_improves"] is False
    assert gate["negative_brier_improves"] is False
    assert gate["passed"] is False
