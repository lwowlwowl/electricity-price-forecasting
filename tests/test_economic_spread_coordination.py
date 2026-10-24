from __future__ import annotations

import numpy as np

from scripts.decision_aware.build_economic_spread_coordination import (
    apply_coordinator,
    economic_state,
    raw_features,
)


def test_economic_state_uses_fixed_six_dollar_boundary() -> None:
    values = np.asarray([-6.1, -6.0, 0.0, 6.0, 6.1])
    assert economic_state(values, 6.0).tolist() == [-1, 0, 0, 0, 1]


def test_raw_features_are_hour_aligned() -> None:
    da = np.arange(24, dtype=np.float64).reshape(1, 1, 24)
    rt = da - 2.0
    features = raw_features(da, rt)
    assert features.shape == (24, 6)
    assert np.allclose(features[:, 0], 2.0)
    assert np.allclose(features[:, 1], 2.0)


def test_apply_coordinator_keeps_da_implicit_and_enforces_states() -> None:
    da = np.full((1, 1, 24), 20.0)
    rt = np.full((1, 1, 24), 18.0)
    # Force the first eight hours to state -1, next eight neutral, last eight +1.
    coef = np.zeros((3, 6), dtype=np.float64)
    intercept = np.asarray([1.0, 0.0, -1.0])
    model = {
        "threshold": 6.0,
        "feature_mean": [0.0] * 6,
        "feature_std": [1.0] * 6,
        "classes": [-1, 0, 1],
        "classifier_coefficients": coef.tolist(),
        "classifier_intercepts": intercept.tolist(),
        "magnitude_scale": 1.0,
        "magnitude": {
            "-1": {"coefficients": [8.0, 0.0, 0.0]},
            "1": {"coefficients": [9.0, 0.0, 0.0]},
        },
    }
    calibrated, diagnostics = apply_coordinator(da, rt, model)
    # All hours choose class -1 from the fixed logits.
    assert np.all(diagnostics["predicted_state"] == -1)
    assert np.allclose(da - calibrated, -8.0)
    assert np.allclose(calibrated, 28.0)
