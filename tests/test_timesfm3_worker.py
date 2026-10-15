from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "models" / "workers"))

from foundation import FOUNDATION_REGISTRY, TimesFMForecaster
from worker_timesfm import MODEL_ID, _as_worker_arrays, _past_future_covariates


def test_active_timesfm_adapter_is_v3_and_declares_native_capabilities():
    model = TimesFMForecaster()
    assert model.name == "TimesFM3"
    assert model.supports_covariates is True
    assert model.supports_multivariate is True
    assert FOUNDATION_REGISTRY["TimesFM3"]().name == "TimesFM3"
    assert FOUNDATION_REGISTRY["TimesFM"]().name == "TimesFM3"
    assert MODEL_ID == "google/timesfm-3.0-pytorch"


def test_timesfm3_covariates_join_history_and_known_future(tmp_path):
    request = tmp_path / "request.npz"
    hist = np.arange(6, dtype=np.float32).reshape(3, 2)
    future = np.arange(4, dtype=np.float32).reshape(2, 2) + 10
    np.savez(request, hist_cov__0=hist, future_cov__0=future)

    with np.load(request) as req:
        actual = _past_future_covariates(req, 0)

    assert actual.shape == (2, 5)
    np.testing.assert_array_equal(actual, np.concatenate([hist, future], axis=0).T)


def test_timesfm3_multivariate_output_is_transposed_to_worker_protocol():
    horizon, n_series = 4, 2
    forecast = np.arange(n_series * horizon, dtype=np.float32).reshape(n_series, horizon)
    quantiles = np.zeros((n_series, horizon, 9), dtype=np.float32)
    quantiles[..., 0] = -1
    quantiles[..., 8] = 9
    output = SimpleNamespace(forecast=forecast, quantiles=quantiles)

    mean, q10, q90 = _as_worker_arrays(output, n_series, horizon)

    assert mean.shape == (horizon, n_series)
    np.testing.assert_array_equal(mean, forecast.T)
    np.testing.assert_array_equal(q10, np.full((horizon, n_series), -1, dtype=np.float32))
    np.testing.assert_array_equal(q90, np.full((horizon, n_series), 9, dtype=np.float32))


def test_timesfm3_univariate_output_keeps_horizon_first():
    horizon = 3
    output = SimpleNamespace(
        forecast=np.array([1, 2, 3], dtype=np.float32),
        quantiles=np.tile(np.arange(9, dtype=np.float32), (horizon, 1)),
    )

    mean, q10, q90 = _as_worker_arrays(output, 1, horizon)

    assert mean.shape == q10.shape == q90.shape == (horizon, 1)
    np.testing.assert_array_equal(q10[:, 0], np.zeros(horizon, dtype=np.float32))
    np.testing.assert_array_equal(q90[:, 0], np.full(horizon, 8, dtype=np.float32))
