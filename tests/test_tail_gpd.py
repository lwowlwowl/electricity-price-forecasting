import numpy as np

from decision_aware.tail_gpd import (
    conditional_tail_mean,
    gpd_negative_log_likelihood,
    gpd_shape,
)


def test_shape_constraint_keeps_finite_mean():
    for raw in (-100.0, 0.0, 100.0):
        assert -0.8 <= gpd_shape(raw) <= 0.8
        assert gpd_shape(raw) < 1.0


def test_exponential_limit_matches_known_nll():
    z = np.array([1.0, 2.0])
    features = np.empty((2, 0))
    parameters = np.array([np.log(2.0), 0.0])
    actual = gpd_negative_log_likelihood(parameters, z, features)
    expected = np.mean(np.log(2.0) + z / 2.0)
    assert np.isclose(actual, expected)


def test_tail_mean_uses_threshold_stability():
    prediction, valid = conditional_tail_mean(
        np.array([10.0]), np.array([20.0]), np.array([5.0]), 0.2
    )
    # sigma_20=5+0.2*(20-10)=7; mean=20+7/(1-0.2)=28.75
    assert valid.tolist() == [True]
    assert np.allclose(prediction, [28.75])
