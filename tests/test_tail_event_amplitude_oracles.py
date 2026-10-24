import numpy as np
import pytest

from scripts.decision_aware.audit_tail_event_amplitude_oracles import (
    _oracle_prediction,
)


def _curve():
    prediction = np.asarray([[[10.0, 20.0, 30.0, 40.0] + [0.0] * 20]])
    target = np.asarray([[[10.0, 500.0, -20.0, 40.0] + [0.0] * 20]])
    return prediction, target


def test_positive_event_oracle_only_crosses_boundary_on_true_event_hours():
    prediction, target = _curve()
    result, changed = _oracle_prediction(
        prediction, target, system="positive_event_exact", positive_threshold=432.48
    )
    assert result[0, 0, 1] == pytest.approx(433.48)
    assert changed.sum() == 1
    assert np.array_equal(result[~changed], prediction[~changed])


def test_positive_amplitude_oracle_reveals_truth_only_on_true_event_hours():
    prediction, target = _curve()
    result, changed = _oracle_prediction(
        prediction, target, system="positive_amplitude_exact", positive_threshold=432.48
    )
    assert result[0, 0, 1] == pytest.approx(500.0)
    assert changed.sum() == 1
    assert np.array_equal(result[~changed], prediction[~changed])


def test_negative_event_and_amplitude_oracles_are_distinct():
    prediction, target = _curve()
    event, event_changed = _oracle_prediction(
        prediction, target, system="negative_event_exact", positive_threshold=432.48
    )
    amplitude, amplitude_changed = _oracle_prediction(
        prediction, target, system="negative_amplitude_exact", positive_threshold=432.48
    )
    assert event[0, 0, 2] == pytest.approx(-1.0)
    assert amplitude[0, 0, 2] == pytest.approx(-20.0)
    assert event_changed.sum() == amplitude_changed.sum() == 1


def test_ranked_positive_oracle_uses_model_peak_hour_not_true_peak_hour():
    prediction, target = _curve()
    result, changed = _oracle_prediction(
        prediction, target, system="positive_event_ranked", positive_threshold=432.48
    )
    # 模型日峰值在hour=3，真尖峰在hour=1。
    assert result[0, 0, 3] == pytest.approx(433.48)
    assert result[0, 0, 1] == pytest.approx(prediction[0, 0, 1])
    assert changed.sum() == 1


def test_original_oracle_is_identity():
    prediction, target = _curve()
    result, changed = _oracle_prediction(
        prediction, target, system="original", positive_threshold=432.48
    )
    assert np.array_equal(result, prediction)
    assert not changed.any()
