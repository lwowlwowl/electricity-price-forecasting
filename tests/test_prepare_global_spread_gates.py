import numpy as np

from scripts.decision_aware.prepare_global_spread_gates import build_gate_predictions


def test_spread_gates_do_not_use_targets_and_apply_expected_rules():
    seasonal_da = np.array([[[10.0, 300.0, 10.0]]])
    seasonal_rt = np.array([[[0.0, 0.0, 20.0]]])
    stable_da = np.array([[[8.0, 20.0, 30.0]]])
    stable_rt = np.array([[[2.0, 10.0, 10.0]]])
    candidates = build_gate_predictions(
        seasonal_da, seasonal_rt, stable_da, stable_rt
    )

    blend_da, _ = candidates["gate_blend50"]
    assert np.allclose(blend_da, 0.5 * (seasonal_da + stable_da))
    agree_da, agree_rt = candidates["gate_agree"]
    assert np.array_equal(agree_da, np.array([[[10.0, 300.0, 30.0]]]))
    assert np.array_equal(agree_rt, np.array([[[0.0, 0.0, 10.0]]]))
    extreme_da, extreme_rt = candidates["gate_extreme50"]
    assert np.array_equal(extreme_da, np.array([[[10.0, 20.0, 10.0]]]))
    assert np.array_equal(extreme_rt, np.array([[[0.0, 10.0, 20.0]]]))
    clip_da, clip_rt = candidates["gate_clip50"]
    assert np.allclose(clip_da - clip_rt, np.array([[[10.0, 50.0, -10.0]]]))
