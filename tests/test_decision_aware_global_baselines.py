import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.baselines_global import (
    CutoffSafeLEAR24,
    InvariantMedianScaler,
    cutoff_safe_lear_arrays,
    prediction_metrics,
    rolling_cutoff_safe_lear_predictions,
    portable_xgboost_target,
    restore_portable_xgboost_prediction,
    xgboost_feature,
)


class _DailyDataset:
    def __init__(self):
        self.price_da = np.arange(300, dtype=np.float32)
        self.price_rt = self.price_da + 1000
        self.windows = [
            SimpleNamespace(
                context_start=0,
                context_end=200,
                delivery_date=date(2023, 1, 2),
            )
        ]

    def __getitem__(self, index):
        return {
            "price_da_tgt": torch.arange(24, dtype=torch.float32),
            "price_rt_at_da_tgt": torch.arange(24, dtype=torch.float32) + 10,
        }


def test_invariant_scaler_round_trip_and_constant_column():
    values = np.column_stack([np.arange(20), np.ones(20)])
    scaler = InvariantMedianScaler()
    transformed = scaler.fit_transform(values)
    restored = scaler.inverse_transform(transformed)
    assert np.all(np.isfinite(transformed))
    assert np.allclose(restored, values)


def test_cutoff_safe_lear_features_never_cross_issue_position():
    dataset = _DailyDataset()
    features, target = cutoff_safe_lear_arrays(dataset, "da")
    assert features.shape == (1, 103)
    assert target.shape == (1, 24)
    # 四块分别终止于issue、issue-24、issue-48、issue-144；最大引用199。
    assert features[0, :24].tolist() == pytest.approx(list(range(176, 200)))
    assert features[0, 72:96].tolist() == pytest.approx(list(range(32, 56)))
    assert features[0, -7:].sum() == 1.0
    assert features[0, -7] == 1.0  # 2023-01-02是周一


def test_cutoff_safe_lear_rejects_insufficient_samples():
    model = CutoffSafeLEAR24()
    with pytest.raises(ValueError, match="样本数"):
        model.fit(np.zeros((20, 103)), np.zeros((20, 24)))


def test_xgboost_feature_uses_only_declared_visible_fields():
    sample = {
        "price_da_ctx": torch.arange(30.0),
        "price_rt_ctx": torch.arange(30.0),
        "load_ctx": torch.arange(30.0).unsqueeze(-1),
        "system_ctx": torch.zeros(30, 2),
        "cal_ctx": torch.zeros(30, 8),
        "price_da_tgt_known": torch.ones(4),
        "cal_tgt": torch.zeros(4, 8),
        # 故意加入巨大目标值；函数不允许读取它。
        "price_rt_tgt": torch.full((4,), 999999.0),
    }
    feature = xgboost_feature(sample, "rt", history_steps=24)
    assert np.max(feature) < 999999.0
    assert feature.shape == (24 * 13 + 4 + 4 * 8,)


@pytest.mark.parametrize(
    ("task", "target_key", "prefix"),
    [
        ("da", "price_da_tgt", "price_da"),
        ("rt_at_da", "price_rt_at_da_tgt", "price_rt"),
        ("rt", "price_rt_tgt", "price_rt"),
    ],
)
def test_portable_xgboost_target_round_trip_ignores_identity(
    task, target_key, prefix
):
    sample = {
        target_key: torch.tensor([8.0, 12.0]),
        f"{prefix}_mean": torch.tensor(10.0),
        f"{prefix}_std": torch.tensor(2.0),
        # 这两个字段可以供数据分组，但绝不能改变模型目标或特征。
        "node_id": torch.tensor(999999),
        "market_id": torch.tensor(999999),
    }
    normalized, center, scale = portable_xgboost_target(sample, task)
    assert normalized.tolist() == pytest.approx([-1.0, 1.0])
    restored = restore_portable_xgboost_prediction(
        normalized[None, :], np.array([center]), np.array([scale])
    )
    assert restored[0].tolist() == pytest.approx([8.0, 12.0])


def test_restore_portable_xgboost_prediction_rejects_bad_scale():
    with pytest.raises(ValueError, match="尺度必须为正"):
        restore_portable_xgboost_prediction(
            np.zeros((1, 2)), np.zeros(1), np.zeros(1)
        )


def test_xgboost_fit_rejects_unknown_device_before_training():
    from decision_aware.baselines_global import fit_xgboost_multioutput

    with pytest.raises(ValueError, match="device"):
        fit_xgboost_multioutput(
            np.zeros((2, 3)), np.zeros((2, 1)), device="magic"
        )


def test_prediction_metrics_shape_guard_and_values():
    metrics = prediction_metrics(np.array([[1.0, 3.0]]), np.array([[2.0, 1.0]]))
    assert metrics.mae == pytest.approx(1.5)
    assert metrics.rmse == pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError, match="形状"):
        prediction_metrics(np.zeros((2, 2)), np.zeros((2, 3)))


def test_rolling_lear_rejects_invalid_refit_interval_before_fitting():
    with pytest.raises(ValueError, match="refit_every_days"):
        rolling_cutoff_safe_lear_predictions(
            _DailyDataset(), _DailyDataset(), "da", refit_every_days=0
        )
