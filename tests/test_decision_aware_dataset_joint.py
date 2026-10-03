from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_da import DecisionAwareDADataset  # noqa: E402
from decision_aware.dataset_joint import JointDecisionAwareDataset  # noqa: E402
from decision_aware.dataset_rt import DecisionAwareRTDataset  # noqa: E402
from decision_aware.dataset_rt_da import DecisionAwareRTAtDADataset  # noqa: E402


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-12-15", "2025-01-15", freq="h", tz="UTC")
    local = index.tz_convert("America/Chicago")
    values = np.arange(len(index), dtype=np.float32)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()
    return pd.DataFrame({
        "price_da": 30.0 + values / 100.0,
        "price_rt": 35.0 + values / 90.0,
        "load": 1000.0 + values,
        "wind": 100.0 + values / 10.0,
        "solar": 50.0 + values / 20.0,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "is_weekend": (dow >= 5).astype(np.float32),
        "is_holiday": np.zeros(len(index), dtype=np.float32),
    }, index=index)


def test_joint_dataset_aligns_one_day_with_24_rolling_rt_samples():
    cfg = PilotConfig(
        context_len=24,
        horizon_da=24,
        horizon_rt=4,
        train_stride=1,
        split_train_start="2025-01-02 00:00",
        split_train_end="2025-01-10 23:00",
    )
    frame = _frame()
    da = DecisionAwareDADataset(frame, cfg, "train")
    rt_da = DecisionAwareRTAtDADataset(
        frame, cfg, "train", norm_stats=da.norm_stats
    )
    rt = DecisionAwareRTDataset(frame, cfg, "train", norm_stats=da.norm_stats)
    joint = JointDecisionAwareDataset(da, rt_da, rt)

    sample = joint[0]
    assert len(joint.windows[0].rt_indices) == 24
    assert sample["da"]["price_da_tgt"].shape == (24,)
    assert sample["rt_at_da"]["price_rt_at_da_tgt"].shape == (24,)
    assert sample["rt"]["price_rt_tgt"].shape == (24, 4)
    assert sample["starts_new_segment"].item() is True
    assert joint.local_timestamps()[:24] == sorted(joint.local_timestamps()[:24])


def test_joint_dataset_drops_split_tail_without_four_hour_rt_labels():
    cfg = PilotConfig(
        context_len=24,
        horizon_da=24,
        horizon_rt=4,
        train_stride=1,
        split_train_start="2025-01-02 00:00",
        split_train_end="2025-01-10 23:00",
    )
    frame = _frame()
    da = DecisionAwareDADataset(frame, cfg, "train")
    rt_da = DecisionAwareRTAtDADataset(
        frame, cfg, "train", norm_stats=da.norm_stats
    )
    rt = DecisionAwareRTDataset(frame, cfg, "train", norm_stats=da.norm_stats)
    joint = JointDecisionAwareDataset(da, rt_da, rt)

    assert pd.Timestamp("2025-01-10").date() in joint.excluded_missing_rt_dates
    assert all(
        window.delivery_date != pd.Timestamp("2025-01-10").date()
        for window in joint.windows
    )
