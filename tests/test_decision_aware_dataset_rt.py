from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.config import PilotConfig
from decision_aware.dataset_rt import DecisionAwareRTDataset


def _frame():
    index = pd.date_range("2024-12-20", "2025-01-03", freq="1h", tz="UTC")
    n = len(index)
    phase = np.arange(n, dtype=np.float32)
    return pd.DataFrame({
        "price_da": 20.0 + phase / 100.0,
        "price_rt": 22.0 + phase / 90.0,
        "load": 1000.0 + phase,
        "wind": 100.0 + phase / 10.0,
        "solar": 50.0 + phase / 20.0,
        "hour_sin": np.sin(phase),
        "hour_cos": np.cos(phase),
        "dow_sin": np.sin(phase / 7),
        "dow_cos": np.cos(phase / 7),
        "month_sin": np.sin(phase / 30),
        "month_cos": np.cos(phase / 30),
        "is_weekend": np.zeros(n, dtype=np.float32),
        "is_holiday": np.zeros(n, dtype=np.float32),
    }, index=index)


def test_rt_window_ends_before_issue_and_predicts_four_hours():
    cfg = PilotConfig(context_len=24, horizon_rt=4, train_stride=1, val_stride=1)
    frame = _frame()
    train = DecisionAwareRTDataset(frame, cfg, "train")
    val = DecisionAwareRTDataset(frame, cfg, "val", train.norm_stats)
    sample = val[0]
    contract = val.sample_time_contract(0)
    assert contract["issue_time"].hour == 0
    assert contract["context_end"] + pd.Timedelta(hours=1) == contract["issue_time"]
    assert contract["target_start"] == contract["issue_time"]
    assert contract["executed_target"] == contract["target_start"]
    assert sample["price_rt_tgt"].shape == (4,)
    assert sample["price_da_tgt_known"].shape == (4,)
    assert sample["cal_tgt"].shape == (4, 8)


def test_future_actual_load_is_not_exposed():
    cfg = PilotConfig(context_len=24, horizon_rt=4, train_stride=1, val_stride=1)
    frame = _frame()
    train = DecisionAwareRTDataset(frame, cfg, "train")
    first = DecisionAwareRTDataset(frame, cfg, "val", train.norm_stats)
    changed = frame.copy()
    positions = first.windows[0].target_positions
    changed.iloc[positions, changed.columns.get_loc("load")] += 1_000_000.0
    second = DecisionAwareRTDataset(changed, cfg, "val", train.norm_stats)
    assert np.allclose(first[0]["load_ctx"].numpy(), second[0]["load_ctx"].numpy())
    assert "load_tgt" not in first[0]


def test_rt_stride_is_applied_per_split():
    cfg = PilotConfig(context_len=24, horizon_rt=4, train_stride=2, val_stride=3)
    frame = _frame()
    train = DecisionAwareRTDataset(frame, cfg, "train")
    val = DecisionAwareRTDataset(frame, cfg, "val", train.norm_stats)
    starts = [window.issue_position for window in val.windows]
    assert all((right - left) == 3 for left, right in zip(starts, starts[1:]))
