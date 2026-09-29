from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.config import PilotConfig
from decision_aware.dataset_rt_da import DecisionAwareRTAtDADataset


def _frame() -> pd.DataFrame:
    index = pd.date_range("2019-12-20", "2020-01-04", freq="h", tz="UTC")
    local = index.tz_convert("America/Chicago")
    values = np.arange(len(index), dtype=np.float32)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()
    return pd.DataFrame({
        "price_da": values,
        "price_rt": values + 100,
        "load": values + 2,
        "wind": values + 3,
        "solar": values + 4,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "is_weekend": (dow >= 5).astype(np.float32),
        "is_holiday": np.zeros(len(index), dtype=np.float32),
    }, index=index)


def _cfg() -> PilotConfig:
    return PilotConfig(
        market="ERCOT", context_len=168, horizon_da=24,
        da_issue_hour_local=10, data_version="v3",
    )


def test_rt_at_da_uses_same_delivery_day_contract_and_explicit_target():
    dataset = DecisionAwareRTAtDADataset(_frame(), _cfg(), "train")
    idx = next(i for i, w in enumerate(dataset.windows)
               if str(w.delivery_date) == "2020-01-02")
    contract = dataset.sample_time_contract(idx)
    sample = dataset[idx]

    assert str(contract["issue_time"]) == "2020-01-01 10:00:00-06:00"
    assert str(contract["target_start"]) == "2020-01-02 00:00:00-06:00"
    assert str(contract["target_end"]) == "2020-01-02 23:00:00-06:00"
    assert sample["price_rt_at_da_tgt"].shape == (24,)
    np.testing.assert_array_equal(
        sample["price_rt_at_da_tgt"].numpy(),
        dataset.price_rt[dataset.windows[idx].target_positions],
    )
    assert "price_da_tgt" not in sample
    assert "price_rt_tgt" not in sample
    assert "price_da_tgt_known" not in sample
    assert "load_tgt" not in sample
    assert "system_tgt" not in sample


def test_future_actual_values_cannot_change_rt_at_da_context():
    frame = _frame()
    before = DecisionAwareRTAtDADataset(frame, _cfg(), "train")
    idx = next(i for i, w in enumerate(before.windows)
               if str(w.delivery_date) == "2020-01-02")
    issue = before.windows[idx].issue_time_utc
    changed_frame = frame.copy()
    future_columns = ["price_rt", "load", "wind", "solar"]
    changed_frame.loc[changed_frame.index >= issue, future_columns] += 1_000_000
    changed = DecisionAwareRTAtDADataset(
        changed_frame, _cfg(), "train", norm_stats=before.norm_stats
    )
    changed_idx = next(i for i, w in enumerate(changed.windows)
                       if str(w.delivery_date) == "2020-01-02")

    for key in ("price_rt_ctx", "load_ctx", "system_ctx"):
        np.testing.assert_array_equal(
            before[idx][key].numpy(), changed[changed_idx][key].numpy()
        )
