from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.config import PilotConfig
from decision_aware.dataset_da import DecisionAwareDADataset


def _frame(start: str, end: str) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="h", tz="UTC")
    local = index.tz_convert("America/Chicago")
    values = np.arange(len(index), dtype=np.float32)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()
    return pd.DataFrame(
        {
            "price_da": values,
            "price_rt": values + 1,
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
        },
        index=index,
    )


def _cfg() -> PilotConfig:
    return PilotConfig(
        market="ERCOT",
        node="HB_HOUSTON",
        context_len=168,
        horizon_da=24,
        da_issue_hour_local=10,
        data_version="v3",
    )


def test_da_sample_uses_previous_day_10am_and_full_delivery_day() -> None:
    frame = _frame("2019-12-20", "2020-01-04")
    dataset = DecisionAwareDADataset(frame, _cfg(), "train")
    index = next(
        i
        for i, window in enumerate(dataset.windows)
        if str(window.delivery_date) == "2020-01-02"
    )

    contract = dataset.sample_time_contract(index)
    sample = dataset[index]

    assert str(contract["issue_time"]) == "2020-01-01 10:00:00-06:00"
    assert str(contract["context_end"]) == "2020-01-01 09:00:00-06:00"
    assert str(contract["target_start"]) == "2020-01-02 00:00:00-06:00"
    assert str(contract["target_end"]) == "2020-01-02 23:00:00-06:00"
    assert sample["price_da_ctx"].shape == (168,)
    assert sample["load_ctx"].shape == (168, 1)
    assert sample["cal_ctx"].shape == (168, 8)
    assert sample["cal_tgt"].shape == (24, 8)
    assert sample["price_da_tgt"].shape == (24,)
    assert sample["price_rt_tgt"].shape == (24,)


def test_data_after_issue_time_cannot_change_context() -> None:
    frame = _frame("2019-12-20", "2020-01-04")
    cfg = _cfg()
    before = DecisionAwareDADataset(frame, cfg, "train")
    index = next(
        i
        for i, window in enumerate(before.windows)
        if str(window.delivery_date) == "2020-01-02"
    )
    issue = before.windows[index].issue_time_utc

    changed_frame = frame.copy()
    changed_frame.loc[changed_frame.index >= issue, "price_da"] += 1_000_000
    changed = DecisionAwareDADataset(
        changed_frame, cfg, "train", norm_stats=before.norm_stats
    )
    changed_index = next(
        i
        for i, window in enumerate(changed.windows)
        if str(window.delivery_date) == "2020-01-02"
    )

    np.testing.assert_array_equal(
        before[index]["price_da_ctx"].numpy(),
        changed[changed_index]["price_da_ctx"].numpy(),
    )


def test_only_known_future_calendar_is_exposed() -> None:
    frame = _frame("2019-12-20", "2020-01-04")
    dataset = DecisionAwareDADataset(frame, _cfg(), "train")
    sample = dataset[0]

    assert "cal_tgt" in sample
    assert "load_tgt" not in sample
    assert "system_tgt" not in sample


def test_spring_dst_day_is_excluded_from_24_hour_contract() -> None:
    frame = _frame("2020-02-25", "2020-03-15")
    dataset = DecisionAwareDADataset(frame, _cfg(), "train")

    assert pd.Timestamp("2020-03-08").date() in dataset.excluded_non_24h_days
    assert all(str(window.delivery_date) != "2020-03-08" for window in dataset.windows)


def test_validation_reuses_training_statistics() -> None:
    frame = _frame("2019-12-20", "2021-01-10")
    cfg = _cfg()
    train = DecisionAwareDADataset(frame, cfg, "train")
    val = DecisionAwareDADataset(frame, cfg, "val", norm_stats=train.norm_stats)

    assert val.norm_stats is train.norm_stats


def test_experiment_can_freeze_explicit_time_split() -> None:
    cfg = PilotConfig(
        data_version="v3",
        split_train_start="2020-02-01 00:00",
        split_train_end="2023-12-31 23:00",
        split_val_start="2024-01-01 00:00",
        split_val_end="2024-06-30 23:00",
        split_test_start="2024-07-01 00:00",
        split_test_end="2024-12-31 23:00",
    )
    assert cfg.split_bounds("train") == (
        "2020-02-01 00:00", "2023-12-31 23:00"
    )
    assert cfg.split_bounds("val") == (
        "2024-01-01 00:00", "2024-06-30 23:00"
    )
    assert cfg.split_bounds("test") == (
        "2024-07-01 00:00", "2024-12-31 23:00"
    )
