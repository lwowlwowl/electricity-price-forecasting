from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_v3 import DecisionAwareDatasetV3  # noqa: E402


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    values = np.arange(n, dtype=np.float32)
    return pd.DataFrame(
        {
            "price_da": values,
            "price_rt": values + 1,
            "load": values + 100,
            "wind": values + 10,
            "solar": values + 20,
            "hour_sin": np.sin(2 * np.pi * index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * index.hour / 24),
            "dow_sin": np.sin(2 * np.pi * index.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * (index.month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (index.month - 1) / 12),
            "is_weekend": (index.dayofweek >= 5).astype(float),
            "is_holiday": np.zeros(n),
        },
        index=index,
    )


class DecisionAwareDatasetV3Tests(unittest.TestCase):
    def _cfg(self, dual: bool = False) -> PilotConfig:
        return PilotConfig(
            data_version="v3",
            context_len=4,
            horizon_da=2,
            use_dual_split=dual,
            train_stride=1,
            eval_stride=1,
            freq="1h",
        )

    def test_dual_split_doubles_target_length(self):
        index = pd.date_range("2020-01-01", periods=24, freq="1h", tz="UTC")
        ds = DecisionAwareDatasetV3(_frame(index), self._cfg(dual=True), "train")
        self.assertEqual(ds.horizon_tgt, 4)
        self.assertEqual(ds[0]["price_da_tgt"].shape, (4,))

    def test_windows_never_cross_missing_hour(self):
        full = pd.date_range("2020-01-01", periods=24, freq="1h", tz="UTC")
        index = full.delete(10)
        ds = DecisionAwareDatasetV3(_frame(index), self._cfg(), "train")
        for sample_index in range(len(ds)):
            selected = ds.valid_starts[sample_index]
            end = selected + ds.context_len + ds.horizon_tgt
            steps = np.diff(ds.index[selected:end].as_unit("ns").asi8)
            self.assertTrue(np.all(steps == pd.Timedelta(hours=1).value))

    def test_sample_time_bounds_match_tensor_slices(self):
        index = pd.date_range("2020-01-01", periods=24, freq="1h", tz="UTC")
        ds = DecisionAwareDatasetV3(_frame(index), self._cfg(), "train")
        bounds = ds.sample_time_bounds(0)
        self.assertEqual(bounds["context_start"], index[0])
        self.assertEqual(bounds["context_end"], index[3])
        self.assertEqual(bounds["target_start"], index[4])
        self.assertEqual(bounds["target_end"], index[5])

    def test_future_actual_covariates_are_not_in_context(self):
        index = pd.date_range("2020-01-01", periods=24, freq="1h", tz="UTC")
        original = _frame(index)
        cfg = self._cfg()
        ds = DecisionAwareDatasetV3(original, cfg, "train")
        start = ds.valid_starts[0]
        target_start = start + ds.context_len

        changed = original.copy()
        changed.iloc[target_start:, changed.columns.get_indexer(["load", "wind", "solar"])] += 1e6
        changed_ds = DecisionAwareDatasetV3(
            changed, cfg, "train", norm_stats=ds.norm_stats
        )

        for key in ("load_ctx", "system_ctx"):
            np.testing.assert_array_equal(ds[0][key].numpy(), changed_ds[0][key].numpy())

    def test_normalization_ignores_values_outside_train_period(self):
        index = pd.date_range("2020-01-01", periods=12, freq="1h", tz="UTC")
        original = _frame(index)
        cfg = self._cfg()
        bounds = {
            "train": ("2020-01-01 00:00", "2020-01-01 05:00"),
            "val": ("2020-01-01 06:00", "2020-01-01 08:00"),
            "test": ("2020-01-01 09:00", "2020-01-01 11:00"),
        }
        cfg.split_bounds = lambda split: bounds[split]
        before = DecisionAwareDatasetV3(original, cfg, "train").norm_stats

        changed = original.copy()
        changed.loc[index[6]:, ["price_da", "price_rt", "load", "wind", "solar"]] += 1e6
        after = DecisionAwareDatasetV3(changed, cfg, "train").norm_stats

        self.assertEqual(before["price_da"], after["price_da"])
        self.assertEqual(before["price_rt"], after["price_rt"])
        np.testing.assert_array_equal(before["load"]["mean"], after["load"]["mean"])
        np.testing.assert_array_equal(before["system"]["mean"], after["system"]["mean"])


if __name__ == "__main__":
    unittest.main()
