from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from scripts.decision_aware.run_universal_lomo_timesfm3 import (
    CALENDAR_NAMES,
    _window_arrays,
    make_timesfm_task,
    validate_protocol,
)


def _dataset(task: str):
    length = 220
    index = pd.date_range("2023-01-01", periods=length, freq="h", tz="UTC")
    target = np.arange(168, 172 if task == "rt" else 192, dtype=np.int64)
    window = SimpleNamespace(
        context_start=0,
        context_end=168,
        issue_position=168,
        target_positions=target,
    )
    return SimpleNamespace(
        windows=[window],
        index=index,
        price_da=np.arange(length, dtype=np.float32) + 1000,
        price_rt=np.arange(length, dtype=np.float32) - 50,
        streams={"cal": np.arange(length * 8, dtype=np.float32).reshape(length, 8)},
    )


def test_da_and_rt_at_da_only_expose_calendar_future_covariates():
    for task in ("da", "rt_at_da"):
        dataset = _dataset(task)
        context, target, hist_cov, future_cov, names = _window_arrays(
            dataset, task, 0
        )
        expected_price = dataset.price_da if task == "da" else dataset.price_rt
        np.testing.assert_array_equal(context, expected_price[:168])
        np.testing.assert_array_equal(hist_cov, dataset.streams["cal"][:168])
        np.testing.assert_array_equal(future_cov, dataset.streams["cal"][target])
        assert names == CALENDAR_NAMES
        assert hist_cov.shape == (168, 8)
        assert future_cov.shape == (24, 8)


def test_rolling_rt_adds_only_published_da_to_future_calendar():
    dataset = _dataset("rt")
    context, target, hist_cov, future_cov, names = _window_arrays(
        dataset, "rt", 0
    )
    np.testing.assert_array_equal(context, dataset.price_rt[:168])
    np.testing.assert_array_equal(hist_cov[:, -1], dataset.price_da[:168])
    np.testing.assert_array_equal(future_cov[:, -1], dataset.price_da[target])
    assert names[-1] == "published_da_price"
    assert hist_cov.shape == (168, 9)
    assert future_cov.shape == (4, 9)


def test_task_contains_raw_target_history_and_future_timestamps():
    dataset = _dataset("da")
    task = make_timesfm_task(dataset, "da", 0, "NODE_A")
    assert task.context.shape == (168, 1)
    np.testing.assert_array_equal(task.context[:, 0], dataset.price_da[:168])
    assert task.hist_cov.shape == (168, 8)
    assert task.future_cov.shape == (24, 8)
    assert task.index[0] == dataset.index[168]
    assert task.series_names == ["NODE_A"]


def test_protocol_rejects_identity_embeddings_and_wrong_context():
    base = dict(
        context_len=168,
        horizon_da=24,
        horizon_rt=4,
        global_use_entity_embeddings=False,
        global_normalization_mode="context_robust",
        global_training_markets=["CAISO", "NYISO", "PJM"],
    )
    validate_protocol(SimpleNamespace(**base), "da", "ERCOT")
    for override in (
        {"context_len": 24},
        {"global_use_entity_embeddings": True},
        {"global_normalization_mode": "train_stats"},
        {"global_training_markets": ["CAISO", "ERCOT", "PJM"]},
    ):
        cfg = SimpleNamespace(**{**base, **override})
        try:
            validate_protocol(cfg, "da", "ERCOT")
        except ValueError:
            pass
        else:
            raise AssertionError(f"协议错误未被拒绝: {override}")

