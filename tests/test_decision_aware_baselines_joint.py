from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.baselines_joint import (  # noqa: E402
    JointGRUBaseline,
    flatten_baseline_sample,
    weekly_seasonal_task_predictions,
)


def _sample(batch: int = 2, horizon: int = 24) -> dict:
    return {
        "price_da_ctx": torch.zeros(batch, 168),
        "price_rt_ctx": torch.zeros(batch, 168),
        "load_ctx": torch.zeros(batch, 168, 1),
        "system_ctx": torch.zeros(batch, 168, 2),
        "cal_ctx": torch.zeros(batch, 168, 8),
        "cal_tgt": torch.zeros(batch, horizon, 8),
        "price_da_tgt_known": torch.zeros(batch, horizon),
        "price_da_tgt": torch.ones(batch, horizon),
        "price_rt_at_da_tgt": torch.ones(batch, horizon) * 2,
        "price_rt_tgt": torch.ones(batch, horizon) * 3,
        "price_da_mean": torch.zeros(batch),
        "price_da_std": torch.ones(batch),
        "price_rt_mean": torch.zeros(batch),
        "price_rt_std": torch.ones(batch),
    }


def test_tree_features_do_not_include_future_price_targets():
    sample = {key: value[0] for key, value in _sample(batch=1).items()}
    original = flatten_baseline_sample(sample, "rt")
    sample["price_da_tgt"] = torch.full((24,), 10000.0)
    sample["price_rt_tgt"] = torch.full((24,), -10000.0)
    assert flatten_baseline_sample(sample, "rt") == pytest.approx(original)


@dataclass
class _DailyWindow:
    context_start: int
    context_end: int
    target_positions: np.ndarray


@dataclass
class _RTWindow:
    context_start: int
    issue_position: int
    target_positions: np.ndarray


class _Dataset:
    def __init__(self, window):
        self.windows = [window]
        self.price_da = np.arange(400, dtype=np.float32)
        self.price_rt = self.price_da + 1000

    def __len__(self):
        return 1


def test_weekly_seasonal_uses_t_minus_168_and_never_future():
    daily = _Dataset(_DailyWindow(
        context_start=32,
        context_end=200,
        target_positions=np.arange(214, 238),
    ))
    rolling = _Dataset(_RTWindow(
        context_start=32,
        issue_position=200,
        target_positions=np.arange(200, 204),
    ))
    assert weekly_seasonal_task_predictions(daily, "da")[0] == pytest.approx(
        daily.price_da[np.arange(46, 70)]
    )
    assert weekly_seasonal_task_predictions(
        rolling, "rt"
    )[0] == pytest.approx(rolling.price_rt[np.arange(32, 36)])


@pytest.mark.parametrize(
    ("task", "horizon"), (("da", 24), ("rt_at_da", 24), ("rt", 4))
)
def test_gru_tasks_have_expected_shape_and_price_scale(task: str, horizon: int):
    model = JointGRUBaseline(task, hidden_size=8)
    batch = _sample(batch=2, horizon=horizon)
    normalized = model(batch)
    price = model.to_price(normalized, batch)
    assert normalized.shape == (2, horizon)
    assert price.shape == (2, horizon)
    assert torch.isfinite(price).all()

