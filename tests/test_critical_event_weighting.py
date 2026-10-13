from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.critical_event_weighting import (
    CriticalEventThresholds,
    critical_event_weights,
    fit_critical_event_thresholds,
    validate_critical_event_config,
    weighted_elementwise_mean,
)


def _cfg(**updates):
    values = {
        "global_event_price_lower_quantile": 0.10,
        "global_event_price_upper_quantile": 0.90,
        "global_event_spread_quantile": 0.90,
        "global_event_price_weight": 1.0,
        "global_event_spread_weight": 1.0,
        "global_event_sign_reversal_weight": 0.5,
        "global_event_rt_first_step_weight": 0.5,
        "global_event_max_weight": 3.0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _thresholds():
    return CriticalEventThresholds(
        price_lower=torch.tensor([-10.0, -20.0]),
        price_upper=torch.tensor([100.0, 200.0]),
        abs_spread=torch.tensor([30.0, 50.0]),
    )


def test_normal_da_sample_keeps_unit_weight():
    batch = {
        "node_id": torch.tensor([0]),
        "price_da_tgt": torch.tensor([[20.0, 40.0]]),
        "price_rt_tgt": torch.tensor([[25.0, 35.0]]),
    }
    weights = critical_event_weights(batch, "da", _thresholds(), _cfg())
    assert torch.equal(weights, torch.ones_like(weights))


def test_tail_spread_and_sign_reversal_increase_weight_and_cap_it():
    batch = {
        "node_id": torch.tensor([0]),
        "price_da_tgt": torch.tensor([[150.0, -20.0, 40.0]]),
        "price_rt_at_da_tgt": torch.tensor([[-30.0, 20.0, 35.0]]),
    }
    weights = critical_event_weights(batch, "rt_at_da", _thresholds(), _cfg())
    assert weights[0, 0].item() > 1.0
    assert weights[0, 1].item() > 1.0
    assert weights.max().item() == pytest.approx(3.0)
    assert weights[0, 2].item() == pytest.approx(1.0)


def test_rt_first_executed_step_gets_extra_weight():
    batch = {
        "node_id": torch.tensor([1]),
        "price_da_tgt": torch.tensor([[20.0, 20.0, 20.0, 20.0]]),
        "price_rt_tgt": torch.tensor([[20.0, 20.0, 20.0, 20.0]]),
    }
    weights = critical_event_weights(batch, "rt", _thresholds(), _cfg())
    assert weights.tolist() == [[1.5, 1.0, 1.0, 1.0]]


def test_weighted_mean_is_normalized_by_sum_of_weights():
    loss = torch.tensor([[1.0, 3.0]])
    weights = torch.tensor([[1.0, 3.0]])
    assert weighted_elementwise_mean(loss, weights).item() == pytest.approx(2.5)


def test_thresholds_are_fitted_separately_for_each_training_node():
    class TinyDataset:
        samples = [
            {
                "node_id": torch.tensor(0),
                "price_da_tgt": torch.tensor([0.0, 10.0]),
                "price_rt_tgt": torch.tensor([0.0, 30.0]),
            },
            {
                "node_id": torch.tensor(1),
                "price_da_tgt": torch.tensor([100.0, 200.0]),
                "price_rt_tgt": torch.tensor([50.0, 300.0]),
            },
        ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    thresholds = fit_critical_event_thresholds(
        TinyDataset(), "da", _cfg(), node_count=2
    )
    assert thresholds.price_upper[0] < thresholds.price_lower[1]
    assert thresholds.abs_spread[0] < thresholds.abs_spread[1]


def test_invalid_quantiles_and_cap_are_rejected():
    with pytest.raises(ValueError, match="lower"):
        validate_critical_event_config(_cfg(
            global_event_price_lower_quantile=0.9,
            global_event_price_upper_quantile=0.1,
        ))
    with pytest.raises(ValueError, match="不能小于1"):
        validate_critical_event_config(_cfg(global_event_max_weight=0.5))
