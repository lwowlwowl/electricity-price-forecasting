from __future__ import annotations

import pytest
import torch

from decision_aware.spread_factorization import (
    normalized_spread_target,
    reconstruct_rt_from_da_and_spread,
    spread_location_scale,
)


def _batch() -> dict:
    return {
        "price_da_mean": torch.tensor([40.0, 50.0]),
        "price_rt_mean": torch.tensor([35.0, 55.0]),
        "price_da_std": torch.tensor([10.0, 20.0]),
        "price_rt_std": torch.tensor([20.0, 10.0]),
        "price_da_tgt": torch.tensor([[50.0, 60.0], [70.0, 80.0]]),
        "price_rt_tgt": torch.tensor([[30.0, 55.0], [90.0, 75.0]]),
    }


def test_true_normalized_spread_reconstructs_rt_from_true_da():
    batch = _batch()
    target = normalized_spread_target(batch)
    reconstructed, spread = reconstruct_rt_from_da_and_spread(
        batch["price_da_tgt"], target, batch
    )

    torch.testing.assert_close(reconstructed, batch["price_rt_tgt"])
    torch.testing.assert_close(
        spread, batch["price_da_tgt"] - batch["price_rt_tgt"]
    )


def test_spread_scale_is_market_unit_invariant_and_positive():
    center, scale = spread_location_scale(_batch())
    torch.testing.assert_close(center[:, 0], torch.tensor([5.0, -5.0]))
    torch.testing.assert_close(
        scale[:, 0], torch.tensor([500.0**0.5, 500.0**0.5])
    )
    assert bool((scale > 0).all())


def test_spread_target_requires_both_training_labels():
    batch = _batch()
    del batch["price_da_tgt"]
    with pytest.raises(KeyError, match="DA与RT训练标签"):
        normalized_spread_target(batch)
