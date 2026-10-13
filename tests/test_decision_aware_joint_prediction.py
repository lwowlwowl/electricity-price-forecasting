from __future__ import annotations

import os
import sys

import pytest
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.joint_prediction import spread_auxiliary_losses  # noqa: E402


def test_exact_spread_has_zero_magnitude_and_correct_direction():
    true_da = torch.tensor([[40.0, 10.0, 50.0]])
    true_rt = torch.tensor([[20.0, 30.0, 49.0]])
    losses = spread_auxiliary_losses(
        true_da,
        true_rt,
        true_da,
        true_rt,
        torch.tensor([10.0]),
        torch.tensor([20.0]),
        direction_threshold=5.0,
    )

    assert losses["spread_magnitude"].item() == pytest.approx(0.0)
    assert losses["spread_direction_accuracy"].item() == pytest.approx(1.0)
    assert losses["important_spread_fraction"].item() == pytest.approx(2 / 3)


def test_wrong_direction_is_penalized_more_than_correct_direction():
    true_da = torch.tensor([[60.0, 0.0]])
    true_rt = torch.tensor([[0.0, 60.0]])
    correct = spread_auxiliary_losses(
        true_da, true_rt, true_da, true_rt,
        torch.tensor([20.0]), torch.tensor([20.0]),
    )
    wrong = spread_auxiliary_losses(
        true_rt, true_da, true_da, true_rt,
        torch.tensor([20.0]), torch.tensor([20.0]),
    )

    assert wrong["spread_direction"] > correct["spread_direction"]
    assert wrong["spread_magnitude"] > correct["spread_magnitude"]


def test_spread_loss_backpropagates_to_both_independent_models():
    p_da = torch.tensor([[5.0, 5.0]], requires_grad=True)
    p_rt = torch.tensor([[4.0, 4.0]], requires_grad=True)
    losses = spread_auxiliary_losses(
        p_da,
        p_rt,
        torch.tensor([[30.0, -20.0]]),
        torch.tensor([[0.0, 10.0]]),
        torch.tensor([10.0]),
        torch.tensor([10.0]),
    )
    (losses["spread_magnitude"] + losses["spread_direction"]).backward()

    assert p_da.grad is not None and torch.isfinite(p_da.grad).all()
    assert p_rt.grad is not None and torch.isfinite(p_rt.grad).all()
    assert torch.allclose(p_da.grad, -p_rt.grad)


def test_near_zero_spreads_do_not_create_unstable_direction_labels():
    p_da = torch.zeros(1, 2, requires_grad=True)
    p_rt = torch.zeros(1, 2, requires_grad=True)
    losses = spread_auxiliary_losses(
        p_da,
        p_rt,
        torch.tensor([[1.0, -1.0]]),
        torch.zeros(1, 2),
        torch.ones(1),
        torch.ones(1),
        direction_threshold=6.0,
    )

    assert losses["spread_direction"].item() == pytest.approx(0.0)
    assert losses["important_spread_fraction"].item() == pytest.approx(0.0)
