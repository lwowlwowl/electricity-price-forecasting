import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.config import PilotConfig
from decision_aware.loss_da import da_decision_aware_loss
from decision_aware.policy import BESSSimulator, HardTopKPolicy


def _objects():
    cfg = PilotConfig(
        horizon_da=24,
        bess_kappa=5.7,
        bess_soc_min=0.4,
        bess_soc_max=3.6,
        bess_e_cyc=4.0,
        proxy_scale=24.0,
        zo_K=1,
    )
    simulator = BESSSimulator(
        1.0, 4.0, 0.95, 0.5, kappa=5.7,
        soc_min=0.4, soc_max=3.6, e_cyc=4.0,
    )
    policy = HardTopKPolicy(
        4, 4, spread_threshold=cfg.resolved_spread_threshold
    )
    return cfg, simulator, policy


def test_prediction_only_loss_backpropagates():
    cfg, simulator, policy = _objects()
    prediction = torch.linspace(10.0, 40.0, 48).reshape(2, 24).requires_grad_()
    target = prediction.detach() + 3.0
    loss, metrics = da_decision_aware_loss(
        prediction, target, simulator, policy, 1.0, 0.0, cfg, epsilon=1.0
    )
    loss.backward()
    assert prediction.grad is not None
    assert metrics["loss_proxy"] == 0.0
    assert metrics["revenue"] == metrics["revenue"]


def test_zero_order_proxy_is_finite_and_backpropagates():
    torch.manual_seed(0)
    cfg, simulator, policy = _objects()
    prediction = torch.randn(2, 24, requires_grad=True) * 20.0 + 30.0
    prediction.retain_grad()
    target = torch.randn(2, 24) * 20.0 + 30.0
    loss, metrics = da_decision_aware_loss(
        prediction, target, simulator, policy, 0.5, 0.5, cfg, epsilon=2.0
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert metrics["zo_gradient_norm"] >= 0.0
