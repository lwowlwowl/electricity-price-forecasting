import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig
from scripts.decision_aware.train_joint_decision_aware import (
    _check_contract,
    _configure_trainable_scope,
    _effective_proxy_names,
    _resolve_trainable_models,
    _rt_business_signal_active,
)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.price_head = torch.nn.Linear(2, 1)


def test_da_only_and_zero_rt_topk_expose_only_da_revenue_proxy():
    cfg = PilotConfig(
        joint_da_signal_mode="da_only",
        joint_da_proxy_mode="da_only",
        joint_coordination_mode="plan_track_topk",
        rt_topk_k_charge=0,
        rt_topk_k_discharge=0,
    )

    assert not _rt_business_signal_active(cfg)
    assert _effective_proxy_names(cfg, ("da",)) == ("da",)
    assert _effective_proxy_names(cfg, ("da", "rt")) == ("da",)


def test_spread_proxy_and_rt_proxy_follow_execution_contract():
    cfg = PilotConfig(
        joint_da_signal_mode="spread",
        joint_da_proxy_mode="coupled_spread",
        joint_coordination_mode="plan_track_topk",
        rt_topk_k_charge=1,
        rt_topk_k_discharge=0,
    )

    assert _rt_business_signal_active(cfg)
    assert _effective_proxy_names(
        cfg, ("da", "rt_at_da", "rt")
    ) == ("spread", "rt")

    cfg.joint_coordination_mode = "follow_da"
    assert not _rt_business_signal_active(cfg)
    assert _effective_proxy_names(
        cfg, ("da", "rt_at_da", "rt")
    ) == ("spread",)


def test_explicit_trainable_models_fully_freeze_unlisted_models():
    models = {name: _TinyModel() for name in ("da", "rt_at_da", "rt")}
    selected = _resolve_trainable_models(["da"])
    counts = _configure_trainable_scope(models, "price_head", selected)

    assert counts["da"] > 0
    assert counts["rt_at_da"] == 0
    assert counts["rt"] == 0
    assert all(
        not parameter.requires_grad
        for name in ("rt_at_da", "rt")
        for parameter in models[name].parameters()
    )
    assert all(
        parameter.requires_grad == parameter_name.startswith("price_head.")
        for parameter_name, parameter in models["da"].named_parameters()
    )


def test_source_kappa_mismatch_requires_explicit_sensitivity_opt_in():
    experiment = PilotConfig(bess_kappa=9.5)
    source = PilotConfig(bess_kappa=5.7)

    with pytest.raises(ValueError, match="bess_kappa"):
        _check_contract(experiment, source, "da")
    _check_contract(
        experiment, source, "da", allow_kappa_mismatch=True
    )


def test_kappa_opt_in_does_not_hide_other_physical_mismatch():
    experiment = PilotConfig(bess_kappa=9.5, bess_eta=0.95)
    source = PilotConfig(bess_kappa=5.7, bess_eta=0.90)

    with pytest.raises(ValueError, match="bess_eta"):
        _check_contract(
            experiment, source, "da", allow_kappa_mismatch=True
        )

