import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.config import PilotConfig
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster
from decision_aware.scarcity_decomposition import PhysicalScarcityResidual


def _batch(batch_size=2, context=168):
    torch.manual_seed(19)
    return {
        "price_da_ctx": torch.randn(batch_size, context),
        "price_rt_ctx": torch.randn(batch_size, context),
        "load_ctx": torch.randn(batch_size, context, 1),
        "system_ctx": torch.randn(batch_size, context, 2),
        "cal_ctx": torch.randn(batch_size, context, 8),
        "cal_tgt": torch.randn(batch_size, 24, 8),
        "physical_pressure_ctx": torch.randn(batch_size, context),
        "physical_pressure_tgt_lag168": torch.randn(batch_size, 24),
        "price_rt_mean": torch.full((batch_size,), 35.0),
        "price_rt_std": torch.full((batch_size,), 15.0),
        "price_da_mean": torch.full((batch_size,), 45.0),
        "price_da_std": torch.full((batch_size,), 20.0),
    }


def _config():
    return PilotConfig(
        d_model=32,
        n_heads_enc=4,
        n_heads_fusion=4,
        n_layers_enc=1,
        n_layers_fusion=1,
        dim_ff=64,
        dropout=0.0,
        horizon_da=24,
        global_output_decomposition="physical_scarcity_residual",
    )


def test_zero_initialized_residual_exactly_preserves_base_prediction():
    cfg = _config()
    structured = DecisionAwareRTAtDAForecaster(cfg).eval()
    base_cfg = _config()
    base_cfg.global_output_decomposition = "direct"
    base = DecisionAwareRTAtDAForecaster(base_cfg).eval()
    base.load_state_dict(
        {
            key: value
            for key, value in structured.state_dict().items()
            if not key.startswith("scarcity_residual.")
        },
        strict=True,
    )
    batch = _batch()
    with torch.no_grad():
        expected = base(batch)["p_rt_at_da"]
        actual = structured(batch)["p_rt_at_da"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_pressure_gate_is_monotone_in_each_physical_feature():
    module = PhysicalScarcityResidual(d_model=8).eval()
    representation = torch.ones(1, 24, 8)
    low = _batch(batch_size=1)
    high = {key: value.clone() for key, value in low.items()}
    high["physical_pressure_ctx"] += 2.0
    high["physical_pressure_tgt_lag168"] += 2.0
    with torch.no_grad():
        low_gate = module(representation, low)["gate"]
        high_gate = module(representation, high)["gate"]
    assert bool((high_gate >= low_gate).all())


def test_only_scarcity_branch_changes_after_one_frozen_base_step():
    model = DecisionAwareRTAtDAForecaster(_config())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("scarcity_residual."))
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    loss = model(_batch())["p_rt_at_da"].mean()
    loss.backward()
    optimizer.step()
    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    }
    assert changed
    assert all(name.startswith("scarcity_residual.") for name in changed)
