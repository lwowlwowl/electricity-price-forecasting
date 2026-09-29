import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.config import PilotConfig
from decision_aware.model_da import DecisionAwareDAForecaster
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster


def _config():
    return PilotConfig(
        d_model=32, n_heads_enc=4, n_heads_fusion=4,
        n_layers_enc=1, n_layers_fusion=1, dim_ff=64,
        dropout=0.0, horizon_da=24, pred_clamp=[-500.0, 10000.0],
    )


def _batch(batch_size=2, context=16):
    torch.manual_seed(11)
    return {
        "price_da_ctx": torch.randn(batch_size, context),
        "price_rt_ctx": torch.randn(batch_size, context),
        "load_ctx": torch.randn(batch_size, context, 1),
        "system_ctx": torch.randn(batch_size, context, 2),
        "cal_ctx": torch.randn(batch_size, context, 8),
        "cal_tgt": torch.randn(batch_size, 24, 8),
        "price_rt_mean": torch.full((batch_size,), 35.0),
        "price_rt_std": torch.full((batch_size,), 15.0),
    }


def test_rt_at_da_shapes_and_output_name():
    output = DecisionAwareRTAtDAForecaster(_config()).eval()(_batch())
    assert output["p_rt_at_da"].shape == (2, 24)
    assert output["rep"].shape == (2, 24, 32)
    assert output["memory"].shape == (2, 16, 32)
    assert output["source_weights"].shape == (2, 16, 5)
    assert "p_da" not in output
    assert "p_rt" not in output


def test_da_and_rt_at_da_instances_do_not_share_parameters():
    da_model = DecisionAwareDAForecaster(_config())
    rt_at_da_model = DecisionAwareRTAtDAForecaster(_config())
    assert (
        next(da_model.parameters()).data_ptr()
        != next(rt_at_da_model.parameters()).data_ptr()
    )


def test_rt_at_da_backpropagates_to_every_source_encoder():
    model = DecisionAwareRTAtDAForecaster(_config())
    model(_batch())["p_rt_at_da"].mean().backward()
    for name, encoder in model.encoders.items():
        assert encoder.proj.weight.grad is not None, name
        assert torch.isfinite(encoder.proj.weight.grad).all(), name
