import os
import sys

import pytest
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


@pytest.mark.parametrize(
    ("mode", "expected_memory_steps"),
    [("F0", 16), ("F1", 16), ("F1b", 16), ("F2", 80)],
)
def test_rt_at_da_supports_all_fusion_modes(mode, expected_memory_steps):
    cfg = _config()
    cfg.rt_at_da_fusion_mode = mode
    model = DecisionAwareRTAtDAForecaster(cfg)
    output = model(_batch())
    assert model.fusion_mode in {
        "f0_residual_mlp", "source_attention", "source_attention_concat",
        "f2_global_attention"
    }
    assert output["p_rt_at_da"].shape == (2, 24)
    assert output["memory"].shape == (2, expected_memory_steps, 32)
    output["p_rt_at_da"].mean().backward()
    for name, encoder in model.encoders.items():
        gradient = encoder.proj.weight.grad
        assert gradient is not None, (mode, name)
        assert torch.isfinite(gradient).all(), (mode, name)


def test_rt_at_da_uses_its_own_fusion_setting():
    cfg = _config()
    cfg.da_fusion_mode = "F1"
    cfg.rt_at_da_fusion_mode = "F0"
    model = DecisionAwareRTAtDAForecaster(cfg)
    assert model.fusion_mode == "f0_residual_mlp"


def test_rt_at_da_f1b_removes_scalar_source_pooling():
    cfg = _config()
    cfg.rt_at_da_fusion_mode = "F1b"
    model = DecisionAwareRTAtDAForecaster(cfg)
    output = model(_batch())
    assert hasattr(model.fusion, "source_blocks")
    assert hasattr(model.fusion, "concat_projection")
    assert not hasattr(model.fusion, "source_score")
    assert output["source_weights"] is None


def test_rt_at_da_fusion_parameter_budgets_are_close():
    counts = {}
    for mode in ("F0", "F1", "F1b", "F2"):
        cfg = _config()
        cfg.rt_at_da_fusion_mode = mode
        counts[mode] = sum(
            parameter.numel()
            for parameter in DecisionAwareRTAtDAForecaster(cfg).parameters()
        )
    assert max(counts.values()) / min(counts.values()) < 1.10, counts
