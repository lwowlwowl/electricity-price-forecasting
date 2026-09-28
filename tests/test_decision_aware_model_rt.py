import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.config import PilotConfig
from decision_aware.model_da import DecisionAwareDAForecaster
from decision_aware.model_rt import DecisionAwareRTForecaster


def _config():
    return PilotConfig(
        context_len=16,
        horizon_da=24,
        horizon_rt=4,
        d_model=32,
        n_heads_enc=4,
        n_heads_fusion=4,
        n_layers_enc=1,
        n_layers_fusion=1,
        dim_ff=64,
        dropout=0.0,
    )


def _batch(batch_size=2):
    torch.manual_seed(11)
    return {
        "price_da_ctx": torch.randn(batch_size, 16),
        "price_rt_ctx": torch.randn(batch_size, 16),
        "load_ctx": torch.randn(batch_size, 16, 1),
        "system_ctx": torch.randn(batch_size, 16, 2),
        "cal_ctx": torch.randn(batch_size, 16, 8),
        "price_da_tgt_known": torch.randn(batch_size, 4),
        "cal_tgt": torch.randn(batch_size, 4, 8),
        "price_rt_mean": torch.full((batch_size,), 35.0),
        "price_rt_std": torch.full((batch_size,), 12.0),
    }


def test_rt_model_shapes_and_no_da_head():
    model = DecisionAwareRTForecaster(_config()).eval()
    output = model(_batch())
    assert output["p_rt"].shape == (2, 4)
    assert output["rep"].shape == (2, 4, 32)
    assert output["memory"].shape == (2, 16, 32)
    assert output["source_weights"].shape == (2, 16, 5)
    assert "p_da" not in output


def test_rt_model_backpropagates_to_all_streams():
    model = DecisionAwareRTForecaster(_config())
    model(_batch())["p_rt"].mean().backward()
    for name, encoder in model.encoders.items():
        assert encoder.proj.weight.grad is not None, name


def test_da_and_rt_models_share_no_trainable_weight():
    cfg = _config()
    da = DecisionAwareDAForecaster(cfg)
    rt = DecisionAwareRTForecaster(cfg)
    da_pointers = {parameter.data_ptr() for parameter in da.parameters()}
    rt_pointers = {parameter.data_ptr() for parameter in rt.parameters()}
    assert da_pointers.isdisjoint(rt_pointers)


def test_known_da_target_changes_rt_forecast():
    model = DecisionAwareRTForecaster(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        first = model(batch)["p_rt"]
        changed = dict(batch)
        changed["price_da_tgt_known"] = batch["price_da_tgt_known"] + 3.0
        second = model(changed)["p_rt"]
    assert not torch.allclose(first, second)
