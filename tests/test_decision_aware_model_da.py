import os
import sys

import torch
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.config import PilotConfig
from decision_aware.model_da import DecisionAwareDAForecaster


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
        pred_clamp=[-500.0, 10000.0],
    )


def _batch(batch_size=2, context=16):
    torch.manual_seed(7)
    return {
        "price_da_ctx": torch.randn(batch_size, context),
        "price_rt_ctx": torch.randn(batch_size, context),
        "load_ctx": torch.randn(batch_size, context, 1),
        "system_ctx": torch.randn(batch_size, context, 2),
        "cal_ctx": torch.randn(batch_size, context, 8),
        "cal_tgt": torch.randn(batch_size, 24, 8),
        "price_da_mean": torch.full((batch_size,), 30.0),
        "price_da_std": torch.full((batch_size,), 10.0),
    }


def test_da_model_shapes_and_no_rt_head():
    model = DecisionAwareDAForecaster(_config()).eval()
    output = model(_batch())
    assert output["p_da"].shape == (2, 24)
    assert output["rep"].shape == (2, 24, 32)
    assert output["memory"].shape == (2, 16, 32)
    assert output["source_weights"].shape == (2, 16, 5)
    assert torch.allclose(output["source_weights"].sum(-1), torch.ones(2, 16))
    assert "p_rt" not in output
    assert "p_rt_da" not in output


def test_da_model_backpropagates_to_each_source_encoder():
    model = DecisionAwareDAForecaster(_config())
    loss = model(_batch())["p_da"].mean()
    loss.backward()
    for name, encoder in model.encoders.items():
        assert encoder.proj.weight.grad is not None, name
        assert torch.isfinite(encoder.proj.weight.grad).all(), name


def test_target_calendar_changes_predictions():
    model = DecisionAwareDAForecaster(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        first = model(batch)["p_da"]
        changed = dict(batch)
        changed["cal_tgt"] = batch["cal_tgt"] + 2.0
        second = model(changed)["p_da"]
    assert not torch.allclose(first, second)


def test_two_da_models_do_not_share_parameters():
    first = DecisionAwareDAForecaster(_config())
    second = DecisionAwareDAForecaster(_config())
    first_parameter = next(first.parameters())
    second_parameter = next(second.parameters())
    assert first_parameter.data_ptr() != second_parameter.data_ptr()


@pytest.mark.parametrize(
    ("mode", "expected_memory_steps"),
    [("F0", 16), ("F1", 16), ("F2", 80)],
)
def test_all_fusion_modes_have_valid_shapes_and_gradients(
    mode, expected_memory_steps
):
    cfg = _config()
    cfg.da_fusion_mode = mode
    model = DecisionAwareDAForecaster(cfg)
    output = model(_batch())
    assert output["p_da"].shape == (2, 24)
    assert output["memory"].shape == (2, expected_memory_steps, 32)
    output["p_da"].mean().backward()
    for name, encoder in model.encoders.items():
        gradient = encoder.proj.weight.grad
        assert gradient is not None, (mode, name)
        assert torch.isfinite(gradient).all(), (mode, name)


def test_f2_uses_separate_source_and_time_identity_without_global_rope():
    cfg = _config()
    cfg.da_fusion_mode = "F2"
    model = DecisionAwareDAForecaster(cfg)
    fusion = model.fusion
    assert fusion.modality_embedding.shape == (5, 32)
    assert fusion.time_embedding.shape == (cfg.context_len, 32)
    assert len(fusion.global_blocks) == 2 * cfg.n_layers_fusion
    assert all(block.attn.rope is None for block in fusion.global_blocks)


def test_fusion_parameter_budgets_are_close():
    counts = {}
    for mode in ("F0", "F1", "F2"):
        cfg = _config()
        cfg.da_fusion_mode = mode
        counts[mode] = sum(
            parameter.numel() for parameter in DecisionAwareDAForecaster(cfg).parameters()
        )
    assert max(counts.values()) / min(counts.values()) < 1.10, counts


def test_fusion_models_do_not_share_parameters():
    models = []
    for mode in ("F0", "F1", "F2"):
        cfg = _config()
        cfg.da_fusion_mode = mode
        models.append(DecisionAwareDAForecaster(cfg))
    pointers = [next(model.parameters()).data_ptr() for model in models]
    assert len(set(pointers)) == len(pointers)
