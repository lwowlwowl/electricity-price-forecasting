import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig
from decision_aware.model_da import DecisionAwareDAForecaster
from decision_aware.model_rt import DecisionAwareRTForecaster
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster


def _config(mode="post_fusion"):
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
        global_use_entity_embeddings=True,
        global_num_nodes=3,
        global_num_markets=2,
        global_entity_conditioning_mode=mode,
    )


def _history(batch_size=2):
    torch.manual_seed(41)
    return {
        "price_da_ctx": torch.randn(batch_size, 16),
        "price_rt_ctx": torch.randn(batch_size, 16),
        "load_ctx": torch.randn(batch_size, 16, 1),
        "system_ctx": torch.randn(batch_size, 16, 2),
        "cal_ctx": torch.randn(batch_size, 16, 8),
        "node_id": torch.tensor([0, 1]),
        "market_id": torch.tensor([0, 1]),
    }


def _da_batch():
    batch = _history()
    batch.update({
        "cal_tgt": torch.randn(2, 24, 8),
        "price_da_mean": torch.full((2,), 30.0),
        "price_da_std": torch.full((2,), 10.0),
        "price_rt_mean": torch.full((2,), 35.0),
        "price_rt_std": torch.full((2,), 12.0),
    })
    return batch


def _rt_batch():
    batch = _history()
    batch.update({
        "price_da_tgt_known": torch.randn(2, 4),
        "cal_tgt": torch.randn(2, 4, 8),
        "price_rt_mean": torch.full((2,), 35.0),
        "price_rt_std": torch.full((2,), 12.0),
    })
    return batch


@pytest.mark.parametrize(
    ("model_class", "batch_factory", "output_name"),
    [
        (DecisionAwareDAForecaster, _da_batch, "p_da"),
        (DecisionAwareRTAtDAForecaster, _da_batch, "p_rt_at_da"),
        (DecisionAwareRTForecaster, _rt_batch, "p_rt"),
    ],
)
def test_three_global_models_use_trainable_entity_embeddings(
    model_class, batch_factory, output_name
):
    model = model_class(_config())
    output = model(batch_factory())[output_name]
    output.mean().backward()

    conditioner = model.entity_conditioner
    assert conditioner is not None
    assert conditioner.node_embedding.weight.grad is not None
    assert conditioner.market_embedding.weight.grad is not None
    assert torch.isfinite(conditioner.node_embedding.weight.grad).all()


def test_entity_identity_changes_prediction_for_identical_inputs():
    model = DecisionAwareDAForecaster(_config()).eval()
    batch = _da_batch()
    # 两个样本所有连续输入完全相同，只保留不同身份。
    for key, value in list(batch.items()):
        if torch.is_tensor(value) and key not in {"node_id", "market_id"}:
            batch[key] = value[:1].repeat((2,) + (1,) * (value.ndim - 1))
    with torch.no_grad():
        prediction = model(batch)["p_da"]
    assert not torch.allclose(prediction[0], prediction[1])


def test_global_model_rejects_missing_or_out_of_range_ids():
    model = DecisionAwareDAForecaster(_config())
    missing = _da_batch()
    missing.pop("node_id")
    with pytest.raises(ValueError, match="node_id"):
        model(missing)

    invalid = _da_batch()
    invalid["node_id"] = torch.tensor([0, 99])
    with pytest.raises(ValueError, match="node_id"):
        model(invalid)


def test_legacy_single_node_model_has_no_extra_embedding_parameters():
    cfg = _config()
    cfg.global_use_entity_embeddings = False
    model = DecisionAwareDAForecaster(cfg)
    assert model.entity_conditioner is None
    assert not any("node_embedding" in name for name, _ in model.named_parameters())


def test_identity_free_model_accepts_batch_without_node_or_market_id():
    cfg = _config()
    cfg.global_use_entity_embeddings = False
    cfg.global_num_nodes = 0
    cfg.global_num_markets = 0
    model = DecisionAwareDAForecaster(cfg).eval()
    batch = _da_batch()
    batch.pop("node_id")
    batch.pop("market_id")
    with torch.no_grad():
        prediction = model(batch)["p_da"]
    assert prediction.shape == (2, 24)
    assert torch.isfinite(prediction).all()


def test_pre_encoder_conditioning_changes_stream_representation_before_fusion():
    model = DecisionAwareDAForecaster(_config("pre_encoder")).eval()
    batch = _da_batch()
    for key, value in list(batch.items()):
        if torch.is_tensor(value) and key not in {"node_id", "market_id"}:
            batch[key] = value[:1].repeat((2,) + (1,) * (value.ndim - 1))
    captured = []
    hook = model.encoders["price_da"].register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach())
    )
    with torch.no_grad():
        model(batch)
    hook.remove()
    assert len(captured) == 1
    assert not torch.allclose(captured[0][0], captured[0][1])


def test_post_fusion_keeps_stream_representation_entity_agnostic():
    model = DecisionAwareDAForecaster(_config("post_fusion")).eval()
    batch = _da_batch()
    for key, value in list(batch.items()):
        if torch.is_tensor(value) and key not in {"node_id", "market_id"}:
            batch[key] = value[:1].repeat((2,) + (1,) * (value.ndim - 1))
    captured = []
    hook = model.encoders["price_da"].register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach())
    )
    with torch.no_grad():
        model(batch)
    hook.remove()
    assert len(captured) == 1
    assert torch.allclose(captured[0][0], captured[0][1])


def test_unknown_entity_conditioning_mode_is_rejected():
    with pytest.raises(ValueError, match="global_entity_conditioning_mode"):
        DecisionAwareDAForecaster(_config("unknown"))
