import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.train_global_forecaster import (  # noqa: E402
    _configure_trainable_scope,
    _economic_regime_target,
    _elementwise_training_loss,
    _forecast_loss,
    _forward_task_model,
    _normalized_prediction_key,
    _normalized_target,
    _per_market_node_macro_mae,
    _rank_candidate_records,
    _save_ranked_candidates,
    _validation_selection_score,
)
from decision_aware.config import PilotConfig  # noqa: E402


def _record(epoch: int, mae: float) -> dict:
    return {"epoch": epoch, "validation": {"macro_mae": mae}}


def test_rank_candidate_records_uses_mae_then_earlier_epoch():
    history = [_record(1, 10.0), _record(2, 8.0), _record(3, 8.0)]
    assert _rank_candidate_records(history, 2) == [
        {"epoch": 2, "macro_mae": 8.0},
        {"epoch": 3, "macro_mae": 8.0},
    ]


def test_rank_candidate_records_skips_unevaluated_frozen_plan_epochs():
    history = [
        {"epoch": 1, "validation": None},
        {"epoch": 2, "validation": None},
        _record(3, 9.0),
    ]
    assert _rank_candidate_records(history, 2) == [
        {"epoch": 3, "macro_mae": 9.0}
    ]


def test_rank_candidate_records_uses_explicit_market_equal_selection_score():
    history = [
        {"epoch": 1, "validation": {"macro_mae": 5.0}, "selection_score": 7.0},
        {"epoch": 2, "validation": {"macro_mae": 6.0}, "selection_score": 4.0},
    ]
    ranked = _rank_candidate_records(history, 1)
    assert ranked == [
        {"epoch": 2, "macro_mae": 6.0, "selection_score": 4.0}
    ]


def test_validation_selection_can_equal_weight_markets():
    validation = {"macro_mae": 3.0, "market_macro_mae": 5.0}
    assert _validation_selection_score(validation, PilotConfig()) == 3.0
    cfg = PilotConfig(global_validation_aggregation="market_macro")
    assert _validation_selection_score(validation, cfg) == 5.0


def test_market_macro_first_equalizes_nodes_then_markets():
    # CAISO只有一个节点，ERCOT有两个；节点样本数也故意不同。
    per_market = _per_market_node_macro_mae(
        abs_sum=torch.tensor([10.0, 20.0, 900.0]),
        counts=torch.tensor([1, 2, 100]),
        node_market_ids=torch.tensor([0, 1, 1]),
        market_names_by_id=["CAISO", "ERCOT"],
    )
    assert per_market == {"CAISO": 10.0, "ERCOT": 9.5}
    assert sum(per_market.values()) / len(per_market) == 9.75


def test_save_ranked_candidates_keeps_only_top_k(tmp_path):
    history = []
    for epoch, mae in ((1, 10.0), (2, 7.0), (3, 8.0)):
        history.append(_record(epoch, mae))
        payload = {"epoch": epoch, "marker": epoch}
        _save_ranked_candidates(tmp_path, payload, history, keep_top_k=2)

    assert sorted(path.name for path in tmp_path.glob("candidate_epoch_*.pt")) == [
        "candidate_epoch_002.pt",
        "candidate_epoch_003.pt",
    ]
    manifest = json.loads(
        (tmp_path / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert [item["epoch"] for item in manifest] == [2, 3]
    assert all(item["requires_full_settlement_selection"] for item in manifest)


def test_global_forecast_loss_supports_robust_mae_and_pseudo_huber():
    prediction = torch.tensor([0.0, 10.0])
    target = torch.zeros(2)
    mae_cfg = PilotConfig(global_training_loss="mae")
    pseudo_cfg = PilotConfig(global_training_loss="pseudo_huber", global_loss_delta=1.0)
    assert _forecast_loss(prediction, target, mae_cfg).item() == 5.0
    pseudo = _forecast_loss(prediction, target, pseudo_cfg)
    assert torch.isfinite(pseudo)
    assert 0.0 < pseudo.item() < 5.0


def test_global_trainable_scope_freezes_backbone_exactly():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(3, 3)
    model.decoder = torch.nn.Linear(3, 3)
    model.price_head = torch.nn.Linear(3, 1)
    parameters, counts = _configure_trainable_scope(model, "decoder_head")
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(parameter.requires_grad for parameter in model.price_head.parameters())
    assert counts["trainable"] == sum(parameter.numel() for parameter in parameters)


def test_spread_factorized_rt_at_da_uses_spread_supervision_and_frozen_da():
    cfg = PilotConfig(rt_at_da_parameterization="da_minus_spread")
    task = {
        "prediction_normalized": "p_rt_at_da_normalized",
        "target": "price_rt_at_da_tgt",
        "mean": "price_rt_mean",
        "std": "price_rt_std",
    }
    batch = {
        "price_da_mean": torch.tensor([40.0]),
        "price_rt_mean": torch.tensor([30.0]),
        "price_da_std": torch.tensor([10.0]),
        "price_rt_std": torch.tensor([20.0]),
        "price_da_tgt_spread_only": torch.tensor([[50.0, 45.0]]),
        "price_rt_at_da_tgt": torch.tensor([[25.0, 35.0]]),
    }
    target = _normalized_target(batch, task, cfg, "rt_at_da")
    expected = torch.tensor([[15.0, 0.0]]) / (500.0**0.5)
    torch.testing.assert_close(target, expected)
    assert _normalized_prediction_key(task, cfg, "rt_at_da") == (
        "p_da_minus_rt_at_da_normalized"
    )

    class FrozenDA(torch.nn.Module):
        def forward(self, received):
            return {"p_da": torch.full((1, 2), 60.0, requires_grad=True)}

    class SpreadModel(torch.nn.Module):
        def forward(self, received):
            assert "p_da_upstream" in received
            return {"seen": received["p_da_upstream"]}

    output = _forward_task_model(
        SpreadModel(), batch, "rt_at_da", FrozenDA()
    )
    assert output["seen"].requires_grad is False


def test_economic_regime_target_uses_bess_break_even_without_tuned_threshold():
    cfg = PilotConfig(bess_kappa=5.7, bess_eta=0.95)
    batch = {
        "price_da_tgt_spread_only": torch.tensor([[10.0, 10.0, 10.0]]),
        "price_rt_at_da_tgt": torch.tensor([[3.0, 7.0, 17.0]]),
    }
    # κ/η=6：价差+7、+3、-7分别对应正可交易、中性、负可交易。
    assert _economic_regime_target(batch, cfg).tolist() == [[1, 0, 2]]


def test_economic_regime_break_even_includes_efficiency_loss():
    cfg = PilotConfig(bess_kappa=5.7, bess_eta=0.95)
    assert cfg.resolved_spread_threshold == pytest.approx(6.0)
    batch = {
        "price_da_tgt_spread_only": torch.tensor([[5.8, 6.1, -5.8, -6.1]]),
        "price_rt_at_da_tgt": torch.zeros(1, 4),
    }
    assert _economic_regime_target(batch, cfg).tolist() == [[0, 1, 0, 2]]


def test_economic_regime_training_is_pure_three_class_cross_entropy():
    cfg = PilotConfig(
        rt_at_da_parameterization="economic_regime",
        global_training_loss="economic_regime_cross_entropy",
    )
    task = {
        "prediction_normalized": "p_rt_at_da_normalized",
        "target": "price_rt_at_da_tgt",
        "mean": "price_rt_mean",
        "std": "price_rt_std",
    }
    batch = {
        "price_da_tgt_spread_only": torch.tensor([[10.0, 10.0, 10.0]]),
        "price_rt_at_da_tgt": torch.tensor([[3.0, 7.0, 17.0]]),
    }
    logits = torch.tensor([[[0.0, 4.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0]]])
    loss, target = _elementwise_training_loss(
        {"economic_regime_logits": logits}, batch, task, cfg, "rt_at_da"
    )
    assert target.tolist() == [[1, 0, 2]]
    assert loss.shape == target.shape
    assert bool((loss < 0.1).all())


def test_economic_regime_scope_freezes_everything_except_linear_head():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(3, 3)
    model.price_head = torch.nn.Linear(3, 1)
    model.economic_regime_head = torch.nn.Linear(4, 3)
    parameters, counts = _configure_trainable_scope(model, "economic_regime")
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.price_head.parameters())
    assert all(
        parameter.requires_grad for parameter in model.economic_regime_head.parameters()
    )
    assert counts["trainable"] == sum(parameter.numel() for parameter in parameters)
