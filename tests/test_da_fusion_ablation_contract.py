from __future__ import annotations

from argparse import Namespace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.decision_aware import train_da_fusion_ablation as training

PilotConfig = training.PilotConfig


def _dataset(delivery_date: date = date(2025, 7, 1)):
    class Dataset:
        windows = [SimpleNamespace(delivery_date=delivery_date)]

        def __len__(self):
            return len(self.windows)

    return Dataset()


def test_legacy_evaluation_contract_stays_backward_compatible():
    contract = training._resolve_evaluation_contract(PilotConfig())

    assert contract == {
        "coordination_mode": "rt_only",
        "da_signal_mode": "spread",
        "rt_k_charge": 1,
        "rt_k_discharge": 1,
        "deviation_penalty_enabled": False,
    }


def test_penalty_da_contract_maps_joint_fields_and_rt_topk():
    cfg = PilotConfig(
        use_deviation_penalty=True,
        joint_coordination_mode="follow_da",
        joint_da_signal_mode="da_only",
        rt_topk_k_charge=0,
        rt_topk_k_discharge=0,
    )

    assert training._resolve_evaluation_contract(cfg) == {
        "coordination_mode": "follow_da",
        "da_signal_mode": "da_only",
        "rt_k_charge": 0,
        "rt_k_discharge": 0,
        "deviation_penalty_enabled": True,
    }


def test_conflicting_coordination_fields_are_rejected():
    cfg = PilotConfig(
        frozen_coordination_mode="follow_da",
        joint_coordination_mode="plan_track_topk",
    )

    with pytest.raises(ValueError, match="冲突"):
        training._resolve_evaluation_contract(cfg)


def test_da_only_evaluator_omits_rt_at_da_and_uses_configured_rt_k(monkeypatch):
    cfg = PilotConfig(
        use_deviation_penalty=True,
        joint_coordination_mode="follow_da",
        joint_da_signal_mode="da_only",
        rt_topk_k_charge=0,
        rt_topk_k_discharge=0,
    )
    evaluator = training.FrozenDualEvaluator.__new__(training.FrozenDualEvaluator)
    evaluator.cfg = cfg
    evaluator.contract = training._resolve_evaluation_contract(cfg)
    evaluator.da_datasets = {"validation": _dataset()}
    evaluator.rt_da_datasets = {}
    evaluator.fixed = {
        "validation": {
            "rt": np.zeros((1, 4), dtype=np.float32),
            "realized_da": torch.zeros(1),
            "realized_rt": torch.zeros(1),
            "timestamps": ["unused"],
        }
    }
    captured = {}

    def fake_backtest(*args, **kwargs):
        captured.update(kwargs)
        return {"mean_daily_revenue": 0.0}

    monkeypatch.setattr(training, "locked_dual_backtest", fake_backtest)
    evaluator.evaluate("validation", np.zeros((1, 24), dtype=np.float32))

    assert captured["coordination_mode"] == "follow_da"
    assert captured["rt_k_charge"] == 0
    assert captured["rt_k_discharge"] == 0
    assert captured["rt_at_da_price_forecasts"] is None


def test_spread_evaluator_passes_rt_at_da_mapping(monkeypatch):
    cfg = PilotConfig(joint_da_signal_mode="spread")
    evaluator = training.FrozenDualEvaluator.__new__(training.FrozenDualEvaluator)
    evaluator.cfg = cfg
    evaluator.contract = training._resolve_evaluation_contract(cfg)
    evaluator.da_datasets = {"validation": _dataset()}
    evaluator.rt_da_datasets = {"validation": _dataset()}
    evaluator.fixed = {
        "validation": {
            "rt": np.zeros((1, 4), dtype=np.float32),
            "rt_at_da": np.ones((1, 24), dtype=np.float32),
            "realized_da": torch.zeros(1),
            "realized_rt": torch.zeros(1),
            "timestamps": ["unused"],
        }
    }
    captured = {}

    def fake_backtest(*args, **kwargs):
        captured.update(kwargs)
        return {"mean_daily_revenue": 0.0}

    monkeypatch.setattr(training, "locked_dual_backtest", fake_backtest)
    evaluator.evaluate("validation", np.zeros((1, 24), dtype=np.float32))

    mapping = captured["rt_at_da_price_forecasts"]
    assert list(mapping) == [date(2025, 7, 1)]
    assert np.array_equal(mapping[date(2025, 7, 1)], np.ones(24))


def test_validation_day_contract_rejects_wrong_split():
    cfg = PilotConfig(frozen_expected_validation_days=182)

    training._check_validation_day_contract(cfg, {"days": 182})
    with pytest.raises(ValueError, match="182"):
        training._check_validation_day_contract(cfg, {"days": 179})


def test_checkpoint_payload_records_contract_and_report_only_state():
    cfg = PilotConfig(
        use_deviation_penalty=True,
        joint_coordination_mode="follow_da",
        joint_da_signal_mode="da_only",
        rt_topk_k_charge=0,
        rt_topk_k_discharge=0,
    )
    args = Namespace(
        fusion_mode="F2", task="da", skip_report_only=True
    )

    payload = training._training_checkpoint_payload(
        model_state={"weight": torch.ones(1)},
        cfg=cfg,
        norm_stats={},
        args=args,
        epoch=2,
        history=[{"epoch": 1}, {"epoch": 2}],
        best_epoch=1,
        best_revenue=12.5,
    )

    assert payload["epoch"] == 2
    assert payload["best_epoch"] == 1
    assert payload["report_only_evaluated"] is False
    assert payload["evaluation_contract"]["coordination_mode"] == "follow_da"
    assert payload["evaluation_contract"]["rt_k_charge"] == 0


def test_penalty_da_config_freezes_the_182_day_selection_contract():
    cfg = PilotConfig.from_yaml(
        "configs/decision_aware/da_fusion_ablation_v2_penalty.yaml"
    )

    assert cfg.use_deviation_penalty is True
    assert cfg.joint_coordination_mode == "follow_da"
    assert cfg.joint_da_signal_mode == "da_only"
    assert (cfg.topk_k_charge, cfg.topk_k_discharge) == (3, 3)
    assert (cfg.rt_topk_k_charge, cfg.rt_topk_k_discharge) == (0, 0)
    assert cfg.frozen_expected_validation_days == 182
