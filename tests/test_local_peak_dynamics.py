from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from decision_aware.config import PilotConfig
from decision_aware.local_peak_dynamics import (
    daily_peak_excursion,
    daily_peak_excursion_errors,
    daily_peak_excursion_loss,
)
from decision_aware.peak_dynamic_selection import select_peak_dynamic_candidate


ROOT = Path(__file__).resolve().parents[1]


def test_daily_peak_excursion_uses_peak_relative_to_median() -> None:
    curve = torch.arange(24, dtype=torch.float32).reshape(1, 24)
    assert daily_peak_excursion(curve).item() == pytest.approx(23.0 - 11.5)


def test_daily_peak_excursion_loss_is_zero_for_matching_curve() -> None:
    target = torch.linspace(-5.0, 50.0, 24).reshape(1, 24)
    loss = daily_peak_excursion_loss(target, target, torch.tensor([10.0]))
    assert loss.item() == pytest.approx(0.0)


def test_daily_peak_excursion_loss_detects_flattening_and_backpropagates() -> None:
    target = torch.zeros(2, 24)
    target[:, 17] = torch.tensor([100.0, 50.0])
    prediction = (target * 0.1).clone().requires_grad_(True)
    loss = daily_peak_excursion_loss(
        prediction, target, torch.tensor([20.0, 10.0])
    )
    assert loss.item() > 0
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 17].abs().sum().item() > 0


def test_daily_peak_excursion_errors_are_per_day() -> None:
    truth = np.zeros((2, 24), dtype=np.float64)
    prediction = np.zeros_like(truth)
    truth[0, 4] = 100.0
    prediction[0, 4] = 80.0
    truth[1, 9] = 50.0
    prediction[1, 9] = 50.0
    result = daily_peak_excursion_errors(prediction, truth)
    np.testing.assert_allclose(result["actual"], [100.0, 50.0])
    np.testing.assert_allclose(result["predicted"], [80.0, 50.0])
    np.testing.assert_allclose(result["absolute_error"], [20.0, 0.0])


def test_peakdynamic_config_only_changes_preregistered_fields() -> None:
    base_path = ROOT / "configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml"
    candidate_path = ROOT / "configs/decision_aware/universal_lomo_holdout_ercot_da_peakdynamic_v1.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    allowed = {
        "global_use_daily_peak_excursion_loss",
        "global_daily_peak_excursion_weight",
        "global_daily_peak_excursion_beta",
        "global_trainable_scope",
        "lr",
        "epochs",
        "early_stop_patience",
        "global_keep_top_k_checkpoints",
        "global_use_lr_scheduler",
        "checkpoint_dir",
    }
    differences = {
        key for key in set(base) | set(candidate)
        if base.get(key) != candidate.get(key)
    }
    assert differences <= allowed
    cfg = PilotConfig.from_yaml(candidate_path)
    assert cfg.global_use_daily_peak_excursion_loss is True
    assert cfg.global_daily_peak_excursion_weight == 1.0
    assert cfg.global_daily_peak_excursion_beta == 1.0
    assert cfg.global_trainable_scope == "decoder_head"
    assert cfg.lr == pytest.approx(5e-5)
    assert cfg.epochs == 6
    assert cfg.global_keep_top_k_checkpoints == 7
    assert cfg.global_use_lr_scheduler is False


def test_peak_dynamic_selection_applies_mae_guard_before_shape_metric() -> None:
    records = [
        {
            "epoch": 0,
            "validation": {
                "market_macro_mae": 10.0,
                "market_macro_daily_peak_excursion_mae": 20.0,
            },
        },
        {
            "epoch": 1,
            "validation": {
                "market_macro_mae": 10.1,
                "market_macro_daily_peak_excursion_mae": 15.0,
            },
        },
        {
            "epoch": 2,
            "validation": {
                "market_macro_mae": 10.3,
                "market_macro_daily_peak_excursion_mae": 1.0,
            },
        },
    ]
    selected = select_peak_dynamic_candidate(records, mae_guard_fraction=0.02)
    assert selected["epoch"] == 1
    assert selected["eligible_epochs"] == [0, 1]
    assert selected["selection_uses_target_market"] is False
    assert selected["selection_uses_profit"] is False
