from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402


EXPECTED_HOURS = {"CAISO": 10, "ERCOT": 10, "NYISO": 5, "PJM": 11}


def test_multimarket_task_configs_freeze_equal_market_contracts():
    expected = {
        "global_multimarket_da_mae_v1.yaml": "mae",
        "global_multimarket_rt_at_da_pseudohuber_v1.yaml": "pseudo_huber",
        "global_multimarket_rt_mae_v1.yaml": "mae",
    }
    checkpoint_dirs = set()
    for name, loss in expected.items():
        cfg = PilotConfig.from_yaml(ROOT / "configs/decision_aware" / name)
        assert cfg.global_training_markets == ["CAISO", "ERCOT", "NYISO", "PJM"]
        assert cfg.global_sampling_mode == "market_node_balanced"
        assert cfg.global_validation_aggregation == "market_macro"
        assert cfg.global_num_nodes == 64
        assert cfg.global_num_markets == 4
        assert cfg.global_training_loss == loss
        assert cfg.global_residual_anchor == "none"
        assert {
            market: cfg.resolved_da_issue_hour_local(
                market, require_explicit=True
            )
            for market in cfg.global_training_markets
        } == EXPECTED_HOURS
        checkpoint_dirs.add(cfg.checkpoint_dir)
    assert len(checkpoint_dirs) == 3


def test_multimarket_entitypre_configs_change_only_interpretable_conditioning():
    expected = {
        "global_multimarket_entitypre_da_mae_v1.yaml": "mae",
        "global_multimarket_entitypre_rt_at_da_pseudohuber_v1.yaml": "pseudo_huber",
        "global_multimarket_entitypre_rt_mae_v1.yaml": "mae",
    }
    for name, loss in expected.items():
        cfg = PilotConfig.from_yaml(ROOT / "configs/decision_aware" / name)
        assert cfg.global_training_markets == ["CAISO", "ERCOT", "NYISO", "PJM"]
        assert cfg.global_entity_conditioning_mode == "pre_encoder"
        assert cfg.global_sampling_mode == "market_node_balanced"
        assert cfg.global_validation_aggregation == "market_macro"
        assert cfg.global_training_loss == loss
        assert cfg.global_use_critical_event_weighting is False
        assert cfg.global_residual_anchor == "none"
        assert cfg.global_num_nodes == 64
        assert cfg.global_num_markets == 4


def test_remaining_cloud_runner_uses_entitypre_configs_and_skips_completed_da():
    script = (
        ROOT
        / "scripts/decision_aware/run_cloud_global_multimarket_entitypre_remaining_v1.sh"
    ).read_text(encoding="utf-8")
    assert "global_multimarket_entitypre_rt_at_da_pseudohuber_v1.yaml" in script
    assert "global_multimarket_entitypre_rt_mae_v1.yaml" in script
    assert "global_multimarket_entitypre_da_mae_v1.yaml" not in script
    assert "global_multimarket_rt_at_da_pseudohuber_v1.yaml rt_at_da" not in script
    assert "global_multimarket_rt_mae_v1.yaml rt" not in script


def test_multimarket_issue_hour_contract_refuses_silent_ercot_fallback():
    cfg = PilotConfig(
        market="ERCOT",
        da_issue_hour_local=10,
        da_issue_hour_local_by_market={"ERCOT": 10},
    )
    with pytest.raises(ValueError, match="NYISO"):
        cfg.resolved_da_issue_hour_local("NYISO", require_explicit=True)


def test_non_ercot_prediction_contracts_have_market_specific_origins():
    expected = {"CAISO": 10, "NYISO": 5, "PJM": 11}
    for market, hour in expected.items():
        cfg = PilotConfig.from_yaml(
            ROOT / "configs/decision_aware"
            / f"global_{market.lower()}_prediction_contract_v1.yaml"
        )
        assert cfg.market == market
        assert cfg.resolved_da_issue_hour_local() == hour
        assert cfg.use_dual_settlement is True
        assert cfg.use_deviation_penalty is True


@pytest.mark.parametrize("bad_hour", [-1, 24, 10.5, True])
def test_market_issue_hour_must_be_valid_local_clock_hour(bad_hour):
    cfg = PilotConfig(da_issue_hour_local_by_market={"PJM": bad_hour})
    with pytest.raises(ValueError, match="0—23"):
        cfg.resolved_da_issue_hour_local("PJM", require_explicit=True)
