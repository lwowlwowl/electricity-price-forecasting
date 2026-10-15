from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig


def test_universal_lomo_configs_remove_identity_and_hold_out_one_market():
    markets = {"CAISO", "ERCOT", "NYISO", "PJM"}
    for holdout in sorted(markets):
        path = ROOT / (
            "configs/decision_aware/"
            f"universal_lomo_holdout_{holdout.lower()}_rt_at_da_v1.yaml"
        )
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = PilotConfig.from_yaml(str(path))
        assert cfg.global_use_entity_embeddings is False
        assert cfg.global_num_nodes == 0
        assert cfg.global_num_markets == 0
        assert cfg.global_normalization_mode == "context_robust"
        assert set(cfg.global_training_markets) == markets - {holdout}
        assert holdout not in raw["global_training_markets"]
        assert cfg.rt_at_da_fusion_mode == "F0"


def test_universal_lomo_da_rt_configs_use_frozen_mae_contract():
    markets = {"CAISO", "ERCOT", "NYISO", "PJM"}
    for holdout in sorted(markets):
        path = ROOT / (
            "configs/decision_aware/"
            f"universal_lomo_holdout_{holdout.lower()}_mae_v1.yaml"
        )
        cfg = PilotConfig.from_yaml(str(path))
        assert cfg.global_use_entity_embeddings is False
        assert cfg.global_num_nodes == 0
        assert cfg.global_num_markets == 0
        assert cfg.global_normalization_mode == "context_robust"
        assert set(cfg.global_training_markets) == markets - {holdout}
        assert cfg.global_training_loss == "mae"
        assert cfg.da_fusion_mode == "F1"
        assert cfg.rt_fusion_mode == "F1"


def test_ercot_universal_ablation_configs_change_only_requested_fusion_family():
    current_mae = PilotConfig.from_yaml(str(ROOT / (
        "configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml"
    )))
    all_f0 = PilotConfig.from_yaml(str(ROOT / (
        "configs/decision_aware/"
        "universal_lomo_holdout_ercot_ablation_all_f0_mae_v1.yaml"
    )))
    assert all_f0.da_fusion_mode == "F0"
    assert all_f0.rt_fusion_mode == "F0"
    assert all_f0.global_training_loss == current_mae.global_training_loss == "mae"
    assert all_f0.global_training_markets == current_mae.global_training_markets
    assert all_f0.global_samples_per_market == current_mae.global_samples_per_market
    assert all_f0.seed == current_mae.seed == 0

    current_rt_at_da = PilotConfig.from_yaml(str(ROOT / (
        "configs/decision_aware/"
        "universal_lomo_holdout_ercot_rt_at_da_v1.yaml"
    )))
    all_f1 = PilotConfig.from_yaml(str(ROOT / (
        "configs/decision_aware/"
        "universal_lomo_holdout_ercot_ablation_all_f1_rt_at_da_v1.yaml"
    )))
    assert all_f1.rt_at_da_fusion_mode == "F1"
    assert (
        all_f1.global_training_loss
        == current_rt_at_da.global_training_loss
        == "pseudo_huber"
    )
    assert all_f1.global_training_markets == current_rt_at_da.global_training_markets
    assert all_f1.global_samples_per_market == current_rt_at_da.global_samples_per_market
    assert all_f1.seed == current_rt_at_da.seed == 0
