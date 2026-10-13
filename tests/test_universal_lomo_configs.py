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
