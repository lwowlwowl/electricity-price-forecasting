import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.global_data import EntityMappings  # noqa: E402
from scripts.decision_aware.run_global_baselines import (  # noqa: E402
    _cached_predictions,
    _portable_node_key,
    _save_predictions,
    _summary_metadata_matches,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        force=False,
        market="NYISO",
        task="da",
        phase="inner",
        seed=0,
    )


def _mappings() -> EntityMappings:
    return EntityMappings(
        market_to_id={"NYISO": 0},
        node_to_id={"WEST": 0},
        node_to_market={"WEST": "NYISO"},
    )


def test_prediction_cache_is_invalidated_when_issue_hour_changes(tmp_path):
    path = tmp_path / "prediction.npz"
    prediction = np.ones((2, 24), dtype=np.float32)
    target = np.zeros((2, 24), dtype=np.float32)
    cfg = PilotConfig(market="NYISO", da_issue_hour_local=5)
    _save_predictions(
        path,
        prediction,
        target,
        model_name="seasonal",
        node="WEST",
        args=_args(),
        cfg=cfg,
        mappings=_mappings(),
    )
    cached = _cached_predictions(
        path,
        model_name="seasonal",
        node="WEST",
        args=_args(),
        cfg=cfg,
        mappings=_mappings(),
    )
    assert cached is not None
    changed = _cached_predictions(
        path,
        model_name="seasonal",
        node="WEST",
        args=_args(),
        cfg=PilotConfig(market="NYISO", da_issue_hour_local=10),
        mappings=_mappings(),
    )
    assert changed is None


def test_legacy_cache_without_schema_is_never_silently_reused(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        prediction=np.ones((1, 24), dtype=np.float32),
        target=np.zeros((1, 24), dtype=np.float32),
    )
    assert _cached_predictions(
        path,
        model_name="seasonal",
        node="WEST",
        args=_args(),
        cfg=PilotConfig(market="NYISO", da_issue_hour_local=5),
        mappings=_mappings(),
    ) is None


def test_portable_node_key_preserves_safe_names_and_encodes_pjm_separators():
    assert _portable_node_key("TH_NP15_GEN-APND") == "TH_NP15_GEN-APND"
    assert _portable_node_key("HUD VL") == "HUD VL"
    hub = _portable_node_key("HUB:AEP GEN HUB")
    zone = _portable_node_key("ZONE:MID-ATL/APS")
    assert hub == "HUB%3AAEP GEN HUB"
    assert zone == "ZONE%3AMID-ATL%2FAPS"
    assert not any(character in hub for character in '<>:"/\\|?*')
    assert not any(character in zone for character in '<>:"/\\|?*')


def test_legacy_ercot_summary_accepts_missing_new_schema_fields_only_when_contract_matches():
    legacy = {
        "task": "da", "phase": "inner", "smoke_only": False,
        "contract_hash": "contract", "entity_mapping_hash": "mapping",
        "nodes": ["N"], "models": {"seasonal": {"complete": True}},
    }
    kwargs = {
        "task": "da", "market": "ERCOT", "phase": "inner",
        "smoke_only": False, "configured_market": "ERCOT",
        "da_issue_hour_local": 10, "contract_hash": "contract",
        "entity_mapping_hash": "mapping", "nodes": ["N"],
    }
    assert _summary_metadata_matches(legacy, **kwargs)
    assert not _summary_metadata_matches(
        legacy, **{**kwargs, "contract_hash": "changed"}
    )
    assert not _summary_metadata_matches(
        legacy, **{**kwargs, "market": "NYISO"}
    )
