import json
from pathlib import Path

import numpy as np
import pytest

from scripts.decision_aware import audit_paper_gap_artifacts as module


def _write_archive(path: Path, *, protocol: str, holdout: str, task: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": protocol,
        "holdout_market": holdout.upper(),
        "task": task,
        "training_markets": module.EXPECTED_TRAINING_MARKETS[holdout],
        "uses_node_id_for_prediction": False,
        "uses_market_id_for_prediction": False,
        "uses_target_market_training_labels": False,
        "node_macro_mae": 1.0,
    }
    np.savez_compressed(
        path,
        nodes=np.asarray(["N1"]),
        dates=np.asarray(["2023-01-01"]),
        prediction=np.ones((1, 1, 2), dtype=np.float32),
        target=np.zeros((1, 1, 2), dtype=np.float32),
        metadata=json.dumps(metadata),
    )
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")


def test_archive_record_accepts_zero_shot_contract(tmp_path):
    path = tmp_path / "holdout_ercot_da_inner_seed0.npz"
    _write_archive(
        path,
        protocol="pooled_lomo_xgboost_equal_training_years",
        holdout="ercot",
        task="da",
    )
    record, errors = module._archive_record(
        path,
        holdout="ercot",
        task="da",
        protocol="pooled_lomo_xgboost_equal_training_years",
        require_no_target_labels=True,
    )
    assert not errors
    assert record["node_macro_mae_recomputed"] == pytest.approx(1.0)
    assert record["years"] == ["2023"]


def test_archive_record_rejects_2025_and_identity_leak(tmp_path):
    path = tmp_path / "bad.npz"
    metadata = {
        "protocol": "leave_one_market_out_zero_shot",
        "holdout_market": "ERCOT",
        "task": "rt",
        "training_markets": ["CAISO", "NYISO", "PJM"],
        "uses_node_id_for_prediction": True,
        "uses_market_id_for_prediction": False,
    }
    np.savez_compressed(
        path,
        nodes=np.asarray(["N1"]),
        dates=np.asarray(["2025-01-01"]),
        prediction=np.zeros((1, 1, 1)),
        target=np.zeros((1, 1, 1)),
        metadata=json.dumps(metadata),
    )
    _, errors = module._archive_record(
        path,
        holdout="ercot",
        task="rt",
        protocol="leave_one_market_out_zero_shot",
    )
    assert any("2025" in item for item in errors)
    assert any("node_id" in item for item in errors)


def test_settlement_summary_selects_only_homogeneous_systems(tmp_path):
    path = tmp_path / "settlement.json"
    systems = {}
    for family, value in (("current", 2.0), ("all_f0", 1.0), ("all_f1", 3.0)):
        systems[f"da_{family}__rt_at_da_{family}__rt_{family}"] = {
            "macro_mean_daily_revenue": value,
            "macro_mean_daily_revenue_7day_block_bootstrap_95ci": [value - 1, value + 1],
        }
    path.write_text(json.dumps({"split": "2023 inner validation only", "systems": systems}), encoding="utf-8")
    result, errors = module._settlement_summary(path, ("current", "all_f0", "all_f1"))
    assert not errors
    assert result["selected_systems"]["all_f1"]["macro_mean_daily_revenue"] == 3.0
