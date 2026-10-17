from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dlinear import DLinearForecaster  # noqa: E402
from scripts.decision_aware.run_universal_lomo_timesfm3 import validate_protocol  # noqa: E402


def test_baseline_contract_is_pre_2025_and_fixed() -> None:
    path = ROOT / "configs/decision_aware/allmarket_final_baselines_v1.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert value["training_years"] == [2020, 2021, 2022, 2023, 2024]
    assert value["locked_test_year"] == 2025
    assert value["selection_uses_2025"] is False
    assert value["markets"] == ["CAISO", "ERCOT", "NYISO", "PJM"]
    assert value["timesfm3"]["model_id"] == "google/timesfm-3.0-pytorch"
    assert value["timesfm25"]["fixed_epochs"] == {"da": 2, "rt_at_da": 2, "rt": 3}
    assert value["pooled_xgboost"]["uses_node_or_market_id_as_feature"] is False


def test_dlinear_shapes_and_kernel_contract() -> None:
    model = DLinearForecaster(168, 24, 25)
    result = model(torch.randn(7, 168))
    assert result.shape == (7, 24)
    with pytest.raises(ValueError, match="奇数"):
        DLinearForecaster(168, 24, 24)


def test_timesfm3_locked_protocol_accepts_all_market_config() -> None:
    cfg = PilotConfig.from_yaml(ROOT / "configs/decision_aware/allmarket_final_da_mae_v1.yaml")
    validate_protocol(cfg, "da", "ERCOT", locked_2025=True)
    with pytest.raises(ValueError, match="留出市场"):
        validate_protocol(cfg, "da", "ERCOT", locked_2025=False)


def test_baseline_training_runner_cannot_open_2025() -> None:
    text = (ROOT / "scripts/decision_aware/run_cloud_allmarket_final_baselines_v1.sh").read_text(encoding="utf-8")
    assert "LOCKED_2025_OPENED.json" in text
    assert "test ! -e" in text
    assert "locked_2025 --output" not in text
    assert "--mode train" in text
    assert "--phase final" in text
    assert "audit_allmarket_final_baselines.py --require-complete" in text


def test_local_final_xgboost_runner_matches_frozen_cloud_contract() -> None:
    text = (ROOT / "scripts/decision_aware/run_local_allmarket_final_xgboost_v1.ps1").read_text(
        encoding="utf-8"
    )
    for market in ("CAISO", "ERCOT", "NYISO", "PJM"):
        assert f"'{market}'" in text
    for task in ("'da'", "'rt_at_da'", "'rt'"):
        assert task in text
    assert "--phase final" in text
    assert "xgboost_absolute" in text
    assert "$model = 'xgboost_absolute'" in text
    assert "LOCKED_2025_OPENED.json" in text
    assert "--force" not in text
    assert "locked_2025" not in text


def test_per_node_xgboost_uses_cross_market_stable_absolute_objective() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/decision_aware/allmarket_final_baselines_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["per_node_xgboost"]["objectives"] == {
        "da": "reg:absoluteerror",
        "rt_at_da": "reg:absoluteerror",
        "rt": "reg:absoluteerror",
    }
    assert config["pooled_xgboost"]["objectives"]["rt_at_da"] == "reg:pseudohubererror"
    assert config["pooled_xgboost"]["compute_device"] == "cpu"


def test_cloud_pooled_xgboost_uses_frozen_cpu_backend() -> None:
    text = (ROOT / "scripts/decision_aware/run_cloud_allmarket_final_baselines_v1.sh").read_text(
        encoding="utf-8"
    )
    assert '--checkpoint "$checkpoint" --seed 0 --device cpu' in text


def test_cloud_input_packager_excludes_obsolete_pseudohuber_cache() -> None:
    text = (ROOT / "scripts/decision_aware/package_allmarket_final_cloud_inputs.py").read_text(
        encoding="utf-8"
    )
    assert "xgboost_absolute_*_seed0.joblib" in text
    assert "xgboost_absolute_*_seed0.npz" in text
    assert "*_final/*.joblib" not in text


def test_final_baseline_auditor_uses_windows_safe_node_keys() -> None:
    import importlib.util

    path = ROOT / "scripts/decision_aware/audit_allmarket_final_baselines.py"
    spec = importlib.util.spec_from_file_location("audit_allmarket_final_baselines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.portable_node_key("N.Y.C.") == "node_87a9a8b1a98fc6a1"
    assert module.portable_node_key("ZONE:PJM-RTO") == "ZONE%3APJM-RTO"


def test_cloud_input_packager_requires_models_and_prediction_caches() -> None:
    text = (ROOT / "scripts/decision_aware/package_allmarket_final_cloud_inputs.py").read_text(
        encoding="utf-8"
    )
    assert "LOCKED_2025_OPENED.json" in text
    assert "len(xgb_checkpoints) != 192" in text
    assert "len(xgb_predictions) != 192" in text
    assert "len(xgb_summaries) != 12" in text
    assert "TASK_CHECKPOINTS" in text
    assert "file_sha256" in text


def test_unified_locked_runner_checks_everything_before_marker() -> None:
    text = (ROOT / "scripts/decision_aware/run_cloud_allmarket_locked_2025_v2.sh").read_text(encoding="utf-8")
    preflight = text.index("preflight()")
    marker = text.index("mark_opened_once()")
    assert preflight < marker
    assert "freeze_allmarket_2025_protocol.py --verify" in text
    assert "freeze_allmarket_2025_baseline_protocol.py --verify" in text
    assert "audit_allmarket_final_baselines.py --require-complete" in text
    for label in ("timesfm3", "timesfm25", "pooled_xgboost", "per_node_xgboost", "dlinear", "seasonal", "oracle"):
        assert label in text
    for seed in (0, 1, 2):
        assert f"da_seed{seed}__rt_at_da_seed{seed}__rt_seed{seed}" in text
    assert "--no-selection-allowed" in text


def test_locked_per_node_xgboost_is_evaluate_only() -> None:
    text = (ROOT / "scripts/decision_aware/run_cloud_allmarket_locked_2025_v2.sh").read_text(encoding="utf-8")
    assert "--phase locked_2025" in text
    assert "--evaluate-only" in text
    runner = (ROOT / "scripts/decision_aware/run_global_baselines.py").read_text(encoding="utf-8")
    assert "锁定2025禁止重新训练XGBoost" in runner
    assert "final_checkpoint" in runner


def test_timesfm3_locked_rt_archive_includes_execution_timestamps() -> None:
    text = (ROOT / "scripts/decision_aware/run_universal_lomo_timesfm3.py").read_text(encoding="utf-8")
    assert 'arrays["executed_timestamps"] = common_timestamps' in text
    assert 'arrays["timezone"]' in text
