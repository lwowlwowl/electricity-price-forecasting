from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.decision_aware.prepare_allmarket_locked_2025 import (  # noqa: E402
    _validate_seed_group,
)
from scripts.decision_aware.summarize_allmarket_locked_2025 import (  # noqa: E402
    _validate_archive_alignment,
)
from scripts.decision_aware.audit_allmarket_locked_2025_profit_claim import (  # noqa: E402
    classify_profit_difference,
)


CONFIGS = {
    "da": ROOT / "configs/decision_aware/allmarket_final_da_mae_v1.yaml",
    "rt_at_da": ROOT / "configs/decision_aware/allmarket_final_rt_at_da_pseudohuber_v1.yaml",
    "rt": ROOT / "configs/decision_aware/allmarket_final_rt_mae_v1.yaml",
}


def test_final_configs_freeze_current_no_id_architecture() -> None:
    for task, path in CONFIGS.items():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["global_training_markets"] == ["CAISO", "ERCOT", "NYISO", "PJM"]
        assert raw["global_use_entity_embeddings"] is False
        assert raw["global_num_nodes"] == raw["global_num_markets"] == 0
        assert raw["global_normalization_mode"] == "context_robust"
        assert raw["global_use_critical_event_weighting"] is False
        assert (raw["da_fusion_mode"], raw["rt_at_da_fusion_mode"], raw["rt_fusion_mode"]) == (
            "F1", "F0", "F1",
        )
        assert raw["global_training_loss"] == (
            "pseudo_huber" if task == "rt_at_da" else "mae"
        )


def test_training_runner_never_opens_locked_2025() -> None:
    text = (ROOT / "scripts/decision_aware/run_cloud_allmarket_final_train_v1.sh").read_text(
        encoding="utf-8"
    )
    assert "--phase final" in text
    assert "--frozen-training-plan-from" in text
    assert "evaluate_universal_holdout.py" not in text
    assert "LOCKED_2025_OPENED" not in text
    assert "locked_2025_read': False" in text


def test_locked_runner_has_explicit_irreversible_gate() -> None:
    text = (ROOT / "scripts/decision_aware/run_cloud_allmarket_locked_2025_v1.sh").read_text(
        encoding="utf-8"
    )
    assert "RESUME_LOCKED_RUN" in text
    assert "LOCKED_2025_OPENED.json" in text
    assert "--evaluation-stage locked_2025" in text
    assert "--no-selection-allowed" in text
    assert "freeze_allmarket_2025_protocol.py --verify" in text


def test_final_trainer_uses_per_market_predicate_cutoffs() -> None:
    text = (ROOT / "scripts/decision_aware/train_global_forecaster.py").read_text(
        encoding="utf-8"
    )
    assert '"final": (AnnualStage.FINAL_TRAIN, AnnualStage.OUTER_VALIDATION)' in text
    for cutoff in (
        '"CAISO": "2025-01-01T08:00:00+00:00"',
        '"ERCOT": "2025-01-01T06:00:00+00:00"',
        '"NYISO": "2025-01-01T05:00:00+00:00"',
        '"PJM": "2025-01-01T05:00:00+00:00"',
    ):
        assert cutoff in text
    assert "final阶段必须使用预先冻结的同seed inner训练计划" in text


def test_seed_group_rejects_selection_enabled_locked_result() -> None:
    base = {
        "nodes": np.asarray(["N1"]),
        "dates": np.asarray(["2025-01-01"]),
        "prediction": np.zeros((1, 1, 24), dtype=np.float32),
        "target": np.zeros((1, 1, 24), dtype=np.float32),
        "metadata": np.asarray(json.dumps({
            "protocol": "all_market_locked_2025",
            "selection_allowed": False,
            "checkpoint_phase": "final",
            "evaluation_market": "ERCOT",
        })),
    }
    valid = [{key: value.copy() for key, value in base.items()} for _ in range(3)]
    _validate_seed_group(valid, "ERCOT", "da")
    invalid = [{key: value.copy() for key, value in item.items()} for item in valid]
    metadata = json.loads(str(invalid[1]["metadata"].item()))
    metadata["selection_allowed"] = True
    invalid[1]["metadata"] = np.asarray(json.dumps(metadata))
    with pytest.raises(ValueError, match="错误允许2025选模"):
        _validate_seed_group(invalid, "ERCOT", "da")


def test_cross_system_alignment_rejects_dates_targets_and_rt_timestamps() -> None:
    reference = {
        "nodes": np.asarray(["N1"]),
        "dates": np.asarray(["2025-01-01"]),
        "prediction": np.zeros((1, 1, 4), dtype=np.float32),
        "target": np.ones((1, 1, 4), dtype=np.float32),
        "executed_timestamps": np.asarray([["2025-01-01T01:00:00Z"] * 4]),
        "timezone": np.asarray("America/Chicago"),
    }
    _validate_archive_alignment(
        reference, {key: value.copy() for key, value in reference.items()},
        system="x", market="ERCOT", task="rt",
    )
    for key, replacement, message in (
        ("dates", np.asarray(["2025-01-02"]), "日期不一致"),
        ("target", np.zeros((1, 1, 4), dtype=np.float32), "真值不一致"),
        ("executed_timestamps", np.asarray([["bad"] * 4]), "executed_timestamps不一致"),
    ):
        invalid = {name: value.copy() for name, value in reference.items()}
        invalid[key] = replacement
        with pytest.raises(ValueError, match=message):
            _validate_archive_alignment(
                reference, invalid, system="x", market="ERCOT", task="rt"
            )


def test_locked_v2_runs_final_artifact_audit_before_packaging() -> None:
    text = (ROOT / "scripts/decision_aware/run_cloud_allmarket_locked_2025_v2.sh").read_text(
        encoding="utf-8"
    )
    assert "audit_allmarket_locked_2025_outputs.py" in text
    assert text.index("audit_allmarket_locked_2025_outputs.py") < text.index("tar -czf")


def test_profit_analysis_preregistration_makes_revenue_primary() -> None:
    raw = yaml.safe_load(
        (ROOT / "configs/decision_aware/allmarket_locked_2025_profit_analysis_v1.yaml")
        .read_text(encoding="utf-8")
    )
    endpoint = raw["primary_economic_endpoint"]
    assert endpoint["market"] == "ERCOT"
    assert endpoint["system"] == "transformer_fixed_seed012_ensemble"
    assert endpoint["comparator"] == "pooled_allmarket_xgboost"
    assert endpoint["metric"] == "macro_mean_daily_net_revenue_usd_per_node_day"
    assert endpoint["stable_superiority_rule"] == "mean_difference_gt_0_and_ci_lower_gt_0"
    assert raw["claim_policy"]["mae_cannot_override_primary_economic_endpoint"] is True
    assert raw["selection_uses_2025"] is False


@pytest.mark.parametrize(
    ("mean_difference", "ci", "expected"),
    [
        (5.0, [1.0, 9.0], "stable_superiority"),
        (5.0, [-1.0, 9.0], "positive_point_estimate_only"),
        (0.0, [-2.0, 2.0], "no_superiority"),
        (-5.0, [-9.0, -1.0], "no_superiority"),
    ],
)
def test_profit_claim_classification_is_not_mae_based(
    mean_difference: float, ci: list[float], expected: str
) -> None:
    assert classify_profit_difference(mean_difference, ci) == expected


def test_pdf_generator_foregrounds_profit_claim_and_risk() -> None:
    text = (ROOT / "scripts/documentation/generate_multinode_technical_report.py").read_text(
        encoding="utf-8"
    )
    assert "预注册盈利主终点" in text
    assert "locked_2025_profit_evidence" in text
    assert "配对7日块95%区间" in text
    assert "去最高10日后日均收益" in text
    assert "MAE仅用于预测与迁移诊断，不能覆盖盈利结论" in text
