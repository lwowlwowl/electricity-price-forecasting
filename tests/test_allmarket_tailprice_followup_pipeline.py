from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/decision_aware/allmarket_final_da_mae_v1.yaml"
CANDIDATE = ROOT / "configs/decision_aware/allmarket_final_da_tailprice_v2.yaml"
TRAIN = ROOT / "scripts/decision_aware/run_cloud_allmarket_final_da_tailprice_v2.sh"
EVALUATE = ROOT / "scripts/decision_aware/run_cloud_allmarket_tailprice_followup_2025_v2.sh"


def test_final_tailprice_config_is_single_factor_da_change() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    event_keys = {key for key in candidate if key.startswith("global_event_")}
    allowed = event_keys | {"global_use_critical_event_weighting", "checkpoint_dir"}
    differences = {
        key for key in set(base) | set(candidate)
        if base.get(key) != candidate.get(key)
    }
    assert differences <= allowed
    assert candidate["global_training_markets"] == ["CAISO", "ERCOT", "NYISO", "PJM"]
    assert candidate["global_use_critical_event_weighting"] is True
    assert candidate["global_event_price_lower_quantile"] == 0.01
    assert candidate["global_event_price_upper_quantile"] == 0.99
    assert candidate["global_event_price_weight"] == 1.0
    assert candidate["global_event_max_weight"] == 2.0
    assert candidate["global_event_spread_weight"] == 0.0
    assert candidate["global_event_sign_reversal_weight"] == 0.0
    assert candidate["global_event_rt_first_step_weight"] == 0.0


def test_training_runner_only_retrains_da_and_does_not_open_2025() -> None:
    text = TRAIN.read_text(encoding="utf-8")
    assert "--task da --phase final" in text
    assert "--task rt_at_da" not in text
    assert "--task rt " not in text
    assert "evaluate_universal_holdout.py" not in text
    assert "--frozen-training-plan-from" in text
    assert "candidate_2025_read_during_training': False" in text


def test_followup_runner_reuses_rt_legs_and_forbids_selection() -> None:
    text = EVALUATE.read_text(encoding="utf-8")
    assert "original_2025_results_were_known_before_followup': True" in text
    assert "--evaluation-stage locked_2025" in text
    assert "--no-selection-allowed" in text
    assert "for task in rt_at_da rt" in text
    assert "evaluate_timesfm" not in text
    assert "run_allmarket_final_xgboost" not in text
    assert "summarize_allmarket_tailprice_followup_2025.py" in text
