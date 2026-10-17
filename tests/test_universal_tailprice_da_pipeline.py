from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.decision_aware.summarize_universal_ercot_tailprice_da import (  # noqa: E402
    acceptance_decision,
)


BASELINE = ROOT / "configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml"
CANDIDATE = ROOT / "configs/decision_aware/universal_lomo_holdout_ercot_da_tailprice_v1.yaml"
RUNNER = ROOT / "scripts/decision_aware/run_cloud_universal_ercot_tailprice_da_v1.sh"


def test_tailprice_da_config_is_a_strict_single_factor_ablation():
    baseline = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    event_keys = {key for key in candidate if key.startswith("global_event_")}
    allowed = event_keys | {"global_use_critical_event_weighting", "checkpoint_dir"}
    differences = {
        key
        for key in set(baseline) | set(candidate)
        if baseline.get(key) != candidate.get(key)
    }
    assert differences <= allowed
    assert candidate["global_training_markets"] == ["CAISO", "NYISO", "PJM"]
    assert candidate["market"] == "ERCOT"
    assert candidate["seed"] == 0
    assert candidate["global_use_entity_embeddings"] is False
    assert candidate["global_training_loss"] == "mae"
    assert candidate["da_fusion_mode"] == "F1"
    assert candidate["global_use_critical_event_weighting"] is True
    assert candidate["global_event_price_lower_quantile"] == 0.01
    assert candidate["global_event_price_upper_quantile"] == 0.99
    assert candidate["global_event_price_weight"] == 1.0
    assert candidate["global_event_spread_weight"] == 0.0
    assert candidate["global_event_sign_reversal_weight"] == 0.0
    assert candidate["global_event_rt_first_step_weight"] == 0.0
    assert candidate["global_event_max_weight"] == 2.0


def test_cloud_runner_trains_only_da_and_keeps_final_years_locked():
    script = RUNNER.read_text(encoding="utf-8")
    assert "--task da --phase inner --seed 0" in script
    assert "--task rt_at_da" not in script
    assert "--task rt " not in script
    assert "--phase outer" not in script
    assert "test_2024_used_for_selection': False" in script
    assert "test_2025_read': False" in script
    assert "summarize_universal_ercot_tailprice_da.py" in script
    assert "da_tail_price_da__rt_at_da_current__rt_current" in script
    assert "universal_lomo_ercot_tailprice_da_completed_20261016.tar.gz" in script


def _price_report(tail_mae, ordinary_mae, negative_recall):
    return {
        "highest_1pct": {"mae": tail_mae},
        "ordinary": {"mae": ordinary_mae},
        "negative_price": {"negative_recall": negative_recall},
    }


def test_acceptance_requires_tail_ordinary_negative_and_profit_checks():
    current = _price_report(100.0, 10.0, 0.25)
    passing = acceptance_decision(
        current,
        _price_report(90.0, 10.1, 0.30),
        current_profit=100.0,
        candidate_profit=101.0,
    )
    assert passing["eligible_to_expand"] is True

    failing = acceptance_decision(
        current,
        _price_report(90.0, 10.3, 0.30),
        current_profit=100.0,
        candidate_profit=101.0,
    )
    assert failing["eligible_to_expand"] is False
    assert failing["checks"]["ordinary_da_mae_within_2pct"] is False
