from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


PAIRS = (
    (
        "global_ercot_taskloss_da_mae_v1.yaml",
        "global_ercot_eventweight_da_mae_v1.yaml",
    ),
    (
        "global_ercot_taskloss_rt_at_da_pseudohuber_v1.yaml",
        "global_ercot_eventweight_rt_at_da_pseudohuber_v1.yaml",
    ),
    (
        "global_ercot_taskloss_rt_mae_v1.yaml",
        "global_ercot_eventweight_rt_mae_v1.yaml",
    ),
)


@pytest.mark.parametrize("baseline_name,weighted_name", PAIRS)
def test_eventweight_is_strict_loss_ablation(baseline_name, weighted_name):
    config_dir = ROOT / "configs/decision_aware"
    baseline = yaml.safe_load((config_dir / baseline_name).read_text(encoding="utf-8"))
    weighted = yaml.safe_load((config_dir / weighted_name).read_text(encoding="utf-8"))
    weighting_keys = {
        key for key in weighted if key.startswith("global_event_")
    } | {"global_use_critical_event_weighting"}
    allowed = weighting_keys | {"checkpoint_dir", "global_entity_conditioning_mode"}
    differences = {
        key
        for key in set(baseline) | set(weighted)
        if baseline.get(key) != weighted.get(key)
    }
    assert differences <= allowed
    assert weighted["global_use_critical_event_weighting"] is True
    assert weighted["global_event_max_weight"] <= 4.0
    assert weighted["global_residual_anchor"] == "none"
    assert weighted["global_entity_conditioning_mode"] == "post_fusion"
