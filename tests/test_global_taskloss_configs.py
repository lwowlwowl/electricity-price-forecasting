from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402


def test_taskloss_configs_are_unanchored_from_scratch_and_task_specific():
    expected = {
        "global_ercot_taskloss_da_mae_v1.yaml": ("mae", "F1", "da"),
        "global_ercot_taskloss_rt_at_da_pseudohuber_v1.yaml": (
            "pseudo_huber", "F0", "rt_at_da"
        ),
        "global_ercot_taskloss_rt_mae_v1.yaml": ("mae", "F1", "rt"),
    }
    checkpoint_dirs = set()
    for name, (loss, fusion, task) in expected.items():
        cfg = PilotConfig.from_yaml(ROOT / "configs" / "decision_aware" / name)
        assert cfg.global_training_loss == loss
        assert cfg.global_residual_anchor == "none"
        assert cfg.global_trainable_scope == "all"
        assert cfg.global_use_entity_embeddings is True
        assert cfg.global_num_nodes == 15
        if task == "da":
            assert cfg.da_fusion_mode == fusion
        elif task == "rt_at_da":
            assert cfg.rt_at_da_fusion_mode == fusion
        else:
            assert cfg.rt_fusion_mode == fusion
        checkpoint_dirs.add(cfg.checkpoint_dir)
    assert len(checkpoint_dirs) == 3


def test_entitypre_configs_only_change_conditioning_stage_and_output_dirs():
    pairs = (
        ("global_ercot_taskloss_da_mae_v1.yaml", "global_ercot_entitypre_da_mae_v1.yaml"),
        (
            "global_ercot_taskloss_rt_at_da_pseudohuber_v1.yaml",
            "global_ercot_entitypre_rt_at_da_pseudohuber_v1.yaml",
        ),
        ("global_ercot_taskloss_rt_mae_v1.yaml", "global_ercot_entitypre_rt_mae_v1.yaml"),
    )
    for baseline_name, candidate_name in pairs:
        baseline = PilotConfig.from_yaml(
            ROOT / "configs" / "decision_aware" / baseline_name
        ).to_dict()
        candidate = PilotConfig.from_yaml(
            ROOT / "configs" / "decision_aware" / candidate_name
        ).to_dict()
        assert baseline["global_entity_conditioning_mode"] == "post_fusion"
        assert candidate["global_entity_conditioning_mode"] == "pre_encoder"
        differing = {
            key for key in baseline if baseline[key] != candidate[key]
        }
        assert differing == {"global_entity_conditioning_mode", "checkpoint_dir"}


def test_spread_factorization_changes_target_parameterization_not_training_budget():
    baseline = PilotConfig.from_yaml(
        ROOT
        / "configs"
        / "decision_aware"
        / "global_ercot_taskloss_rt_at_da_pseudohuber_v1.yaml"
    ).to_dict()
    candidate = PilotConfig.from_yaml(
        ROOT
        / "configs"
        / "decision_aware"
        / "global_ercot_spreadfactor_rt_at_da_v1.yaml"
    ).to_dict()
    differing = {key for key in baseline if baseline[key] != candidate[key]}
    assert differing == {
        "rt_at_da_parameterization",
        "rt_at_da_upstream_da_checkpoint",
        "checkpoint_dir",
    }
    assert candidate["rt_at_da_parameterization"] == "da_minus_spread"
    assert candidate["global_training_loss"] == "pseudo_huber"
