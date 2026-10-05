from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig
from scripts.decision_aware import evaluate_joint_checkpoint as module


def _full_dual(days: int = 2) -> dict:
    inventory = {
        "initial_soc_mwh": 2.0,
        "segment_initial_soc_mwh": [2.0],
        "segment_final_soc_mwh": [1.5],
        "segment_soc_change_mwh": [-0.5],
        "total_segment_soc_change_mwh": -0.5,
        "final_soc_mwh": 1.5,
        "final_minus_initial_soc_mwh": -0.5,
    }
    return {
        "mean_daily_revenue": 12.5,
        "positive_day_rate": 0.5,
        "lower_tail_10pct_mean": -2.0,
        "max_drawdown": 2.0,
        "weekly_block_bootstrap_95ci": [1.0, 20.0],
        "days": days,
        "segments": 1,
        "revenue_components": {
            "da_leg": 30.0,
            "rt_deviation_leg": 0.0,
            "degradation_cost": 5.0,
            "deviation_penalty": 0.0,
        },
        "plan_feasibility": {
            **inventory,
            "total_clipped_mwh": 0.0,
            "clipped_days": 0,
        },
        "actual_state": {**inventory, "action_digest": "abc"},
        "daily_net_revenue": {
            "2025-07-01": -2.0,
            "2025-07-02": 27.0,
        },
    }


def test_contract_mismatch_rejects_changed_penalty_policy():
    evaluation = PilotConfig(
        use_deviation_penalty=True,
        joint_coordination_mode="follow_da",
        topk_k_charge=3,
    )
    recorded = evaluation.to_dict()
    recorded["topk_k_charge"] = 4

    with pytest.raises(ValueError, match="topk_k_charge"):
        module._validate_evaluation_contract(evaluation, recorded)


def test_report_only_requires_explicit_confirmation():
    with pytest.raises(ValueError, match="confirm-report-only"):
        module._validate_split_request("report_only", False)
    module._validate_split_request("report_only", True)
    module._validate_split_request("validation", False)
    with pytest.raises(ValueError, match="不应传入"):
        module._validate_split_request("validation", True)


def test_latest_training_state_is_not_accepted_as_locked_model():
    with pytest.raises(ValueError, match="latest训练状态"):
        module._validate_locked_bundle({"kind": "joint_training_state"})


def test_bundle_flag_allows_only_source_kappa_mismatch():
    training = PilotConfig(
        bess_kappa=9.5,
        bess_eta=0.95,
        joint_allow_source_kappa_mismatch=True,
    )
    source = PilotConfig(bess_kappa=5.7, bess_eta=0.95)

    audit = module._check_source_physical_contract(training, source, "da")

    assert audit == {
        "source_bess_kappa": 5.7,
        "evaluation_bess_kappa": 9.5,
        "kappa_mismatch": True,
        "kappa_mismatch_allowed_by_bundle": True,
    }


def test_source_kappa_mismatch_is_rejected_when_bundle_flag_is_false():
    training = PilotConfig(
        bess_kappa=9.5,
        joint_allow_source_kappa_mismatch=False,
    )
    source = PilotConfig(bess_kappa=5.7)

    with pytest.raises(ValueError, match="bess_kappa"):
        module._check_source_physical_contract(training, source, "da")


def test_bundle_kappa_flag_never_allows_eta_mismatch():
    training = PilotConfig(
        bess_kappa=9.5,
        bess_eta=0.95,
        joint_allow_source_kappa_mismatch=True,
    )
    source = PilotConfig(bess_kappa=5.7, bess_eta=0.90)

    with pytest.raises(ValueError, match="bess_eta"):
        module._check_source_physical_contract(training, source, "da")


def test_json_writer_refuses_to_overwrite(tmp_path: Path):
    output = tmp_path / "evaluation.json"
    module._write_json_exclusive(output, {"first": True})
    with pytest.raises(FileExistsError):
        module._write_json_exclusive(output, {"second": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"first": True}


def test_terminal_soc_is_reported_in_mwh_without_monetization():
    report = module._terminal_soc_report(_full_dual())
    assert report["unit"] == "MWh"
    assert report["monetized"] is False
    assert report["included_in_revenue"] is False
    assert report["actual"]["final_soc_mwh"] == pytest.approx(1.5)
    assert report["da_plan"]["segment_final_soc_mwh"] == [1.5]


def test_main_defaults_to_validation_and_writes_auditable_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = PilotConfig()
    config_path = tmp_path / "locked.yaml"
    config_path.write_text(
        yaml.safe_dump(cfg.to_dict(), allow_unicode=True), encoding="utf-8"
    )
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"placeholder")
    output_path = tmp_path / "result.json"
    bundle = {
        "model_states": {"da": {}, "rt_at_da": {}, "rt": {}},
        "experiment_config": cfg.to_dict(),
        "source_checkpoints": {},
        "training_mode": "decision_aware",
        "best_epoch": 2,
        "best_validation_revenue": 12.5,
        "validation": {"full_dual": {"days": 2}},
    }
    source_checkpoints = {
        name: {"config": cfg.to_dict(), "norm_stats": {}}
        for name in ("da", "rt_at_da", "rt")
    }
    source_audit = {
        name: {"path": f"{name}.pt", "sha256": name * 4}
        for name in source_checkpoints
    }
    validation = SimpleNamespace(
        windows=[
            SimpleNamespace(delivery_date=date(2025, 7, 1)),
            SimpleNamespace(delivery_date=date(2025, 7, 2)),
        ],
        __len__=lambda self: len(self.windows),
    )
    # SimpleNamespace不从实例属性解析特殊方法，因此用一个极小本地类型。
    class Dataset:
        windows = validation.windows

        def __len__(self):
            return len(self.windows)

    validation_dataset = Dataset()
    report_dataset = Dataset()
    seen = {}

    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        module,
        "_load_bundle_source_checkpoints",
        lambda *args: ({}, source_checkpoints, source_audit),
    )
    monkeypatch.setattr(module, "_check_norm_stats", lambda *args: None)
    monkeypatch.setattr(module, "_check_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "build_joint_datasets",
        lambda *args, **kwargs: (object(), validation_dataset, report_dataset, object()),
    )
    monkeypatch.setattr(
        module, "_build_transformers",
        lambda *args, **kwargs: {"da": object(), "rt_at_da": object(), "rt": object()},
    )
    monkeypatch.setattr(module, "_assert_independent", lambda *args: None)
    monkeypatch.setattr(module, "_resolve_device", lambda value: module.torch.device("cpu"))
    monkeypatch.setattr(module, "_sha256", lambda path: "f" * 64)

    def fake_evaluate(models, dataset, config, device, amp):
        seen["dataset"] = dataset
        seen["amp"] = amp
        return {
            "full_dual": _full_dual(),
            "forecast": {
                "da": {"mae": 1.0, "rmse": 2.0},
                "rt_at_da": {"mae": 3.0, "rmse": 4.0},
                "rt_all_horizons": {"mae": 5.0, "rmse": 6.0},
            },
        }

    monkeypatch.setattr(module, "evaluate", fake_evaluate)
    payload = module.main([
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--output", str(output_path),
    ])

    assert seen == {"dataset": validation_dataset, "amp": False}
    assert payload["split"] == "validation"
    assert payload["report_only_explicitly_confirmed"] is False
    assert payload["checkpoint"]["sha256"] == "f" * 64
    assert set(payload["source_checkpoints"]) == {"da", "rt_at_da", "rt"}
    assert payload["source_checkpoints"]["da"]["source_bess_kappa"] == 5.7
    assert payload["source_checkpoints"]["da"]["evaluation_bess_kappa"] == 5.7
    assert payload["source_checkpoints"]["da"][
        "kappa_mismatch_allowed_by_bundle"
    ] is False
    assert payload["metrics"]["mean_daily_revenue"] == pytest.approx(12.5)
    assert payload["metrics"]["forecast"]["rt_all_horizons"]["rmse"] == 6.0
    assert payload["metrics"]["terminal_soc"]["actual"]["final_soc_mwh"] == 1.5
    assert json.loads(output_path.read_text(encoding="utf-8"))["days"] == 2
