from copy import deepcopy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig
from decision_aware.contracts import (
    AnnualStage,
    FORMAL_EXPERIMENT_CONTRACT,
    formal_checkpoint_metadata,
)


def _formal_config(**overrides):
    values = {
        "use_dual_settlement": True,
        "use_deviation_penalty": True,
        "bess_kappa": 5.7,
        "bess_power_mw": 1.0,
        "bess_energy_mwh": 4.0,
        "bess_eta": 0.95,
        "bess_init_soc_frac": 0.5,
        "bess_soc_min": 0.4,
        "bess_soc_max": 3.6,
        "bess_e_cyc": 4.0,
    }
    values.update(overrides)
    return PilotConfig(**values)


def test_formal_contract_hash_is_stable_and_has_frozen_annual_protocol():
    contract = FORMAL_EXPERIMENT_CONTRACT

    assert contract.contract_hash == contract.contract_hash
    assert len(contract.contract_hash) == 64
    assert contract.annual_split.inner_train.start == "2020-01-01"
    assert contract.annual_split.inner_validation.start == "2023-01-01"
    assert contract.annual_split.outer_validation.start == "2024-01-01"
    assert contract.annual_split.locked_retrospective_test.start == "2025-01-01"
    assert contract.annual_split.report_only.end == "2026-06-01"
    assert contract.annual_split.locked_test_can_select_model is False
    assert contract.annual_split.report_only_can_select_model is False
    assert contract.annual_split.range_for(AnnualStage.OUTER_VALIDATION).end == (
        "2024-12-31"
    )
    assert contract.annual_split.range_for("final_train").start == "2020-01-01"


def test_formal_contract_accepts_exact_settlement_config():
    metadata = formal_checkpoint_metadata(_formal_config())

    assert metadata["contract_hash"] == FORMAL_EXPERIMENT_CONTRACT.contract_hash
    assert metadata["experiment_contract"]["settlement"][
        "deviation_penalty_enabled"
    ] is True


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"use_deviation_penalty": False}, "use_deviation_penalty"),
        ({"use_dual_settlement": False}, "use_dual_settlement"),
        ({"bess_kappa": 0.0}, "bess_kappa"),
        ({"bess_eta": 0.90}, "bess_eta"),
        ({"bess_init_soc_frac": 0.25}, "initial_soc_mwh"),
        ({"bess_soc_max": 4.0}, "bess_soc_max"),
    ],
)
def test_formal_contract_rejects_silent_settlement_drift(override, field):
    with pytest.raises(ValueError, match=field):
        FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(
            _formal_config(**override)
        )


def test_checkpoint_contract_round_trip_and_tamper_detection():
    checkpoint = formal_checkpoint_metadata(_formal_config())
    FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(checkpoint)

    tampered = deepcopy(checkpoint)
    tampered["experiment_contract"]["settlement"][
        "degradation_cost_usd_per_mwh"
    ] = 9.5
    with pytest.raises(ValueError, match="合同内容"):
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(tampered)


def test_legacy_checkpoint_is_not_silently_promoted_to_formal():
    with pytest.raises(ValueError, match="缺少"):
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(
            {"model_state": {}, "config": _formal_config().to_dict()}
        )
