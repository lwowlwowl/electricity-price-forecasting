from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.preflight_timesfm25_lora import (
    DEFAULT_CONFIG,
    MODEL_ID,
    validate_protocol,
)


def _protocol():
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_timesfm25_lora_protocol_is_four_fold_three_task_and_identity_free():
    report = validate_protocol(_protocol())
    assert report["model_id"] == MODEL_ID
    assert report["folds"] == 4
    assert report["tasks"] == {"da": 24, "rt_at_da": 24, "rt": 4}
    assert report["context_hours"] == 168
    assert report["input_features"] == ["target_price_history"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda raw: raw["data"].update(uses_holdout_labels_for_training=True),
        lambda raw: raw["data"]["folds"][0]["sources"].append("CAISO"),
        lambda raw: raw["tasks"]["rt"].update(horizon=24),
        lambda raw: raw["data"].update(input_features=["future_true_load"]),
        lambda raw: raw["training"].update(decision_profit_backpropagation=True),
    ),
)
def test_timesfm25_lora_protocol_rejects_leakage_or_contract_drift(mutation):
    raw = deepcopy(_protocol())
    mutation(raw)
    with pytest.raises(ValueError):
        validate_protocol(raw)
