import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "decision_aware" / "run_universal_lomo_xgboost.py"
SPEC = importlib.util.spec_from_file_location("run_universal_lomo_xgboost", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rt_hour_extraction_removes_identity_and_selects_hour():
    sample = {
        "price_rt_ctx": torch.arange(24 * 3).reshape(24, 3),
        "price_rt_tgt": torch.arange(24 * 4).reshape(24, 4),
        "price_rt_mean": torch.arange(24.0),
        "price_rt_std": torch.ones(24),
        "node_id": torch.tensor(8),
        "market_id": torch.tensor(2),
    }
    hourly = MODULE._hour_from_rt_day(sample, 7)
    assert "node_id" not in hourly
    assert "market_id" not in hourly
    assert hourly["price_rt_ctx"].tolist() == [21, 22, 23]
    assert hourly["price_rt_tgt"].tolist() == [28, 29, 30, 31]
    assert hourly["price_rt_mean"].item() == pytest.approx(7.0)


def test_rt_hour_extraction_rejects_invalid_hour():
    with pytest.raises(ValueError, match="RT小时"):
        MODULE._hour_from_rt_day({}, 24)


def test_objectives_follow_task_specific_loss_family():
    assert MODULE.OBJECTIVES == {
        "da": "reg:absoluteerror",
        "rt_at_da": "reg:pseudohubererror",
        "rt": "reg:absoluteerror",
    }
