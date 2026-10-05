from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig
from scripts.decision_aware.select_penalty_active_triplets import (
    _validate_active_contract,
    _write_new_json,
    evaluate_triplet_grid,
)


def _cfg() -> PilotConfig:
    cfg = PilotConfig()
    cfg.use_deviation_penalty = True
    cfg.joint_coordination_mode = "plan_track_topk"
    cfg.joint_da_signal_mode = "spread"
    cfg.topk_k_charge = 1
    cfg.topk_k_discharge = 1
    cfg.rt_topk_k_charge = 1
    cfg.rt_topk_k_discharge = 1
    return cfg


def _candidate(value: float, shape: tuple[int, ...]) -> dict:
    return {
        "path": f"candidate-{value}.pt",
        "sha256": str(value),
        "prediction": np.full(shape, value, dtype=np.float32),
    }


def test_active_triplet_grid_evaluates_full_cartesian_product():
    cfg = _cfg()
    candidates = {
        "da": {
            "da0": _candidate(0.0, (1, 24)),
            "da1": _candidate(1.0, (1, 24)),
        },
        "rt_at_da": {
            "rtda0": _candidate(0.0, (1, 24)),
            "rtda1": _candidate(1.0, (1, 24)),
            "rtda2": _candidate(2.0, (1, 24)),
        },
        "rt": {
            "rt0": _candidate(0.0, (24, cfg.horizon_rt)),
            "rt1": _candidate(1.0, (24, cfg.horizon_rt)),
        },
    }
    contract = {
        "dates": ["2025-07-01"],
        "true_da": np.zeros((1, 24), dtype=np.float32),
        "true_rt": np.zeros((1, 24), dtype=np.float32),
        "starts_new_segment": np.asarray([True]),
    }

    def fake_settle(p_da, p_rt_at_da, p_rt, *args, **kwargs):
        score = p_da[0, 0] + 2 * p_rt_at_da[0, 0] + 3 * p_rt[0, 0, 0]
        one = score.reshape(1)
        return {
            "daily_revenue": one,
            "da_leg": one,
            "rt_deviation_leg": one * 0,
            "degradation_cost": one * 0,
            "deviation_penalty": one * 0,
            "plan_soc_end_mwh": torch.tensor([2.0]),
            "actual_soc_end_mwh": torch.tensor([2.0]),
        }

    rows = evaluate_triplet_grid(
        candidates, contract, cfg, settle_fn=fake_settle, progress_every=0
    )

    assert len(rows) == 2 * 3 * 2
    assert (rows[0]["da"], rows[0]["rt_at_da"], rows[0]["rt"]) == (
        "da1", "rtda2", "rt1"
    )
    assert rows[0]["mean_daily_revenue"] == pytest.approx(8.0)


def test_active_contract_rejects_zero_rt_topk():
    cfg = _cfg()
    cfg.rt_topk_k_charge = 0
    with pytest.raises(ValueError, match="K都大于0"):
        _validate_active_contract(cfg)


def test_new_json_writer_refuses_to_overwrite(tmp_path: Path):
    output = tmp_path / "result.json"
    _write_new_json(output, {"run": 1})
    with pytest.raises(FileExistsError):
        _write_new_json(output, {"run": 2})
