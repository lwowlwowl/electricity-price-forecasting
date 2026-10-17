import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.summarize_local_lear_baseline import (
    lear_result,
    render,
)


def test_lear_result_is_incremental_and_requires_complete(tmp_path):
    assert lear_result(tmp_path, "CAISO", "da")["status"] == "pending"
    path = tmp_path / "data/results/global_baselines_v1/caiso/da_inner"
    path.mkdir(parents=True)
    for node in range(3):
        (path / f"lear_node{node}_seed0.npz").write_bytes(b"cache")
    (path / "summary.json").write_text(json.dumps({
        "models": {"lear": {
            "complete": True,
            "macro_mae": 1.25,
            "elapsed_seconds": 60.0,
            "node_count": 3,
            "protocol": "cutoff_safe_lear_daily_recalibration",
        }}
    }), encoding="utf-8")
    result = lear_result(tmp_path, "CAISO", "da")
    assert result["status"] == "complete"
    assert result["mae"] == pytest.approx(1.25)


def test_lear_result_keeps_legacy_ercot_directory(tmp_path):
    path = tmp_path / "data/results/global_baselines_v1/da_inner"
    path.mkdir(parents=True)
    for node in range(15):
        (path / f"lear_node{node}_seed0.npz").write_bytes(b"cache")
    (path / "summary.json").write_text(json.dumps({
        "models": {"lear": {
            "complete": True, "macro_mae": 2.5, "elapsed_seconds": 1.0,
            "node_count": 15, "protocol": "cutoff_safe_lear_daily_recalibration",
        }}
    }), encoding="utf-8")
    assert lear_result(tmp_path, "ERCOT", "da")["mae"] == pytest.approx(2.5)


def test_lear_result_rejects_subset_summary_marked_complete(tmp_path):
    path = tmp_path / "data/results/global_baselines_v1/pjm/da_inner"
    path.mkdir(parents=True)
    for node in range(23):
        (path / f"lear_node{node}_seed0.npz").write_bytes(b"cache")
    (path / "summary.json").write_text(json.dumps({
        "models": {"lear": {
            "complete": True, "macro_mae": 1.0, "elapsed_seconds": 60.0,
            "node_count": 23, "protocol": "cutoff_safe_lear_daily_recalibration",
        }}
    }), encoding="utf-8")
    result = lear_result(tmp_path, "PJM", "da")
    assert result["status"] == "pending"
    assert result["cached_nodes"] == 23
    assert result["expected_nodes"] == 35


def test_render_marks_partial_without_inventing_rolling_rt():
    row = {
        "market": "CAISO", "task": "da",
        "lear": {"status": "complete", "mae": 1.0, "elapsed_seconds": 120.0},
        "transformer_mae": 2.0,
        "pooled_lomo_xgboost_mae": 3.0,
        "winner_among_available": "lear",
    }
    text = render({"rows": [row], "complete_rows": 1})
    assert "**1.00**" in text
    assert "1/8" in text
    assert "滚动RT不在LEAR定义内" in text
    assert "目标市场监督强基线" in text
