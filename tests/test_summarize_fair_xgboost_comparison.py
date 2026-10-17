import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.summarize_fair_xgboost_comparison import (
    daily_risk,
    macro_mae,
    paired_daily_profit,
)


def test_macro_mae_is_node_macro_and_accepts_generic_target(tmp_path):
    path = tmp_path / "result.npz"
    prediction = np.asarray([[[0.0, 2.0]], [[10.0, 10.0]]])
    target = np.asarray([[[0.0, 0.0]], [[0.0, 0.0]]])
    np.savez_compressed(path, prediction=prediction, target=target)
    # 节点0 MAE=1，节点1 MAE=10，节点宏平均=5.5。
    assert macro_mae(path) == pytest.approx(5.5)


def test_macro_mae_rejects_shape_mismatch(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, prediction=np.zeros((1, 2, 3)), target=np.zeros((1, 2, 4)))
    with pytest.raises(ValueError, match="形状"):
        macro_mae(path)


def test_paired_daily_profit_uses_matched_days_and_reports_mean():
    candidate = {"macro_daily_net_revenue": {"d1": 3.0, "d2": 7.0}}
    reference = {"macro_daily_net_revenue": {"d1": 1.0, "d2": 5.0}}
    result = paired_daily_profit(candidate, reference)
    assert result["mean_daily_difference"] == pytest.approx(2.0)
    assert len(result["seven_day_block_bootstrap_95ci"]) == 2


def test_paired_daily_profit_rejects_unmatched_days():
    candidate = {"macro_daily_net_revenue": {"d1": 3.0}}
    reference = {"macro_daily_net_revenue": {"d2": 1.0}}
    with pytest.raises(ValueError, match="逐日键"):
        paired_daily_profit(candidate, reference)


def test_daily_risk_uses_chronological_drawdown_and_top_days():
    item = {
        "macro_daily_net_revenue": {
            f"d{index:02d}": value
            for index, value in enumerate([5.0, -2.0, -4.0, 3.0, 1.0, 2.0, 4.0,
                                           6.0, 8.0, 10.0, 12.0, 14.0])
        },
        "worst_node_mean_daily_revenue": -1.5,
        "node_mean_daily_net_revenue": {"A": -1.5, "B": 5.5},
    }
    result = daily_risk(item)
    assert result["maximum_drawdown_usd_per_node"] == pytest.approx(6.0)
    assert result["positive_day_rate"] == pytest.approx(10 / 12)
    assert result["worst_node_mean_daily_revenue"] == pytest.approx(-1.5)
    assert len(result["node_bootstrap_mean_95ci"]) == 2
