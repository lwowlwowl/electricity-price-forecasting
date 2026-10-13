import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.backtest_dual import locked_dual_backtest_by_node
from decision_aware.config import PilotConfig


def _node_input(offset: float):
    timestamps = pd.date_range(
        "2025-01-02 00:00", periods=24, freq="h", tz="America/Chicago"
    )
    forecast = torch.arange(24, dtype=torch.float32) + offset
    return {
        "da_price_forecasts": {timestamps[0].date(): forecast},
        "rt_price_forecasts": forecast[:, None].repeat(1, 4),
        "realized_da_price": forecast,
        "realized_rt_price": forecast,
        "local_timestamps": timestamps,
    }


def test_multinode_backtest_starts_each_node_from_its_own_initial_soc():
    cfg = PilotConfig(
        bess_power_mw=1.0,
        bess_energy_mwh=4.0,
        bess_eta=0.95,
        bess_init_soc_frac=0.5,
        bess_kappa=5.7,
        bess_soc_min=0.4,
        bess_soc_max=3.6,
        bess_e_cyc=4.0,
        topk_spread_threshold=5.0,
    )
    result = locked_dual_backtest_by_node(
        {"NODE_B": _node_input(10.0), "NODE_A": _node_input(0.0)},
        cfg,
        da_k_charge=1,
        da_k_discharge=1,
        rt_k_charge=1,
        rt_k_discharge=1,
    )

    assert result["soc_scope"] == "independent_per_node"
    assert result["node_count"] == 2
    for report in result["node_reports"].values():
        assert report["actual_state"]["initial_soc_mwh"] == pytest.approx(2.0)
        assert report["plan_feasibility"]["initial_soc_mwh"] == pytest.approx(2.0)
        assert report["segments"] == 1
