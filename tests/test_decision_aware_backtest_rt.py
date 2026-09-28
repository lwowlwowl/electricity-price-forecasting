import os
import sys
from datetime import date

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decision_aware.backtest_rt import rolling_rt_backtest
from decision_aware.config import PilotConfig


def test_rolling_rt_executes_only_first_action_and_keeps_soc():
    cfg = PilotConfig(
        horizon_rt=4,
        bess_kappa=5.7,
        bess_soc_min=0.4,
        bess_soc_max=3.6,
        bess_e_cyc=4.0,
        topk_spread_threshold=0.0,
    )
    forecasts = torch.tensor([
        [10.0, 20.0, 30.0, 40.0],
        [40.0, 30.0, 20.0, 10.0],
    ])
    prices = torch.tensor([10.0, 40.0])
    report = rolling_rt_backtest(
        forecasts, prices, [date(2025, 1, 1), date(2025, 1, 1)], cfg
    )
    assert report["actual_actions"] == [-1.0, 1.0]
    assert abs(report["total_revenue"] - 18.6) < 1e-5
    assert report["days"] == 1


def test_daily_cycle_counter_resets_but_soc_does_not():
    cfg = PilotConfig(
        horizon_rt=2, bess_e_cyc=1.0, topk_spread_threshold=0.0,
        bess_kappa=0.0, bess_soc_min=0.0, bess_soc_max=4.0,
    )
    forecasts = torch.tensor([[50.0, 10.0], [50.0, 10.0]])
    prices = torch.tensor([50.0, 50.0])
    report = rolling_rt_backtest(
        forecasts, prices, [date(2025, 1, 1), date(2025, 1, 2)], cfg
    )
    assert report["days"] == 2
    assert report["actual_actions"][0] > 0
    assert report["actual_actions"][1] > 0
