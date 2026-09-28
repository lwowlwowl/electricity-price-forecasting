from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.backtest_dual import (  # noqa: E402
    coordinated_rt_intent,
    locked_dual_backtest,
)
from decision_aware.config import PilotConfig  # noqa: E402


def _tiny_config() -> PilotConfig:
    return PilotConfig(
        bess_power_mw=1.0,
        bess_energy_mwh=2.0,
        bess_eta=1.0,
        bess_init_soc_frac=0.5,
        bess_kappa=0.0,
        bess_soc_min=0.0,
        bess_soc_max=2.0,
        bess_e_cyc=2.0,
        topk_spread_threshold=5.0,
    )


def test_locked_plan_and_identical_actual_action_reduce_to_da_revenue():
    cfg = _tiny_config()
    timestamps = pd.date_range(
        "2025-01-02 00:00", periods=24, freq="h", tz="America/Chicago"
    )
    # DA: 第0小时充电，第1小时放电，其余不动。
    da_forecast = torch.full((24,), 5.0)
    da_forecast[0] = 0.0
    da_forecast[1] = 10.0

    # RT每小时都有4小时窗口；只让前两个窗口的第一步产生同样动作。
    rt_forecasts = torch.full((24, 4), 5.0)
    rt_forecasts[0] = torch.tensor([0.0, 10.0, 4.0, 4.0])
    rt_forecasts[1] = torch.tensor([10.0, 0.0, 4.0, 4.0])
    price_da = torch.zeros(24)
    price_da[0], price_da[1] = 10.0, 30.0
    price_rt = torch.linspace(-100.0, 100.0, 24)

    result = locked_dual_backtest(
        {timestamps[0].date(): da_forecast},
        rt_forecasts,
        price_da,
        price_rt,
        timestamps,
        cfg,
        da_k_charge=1,
        da_k_discharge=1,
    )

    # 实际动作与计划相同，所以RT偏差腿为0；DA低买高卖赚20。
    assert result["total_revenue"] == pytest.approx(20.0)
    assert result["revenue_components"]["da_leg"] == pytest.approx(20.0)
    assert result["revenue_components"]["rt_deviation_leg"] == pytest.approx(0.0)
    assert result["plan_feasibility"]["total_clipped_mwh"] == pytest.approx(0.0)
    assert result["actual_state"]["segment_final_soc_mwh"] == pytest.approx([1.0])


def test_dst_day_is_not_forced_into_fake_24_hour_plan():
    cfg = _tiny_config()
    # 芝加哥春季切换日只有23个当地小时。
    timestamps = pd.date_range(
        "2025-03-09 00:00", "2025-03-09 23:00", freq="h", tz="America/Chicago"
    )
    assert len(timestamps) == 23
    forecasts = torch.zeros((23, 4))
    prices = torch.zeros(23)

    with pytest.raises(ValueError, match="没有同时具备完整24小时"):
        locked_dual_backtest(
            {timestamps[0].date(): torch.zeros(24)},
            forecasts,
            prices,
            prices,
            timestamps,
            cfg,
        )


def test_fixed_rt_coordination_modes_are_explicit_and_auditable():
    cfg = _tiny_config()

    action, used_candidate, _ = coordinated_rt_intent(
        da_action=-1.0,
        rt_candidate=1.0,
        predicted_rt_price=50.0,
        cfg=cfg,
        mode="follow_da",
    )
    assert action == -1.0
    assert not used_candidate

    action, used_candidate, gain = coordinated_rt_intent(
        da_action=-1.0,
        rt_candidate=1.0,
        predicted_rt_price=50.0,
        cfg=cfg,
        mode="track_adjust",
    )
    assert action == 1.0
    assert used_candidate
    assert gain == pytest.approx(100.0)

    action, used_candidate, gain = coordinated_rt_intent(
        da_action=-1.0,
        rt_candidate=1.0,
        predicted_rt_price=-50.0,
        cfg=cfg,
        mode="track_adjust",
    )
    assert action == -1.0
    assert not used_candidate
    assert gain == pytest.approx(-100.0)

    with pytest.raises(ValueError, match="未知RT协调方式"):
        coordinated_rt_intent(0.0, 0.0, 0.0, cfg, mode="unknown")
