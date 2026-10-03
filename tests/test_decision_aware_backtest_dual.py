from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.backtest_dual import locked_dual_backtest  # noqa: E402
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
    assert result["actual_state"]["initial_soc_mwh"] == pytest.approx(1.0)
    assert result["actual_state"]["segment_initial_soc_mwh"] == pytest.approx([1.0])
    assert result["actual_state"]["segment_final_soc_mwh"] == pytest.approx([1.0])
    assert result["actual_state"]["segment_soc_change_mwh"] == pytest.approx([0.0])
    assert result["actual_state"]["total_segment_soc_change_mwh"] == pytest.approx(0.0)
    assert result["actual_state"]["final_soc_mwh"] == pytest.approx(1.0)
    assert result["actual_state"]["final_minus_initial_soc_mwh"] == pytest.approx(0.0)
    assert result["terminal_inventory_accounting"] == {
        "unit": "MWh",
        "monetized": False,
        "included_in_revenue": False,
        "reporting_rule": "cash revenue and terminal SOC are reported separately",
    }


def test_soc_change_is_reported_per_segment_without_monetization():
    cfg = _tiny_config()
    # 两个完整交付日之间空一天，所以回测会形成两个连续数据段，并在第二段重置SOC。
    timestamps = pd.date_range(
        "2025-01-02 00:00", periods=24, freq="h", tz="America/Chicago"
    ).append(
        pd.date_range(
            "2025-01-04 00:00", periods=24, freq="h", tz="America/Chicago"
        )
    )
    delivery_dates = sorted(set(timestamps.date))
    da_forecasts = {day: torch.zeros(24) for day in delivery_dates}
    # 每个滚动窗口的第一步都是高价，因此RT动作持续尝试放电；每段都会从1 MWh降到0。
    rt_forecasts = torch.tensor([[10.0, 0.0, 0.0, 0.0]]).repeat(48, 1)

    result = locked_dual_backtest(
        da_forecasts,
        rt_forecasts,
        torch.zeros(48),
        torch.zeros(48),
        timestamps,
        cfg,
    )

    assert result["segments"] == 2
    assert result["state_resets_due_to_gaps"] == 1
    assert result["total_revenue"] == pytest.approx(0.0)
    assert result["actual_state"]["segment_initial_soc_mwh"] == pytest.approx(
        [1.0, 1.0]
    )
    assert result["actual_state"]["segment_final_soc_mwh"] == pytest.approx(
        [0.0, 0.0]
    )
    assert result["actual_state"]["segment_soc_change_mwh"] == pytest.approx(
        [-1.0, -1.0]
    )
    assert result["actual_state"]["total_segment_soc_change_mwh"] == pytest.approx(-2.0)
    assert result["actual_state"]["final_soc_mwh"] == pytest.approx(0.0)
    assert result["actual_state"]["final_minus_initial_soc_mwh"] == pytest.approx(-1.0)
    assert result["terminal_inventory_accounting"]["monetized"] is False
    assert result["terminal_inventory_accounting"]["included_in_revenue"] is False


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


def test_unknown_rt_coordination_mode_is_rejected():
    cfg = _tiny_config()
    timestamps = pd.date_range(
        "2025-01-02 00:00", periods=24, freq="h", tz="America/Chicago"
    )
    with pytest.raises(ValueError, match="未知RT协调方式"):
        locked_dual_backtest(
            {timestamps[0].date(): torch.zeros(24)},
            torch.zeros((24, 4)),
            torch.zeros(24),
            torch.zeros(24),
            timestamps,
            cfg,
            coordination_mode="unknown",
        )


def test_rt_at_da_forecast_changes_da_plan_to_use_predicted_spread():
    cfg = _tiny_config()
    timestamps = pd.date_range(
        "2025-01-02 00:00", periods=24, freq="h", tz="America/Chicago"
    )
    # DA预测本身没有高低差。RT-at-DA在第0小时高、在第1小时低，
    # 所以DA-RT价差会让第0小时充电、第1小时放电。
    da_forecast = torch.full((24,), 50.0)
    rt_at_da_forecast = torch.full((24,), 50.0)
    rt_at_da_forecast[0] = 80.0
    rt_at_da_forecast[1] = 20.0
    rt_forecasts = torch.full((24, 4), 50.0)
    realized_da = torch.zeros(24)
    realized_da[0], realized_da[1] = 10.0, 30.0
    realized_rt = torch.zeros(24)

    result = locked_dual_backtest(
        {timestamps[0].date(): da_forecast},
        rt_forecasts,
        realized_da,
        realized_rt,
        timestamps,
        cfg,
        da_k_charge=1,
        da_k_discharge=1,
        coordination_mode="follow_da",
        rt_at_da_price_forecasts={timestamps[0].date(): rt_at_da_forecast},
    )

    assert result["da_decision_signal"] == "predicted_DA_minus_predicted_RT_at_DA"
    assert result["revenue_components"]["da_leg"] == pytest.approx(20.0)


def test_legacy_da_only_signal_remains_explicit_for_old_reports():
    cfg = _tiny_config()
    timestamps = pd.date_range(
        "2025-01-02 00:00", periods=24, freq="h", tz="America/Chicago"
    )
    result = locked_dual_backtest(
        {timestamps[0].date(): torch.arange(24, dtype=torch.float32)},
        torch.zeros((24, 4)),
        torch.zeros(24),
        torch.zeros(24),
        timestamps,
        cfg,
    )
    assert result["da_decision_signal"] == "legacy_predicted_DA_only"
