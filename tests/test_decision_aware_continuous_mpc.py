from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.continuous_mpc import (
    locked_continuous_mpc_backtest,
    solve_fixed_da_penalty_lp_window,
    solve_predicted_da_penalty_lp_day,
)
from decision_aware.policy import BESSSimulator, solve_dual_penalty_milp_day


def _simulator():
    return BESSSimulator(
        1.0, 4.0, 0.95, 0.5, dt=1.0, kappa=5.7,
        soc_min=0.4, soc_max=3.6, e_cyc=4.0,
    )


def _cfg():
    return SimpleNamespace(
        bess_power_mw=1.0,
        bess_energy_mwh=4.0,
        bess_eta=0.95,
        bess_init_soc_frac=0.5,
        bess_kappa=5.7,
        bess_soc_min=0.4,
        bess_soc_max=3.6,
        bess_e_cyc=4.0,
    )


def test_day_milp_accepts_separate_initial_soc_values():
    solved = solve_dual_penalty_milp_day(
        np.linspace(-10.0, 50.0, 24),
        np.linspace(-5.0, 45.0, 24),
        _simulator(),
        initial_plan_soc_mwh=1.25,
        initial_actual_soc_mwh=2.75,
    )
    assert solved.dispatch.da_soc_path_mwh[0] == pytest.approx(1.25)
    assert solved.dispatch.rt_soc_path_mwh[0] == pytest.approx(2.75)
    assert np.all(solved.dispatch.da_soc_path_mwh >= 0.4 - 1e-7)
    assert np.all(solved.dispatch.rt_soc_path_mwh <= 3.6 + 1e-7)


def test_fixed_da_continuous_lp_respects_soc_cycle_and_penalty_band():
    solved = solve_fixed_da_penalty_lp_window(
        [100.0, 80.0, 30.0, 20.0],
        [1.0, 1.0, 0.0, 0.0],
        _simulator(),
        initial_soc_mwh=3.6,
        initial_discharged_mwh=3.5,
        cycle_reset_mask=[False, False, False, False],
        terminal_value_weight=0.0,
    )
    assert solved.discharge_mwh.sum() <= 0.5 + 1e-7
    assert np.all(solved.soc_path_mwh >= 0.4 - 1e-7)
    assert np.all(solved.soc_path_mwh <= 3.6 + 1e-7)
    assert np.all(solved.discharge_mwh * solved.charge_mwh <= 1e-8)


def test_da_direction_uses_da_minus_rt_at_da_spread():
    early = solve_predicted_da_penalty_lp_day(
        [50.0, 50.0],
        [0.0, 100.0],
        _simulator(),
        initial_plan_soc_mwh=3.6,
    )
    late = solve_predicted_da_penalty_lp_day(
        [50.0, 50.0],
        [100.0, 0.0],
        _simulator(),
        initial_plan_soc_mwh=3.6,
    )
    assert early.dispatch.da_discharge_mwh[0] > 0.0
    assert early.dispatch.da_discharge_mwh[1] == pytest.approx(0.0, abs=1e-8)
    assert late.dispatch.da_discharge_mwh[0] == pytest.approx(0.0, abs=1e-8)
    assert late.dispatch.da_discharge_mwh[1] > 0.0


def test_backtest_decisions_do_not_use_realized_prices():
    delivery_date = date(2023, 1, 3)
    timestamps = pd.date_range(
        "2023-01-03 00:00", periods=24, freq="h", tz="America/Chicago"
    )
    da_prediction = np.r_[np.full(12, -20.0), np.full(12, 100.0)]
    rt_da_prediction = np.zeros(24)
    rt_prediction = np.tile(np.asarray([10.0, 15.0, 20.0, 25.0]), (24, 1))

    def run(realized_scale):
        return locked_continuous_mpc_backtest(
            {delivery_date: da_prediction},
            {delivery_date: rt_da_prediction},
            rt_prediction,
            np.linspace(-30.0, 120.0, 24) * realized_scale,
            np.linspace(-10.0, 90.0, 24) * realized_scale,
            timestamps,
            _cfg(),
        )

    first = run(1.0)
    second = run(-3.0)
    assert first["da_plan_action_digest"] == second["da_plan_action_digest"]
    assert first["actual_state"]["action_digest"] == second["actual_state"]["action_digest"]
    assert first["total_revenue"] != second["total_revenue"]
    assert first["solver_calls"] == {
        "da_joint_penalty_lp": 1,
        "rt_lp": 24,
    }
