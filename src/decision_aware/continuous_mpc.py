"""预测驱动的连续日前计划与滚动实时MPC。

日前阶段只读取DA和RT-at-DA预测，并用方向约束联合LP生成锁定计划。实时阶段
每小时只读取当时可见的四小时RT预测，用连续LP求解并只执行第一步。真实
价格只在动作锁定后用于结算，不能进入任何优化器。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .policy import (
    BESSSimulator,
    DualMILPDispatch,
    LPSolveStatus,
    LPOracleSolveError,
)


@dataclass(frozen=True)
class FixedDAWindowResult:
    """固定日前计划下，一个滚动窗口的连续实时解。"""

    net_mwh: np.ndarray
    discharge_mwh: np.ndarray
    charge_mwh: np.ndarray
    excess_mwh: np.ndarray
    soc_path_mwh: np.ndarray
    status: LPSolveStatus


@dataclass(frozen=True)
class PredictedDAPlanResult:
    """价差方向约束下连续联合LP生成的日前计划。"""

    dispatch: DualMILPDispatch
    joint_status: LPSolveStatus


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name}必须是一维有限非空数组")
    return array


def solve_predicted_da_penalty_lp_day(
    predicted_da_price,
    predicted_rt_at_da_price,
    simulator: BESSSimulator,
    *,
    initial_plan_soc_mwh: float,
    initial_actual_soc_mwh: float | None = None,
    sample_index: int = -1,
) -> PredictedDAPlanResult:
    """用价差方向约束和连续联合LP生成可执行日前计划。

    ``p_DA-p_RT-at-DA``的符号只确定每小时充/放方向；联合LP在方向固定后，
    使用两个预测连续优化功率、SOC、退化和3%偏差罚金。这样避免把
    Oracle的真实价格用于决策，也避免24小时混合整数求解的不稳定耗时。
    """
    from scipy.optimize import linprog

    da = _vector(predicted_da_price, "predicted_da_price")
    rt = _vector(predicted_rt_at_da_price, "predicted_rt_at_da_price")
    if da.shape != rt.shape:
        raise ValueError("predicted_da_price与predicted_rt_at_da_price必须同形")
    horizon = int(da.size)
    power = float(simulator.P * simulator.dt)
    eta = float(simulator.eta)
    soc_min = float(simulator.s_min)
    soc_max = float(simulator.s_max)
    cycle_limit = float(simulator.e_cyc)
    kappa = float(simulator.kappa)
    plan_soc0 = float(initial_plan_soc_mwh)
    actual_soc0 = float(
        plan_soc0 if initial_actual_soc_mwh is None else initial_actual_soc_mwh
    )
    for label, value in (("initial_plan_soc_mwh", plan_soc0),
                         ("initial_actual_soc_mwh", actual_soc0)):
        # HiGHS may return a boundary solution a few ulps outside the stated
        # SOC bounds.  The rolling evaluator already treats deviations up to
        # 1e-7 MWh as numerical noise, so use the same tolerance here and
        # project only those boundary-level deviations back onto the feasible
        # interval.  Larger violations remain hard failures.
        if not soc_min - 1e-7 <= value <= soc_max + 1e-7:
            raise ValueError(
                f"{label}超出SOC上下限: value={value:.12g}, "
                f"bounds=[{soc_min:.12g}, {soc_max:.12g}]"
            )
    plan_soc0 = float(np.clip(plan_soc0, soc_min, soc_max))
    actual_soc0 = float(np.clip(actual_soc0, soc_min, soc_max))

    def trajectory_rows(variable_count, d_start, c_start, initial_soc):
        rows = []
        bounds = []
        for step in range(horizon):
            row = np.zeros(variable_count, dtype=np.float64)
            row[d_start:d_start + step + 1] = -1.0 / eta
            row[c_start:c_start + step + 1] = eta
            rows.extend((row, -row))
            bounds.extend((soc_max - initial_soc, initial_soc - soc_min))
        row = np.zeros(variable_count, dtype=np.float64)
        row[d_start:d_start + horizon] = 1.0
        rows.append(row)
        bounds.append(cycle_limit)
        return rows, bounds

    # 双结算中的日前边际信号p_DA-p_RT-at-DA只负责方向。具体功率由
    # 下方联合LP连续决定；价差近似为0时保持空闲。
    spread = da - rt
    direction_discharge = spread > 1e-8
    direction_charge = spread < -1e-8

    dd = slice(0, horizon)
    dc = slice(horizon, 2 * horizon)
    rd = slice(2 * horizon, 3 * horizon)
    rc = slice(3 * horizon, 4 * horizon)
    excess = slice(4 * horizon, 5 * horizon)
    variable_count = 5 * horizon
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[dd] = -spread
    objective[dc] = spread
    objective[rd] = -rt + kappa
    objective[rc] = rt + kappa
    objective[excess] = 2.0 * np.abs(rt)

    rows, upper = trajectory_rows(
        variable_count, dd.start, dc.start, plan_soc0
    )
    actual_rows, actual_upper = trajectory_rows(
        variable_count, rd.start, rc.start, actual_soc0
    )
    rows.extend(actual_rows)
    upper.extend(actual_upper)
    for step in range(horizon):
        tolerance_terms = (
            (dd.start + step, 1.03), (dc.start + step, -0.97),
            (rd.start + step, -1.0), (rc.start + step, 1.0),
            (excess.start + step, 1.0),
        )
        row = np.zeros(variable_count, dtype=np.float64)
        for column, coefficient in tolerance_terms:
            row[column] = -coefficient
        rows.append(row)
        upper.append(0.0)
        tolerance_terms = (
            (dd.start + step, -0.97), (dc.start + step, 1.03),
            (rd.start + step, 1.0), (rc.start + step, -1.0),
            (excess.start + step, 1.0),
        )
        row = np.zeros(variable_count, dtype=np.float64)
        for column, coefficient in tolerance_terms:
            row[column] = -coefficient
        rows.append(row)
        upper.append(0.0)

    bounds = []
    bounds.extend((0.0, power if allowed else 0.0) for allowed in direction_discharge)
    bounds.extend((0.0, power if allowed else 0.0) for allowed in direction_charge)
    bounds.extend([(0.0, power)] * (2 * horizon))
    bounds.extend([(0.0, 2.0 * power)] * horizon)
    phase2 = linprog(
        objective,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(upper),
        bounds=bounds,
        method="highs",
    )
    joint_status = LPSolveStatus(
        sample_index=int(sample_index),
        leg="predicted_joint_da_rt_at_da_penalty_continuous_lp",
        success=bool(phase2.success),
        status_code=int(phase2.status),
        message=str(phase2.message),
        objective=float(phase2.fun) if phase2.success else None,
    )
    if not phase2.success:
        raise LPOracleSolveError(joint_status)
    solution = np.asarray(phase2.x, dtype=np.float64)
    plan_dis = np.clip(solution[dd], 0.0, power)
    plan_chg = np.clip(solution[dc], 0.0, power)
    actual_dis = np.clip(solution[rd], 0.0, power)
    actual_chg = np.clip(solution[rc], 0.0, power)
    plan_net = plan_dis - plan_chg
    actual_net = actual_dis - actual_chg
    deviation = actual_net - plan_net
    tolerance = 0.03 * np.abs(plan_net)
    exact_excess = np.maximum(np.abs(deviation) - tolerance, 0.0)

    def soc_path(discharge_values, charge_values, initial):
        return np.concatenate((
            [initial], initial + np.cumsum(
                -discharge_values / eta + charge_values * eta
            )
        ))

    dispatch = DualMILPDispatch(
        da_discharge_mwh=plan_dis,
        da_charge_mwh=plan_chg,
        rt_discharge_mwh=actual_dis,
        rt_charge_mwh=actual_chg,
        da_soc_path_mwh=soc_path(plan_dis, plan_chg, plan_soc0),
        rt_soc_path_mwh=soc_path(actual_dis, actual_chg, actual_soc0),
        deviation_mwh=deviation,
        tolerance_mwh=tolerance,
        excess_mwh=exact_excess,
        degradation_cost=kappa * float(np.sum(actual_dis + actual_chg)),
        deviation_penalty=float(np.sum(2.0 * np.abs(rt) * exact_excess)),
    )
    return PredictedDAPlanResult(
        dispatch=dispatch,
        joint_status=joint_status,
    )


def solve_fixed_da_penalty_lp_window(
    predicted_rt_price,
    locked_da_net_mwh,
    simulator: BESSSimulator,
    *,
    initial_soc_mwh: float,
    initial_discharged_mwh: float = 0.0,
    cycle_reset_mask: Sequence[bool] | None = None,
    terminal_value_weight: float = 1.0,
    sample_index: int = -1,
) -> FixedDAWindowResult:
    """求固定DA计划下的连续RT滚动LP。

    偏差罚金系数使用预测RT价，因为求解时真实RT价尚不可见。动作执行后，
    正式账本仍使用真实RT价重新计算收益和罚金。
    """
    from scipy.optimize import linprog

    price = _vector(predicted_rt_price, "predicted_rt_price")
    da = _vector(locked_da_net_mwh, "locked_da_net_mwh")
    if price.shape != da.shape:
        raise ValueError("predicted_rt_price与locked_da_net_mwh必须同形")
    horizon = int(price.size)
    if cycle_reset_mask is None:
        reset = np.zeros(horizon, dtype=bool)
    else:
        reset = np.asarray(cycle_reset_mask, dtype=bool)
        if reset.shape != (horizon,):
            raise ValueError("cycle_reset_mask长度必须等于滚动时域")
    reset[0] = False

    power = float(simulator.P * simulator.dt)
    eta = float(simulator.eta)
    soc_min = float(simulator.s_min)
    soc_max = float(simulator.s_max)
    cycle_limit = float(simulator.e_cyc)
    kappa = float(simulator.kappa)
    soc0 = float(initial_soc_mwh)
    discharged0 = float(initial_discharged_mwh)
    if power <= 0.0 or eta <= 0.0 or cycle_limit < 0.0:
        raise ValueError("无效的功率、效率或日放电上限")
    if not soc_min - 1e-9 <= soc0 <= soc_max + 1e-9:
        raise ValueError("initial_soc_mwh超出SOC上下限")
    if not 0.0 <= discharged0 <= cycle_limit + 1e-9:
        raise ValueError("initial_discharged_mwh超出日放电上限")
    if terminal_value_weight < 0.0:
        raise ValueError("terminal_value_weight不能为负")

    discharge = slice(0, horizon)
    charge = slice(horizon, 2 * horizon)
    excess = slice(2 * horizon, 3 * horizon)
    objective = np.zeros(3 * horizon, dtype=np.float64)
    objective[discharge] = -price + kappa
    objective[charge] = price + kappa
    objective[excess] = 2.0 * np.abs(price)

    # 与旧离散MPC相同：有限窗口末端按最后一个可见预测价保留库存机会价值。
    terminal_price = terminal_value_weight * max(float(price[-1]), 0.0)
    objective[discharge] += terminal_price
    objective[charge] -= terminal_price * eta * eta

    rows: list[np.ndarray] = []
    upper: list[float] = []

    def add(terms: Sequence[tuple[int, float]], bound: float) -> None:
        row = np.zeros(3 * horizon, dtype=np.float64)
        for column, coefficient in terms:
            row[column] += coefficient
        rows.append(row)
        upper.append(float(bound))

    # 每个前缀都必须满足真实SOC边界。
    for step in range(horizon):
        terms = []
        for index in range(step + 1):
            terms.extend(((discharge.start + index, -1.0 / eta),
                          (charge.start + index, eta)))
        add(terms, soc_max - soc0)
        add([(column, -coefficient) for column, coefficient in terms],
            soc0 - soc_min)

    # 当前日使用剩余额度；跨到下一当地日后重新获得完整额度。
    segment_start = 0
    segment_limit = max(0.0, cycle_limit - discharged0)
    for step in range(1, horizon + 1):
        if step == horizon or reset[step]:
            add(
                [(discharge.start + index, 1.0)
                 for index in range(segment_start, step)],
                segment_limit,
            )
            segment_start = step
            segment_limit = cycle_limit

    tolerance = 0.03 * np.abs(da)
    for step in range(horizon):
        # excess >= +(uRT-uDA)-tolerance
        add(
            ((discharge.start + step, 1.0),
             (charge.start + step, -1.0),
             (excess.start + step, -1.0)),
            da[step] + tolerance[step],
        )
        # excess >= -(uRT-uDA)-tolerance
        add(
            ((discharge.start + step, -1.0),
             (charge.start + step, 1.0),
             (excess.start + step, -1.0)),
            -da[step] + tolerance[step],
        )

    result = linprog(
        objective,
        A_ub=np.asarray(rows, dtype=np.float64),
        b_ub=np.asarray(upper, dtype=np.float64),
        bounds=[(0.0, power)] * (2 * horizon)
        + [(0.0, 2.0 * power)] * horizon,
        method="highs",
    )
    status = LPSolveStatus(
        sample_index=int(sample_index),
        leg="predicted_rt_fixed_da_continuous_mpc",
        success=bool(result.success),
        status_code=int(result.status),
        message=str(result.message),
        objective=float(result.fun) if result.success else None,
    )
    if not result.success:
        raise LPOracleSolveError(status)
    solution = np.asarray(result.x, dtype=np.float64)
    dis = np.clip(solution[discharge], 0.0, power)
    chg = np.clip(solution[charge], 0.0, power)
    dis[np.abs(dis) < 1e-9] = 0.0
    chg[np.abs(chg) < 1e-9] = 0.0
    net = dis - chg
    soc_path = np.concatenate((
        [soc0], soc0 + np.cumsum(-dis / eta + chg * eta)
    ))
    exact_excess = np.maximum(np.abs(net - da) - tolerance, 0.0)
    return FixedDAWindowResult(
        net_mwh=net,
        discharge_mwh=dis,
        charge_mwh=chg,
        excess_mwh=exact_excess,
        soc_path_mwh=soc_path,
        status=status,
    )


def _simulator(cfg) -> BESSSimulator:
    return BESSSimulator(
        cfg.bess_power_mw,
        cfg.bess_energy_mwh,
        cfg.bess_eta,
        cfg.bess_init_soc_frac,
        dt=1.0,
        kappa=cfg.bess_kappa,
        soc_min=cfg.bess_soc_min,
        soc_max=cfg.bess_soc_max,
        e_cyc=cfg.bess_e_cyc,
    )


def _digest(values: Sequence[float]) -> str:
    import hashlib

    array = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _soc_summary(initial_soc: float, finals: Sequence[float]) -> dict:
    return {
        "initial_soc_mwh_each_segment": float(initial_soc),
        "segment_final_soc_mwh": [float(value) for value in finals],
        "final_soc_mwh": float(finals[-1]),
        "segment_soc_change_mwh": [float(value - initial_soc) for value in finals],
    }


def locked_continuous_mpc_backtest(
    da_price_forecasts: Mapping[date, Sequence[float]],
    rt_at_da_price_forecasts: Mapping[date, Sequence[float]],
    rt_price_forecasts,
    realized_da_price,
    realized_rt_price,
    local_timestamps: Sequence,
    cfg,
    *,
    terminal_value_weight: float = 1.0,
    include_hourly_details: bool = False,
) -> dict:
    """以严格信息边界运行连续DA-LP与每小时RT-LP回测。"""
    rt_forecasts = np.asarray(rt_price_forecasts, dtype=np.float64)
    true_da = _vector(realized_da_price, "realized_da_price")
    true_rt = _vector(realized_rt_price, "realized_rt_price")
    if rt_forecasts.ndim != 2 or not np.isfinite(rt_forecasts).all():
        raise ValueError("rt_price_forecasts必须是二维有限数组")
    if not (len(rt_forecasts) == len(true_da) == len(true_rt) == len(local_timestamps)):
        raise ValueError("RT预测、真值和时间戳长度必须一致")
    timestamps = [pd.Timestamp(value) for value in local_timestamps]
    if any(value.tzinfo is None for value in timestamps):
        raise ValueError("local_timestamps必须带时区")
    utc = [value.tz_convert("UTC") for value in timestamps]
    one_hour = pd.Timedelta(hours=1)

    positions_by_date: dict[date, list[int]] = defaultdict(list)
    for position, timestamp in enumerate(timestamps):
        positions_by_date[timestamp.date()].append(position)
    eligible: dict[date, list[int]] = {}
    for delivery_date, positions in positions_by_date.items():
        local_hours = [timestamps[position].hour for position in positions]
        continuous = all(
            utc[right] - utc[left] == one_hour
            for left, right in zip(positions, positions[1:])
        )
        if (
            len(positions) == 24
            and sorted(local_hours) == list(range(24))
            and continuous
            and delivery_date in da_price_forecasts
            and delivery_date in rt_at_da_price_forecasts
        ):
            eligible[delivery_date] = positions
    if not eligible:
        raise ValueError("没有具备完整预测和24小时数据的交付日")

    simulator = _simulator(cfg)
    initial_soc = float(cfg.bess_energy_mwh * cfg.bess_init_soc_frac)
    ordered_dates = sorted(eligible)
    da_plans: dict[date, np.ndarray] = {}
    plan_soc_end: dict[date, float] = {}
    plan_soc = initial_soc
    segment_starts: list[str] = []
    plan_segment_finals: list[float] = []
    previous_last = None
    da_solver_calls = 0
    da_solver_seconds = 0.0
    for delivery_date in ordered_dates:
        positions = eligible[delivery_date]
        starts_segment = previous_last is None or utc[positions[0]] - previous_last != one_hour
        if starts_segment:
            if segment_starts:
                plan_segment_finals.append(plan_soc)
            segment_starts.append(str(delivery_date))
            plan_soc = initial_soc
        solver_started = perf_counter()
        solved = solve_predicted_da_penalty_lp_day(
            da_price_forecasts[delivery_date],
            rt_at_da_price_forecasts[delivery_date],
            simulator,
            sample_index=da_solver_calls,
            initial_plan_soc_mwh=plan_soc,
            initial_actual_soc_mwh=plan_soc,
        )
        da_solver_seconds += perf_counter() - solver_started
        plan = solved.dispatch.da_discharge_mwh - solved.dispatch.da_charge_mwh
        da_plans[delivery_date] = plan
        plan_soc = float(solved.dispatch.da_soc_path_mwh[-1])
        plan_soc_end[delivery_date] = plan_soc
        previous_last = utc[positions[-1]]
        da_solver_calls += 1
    plan_segment_finals.append(plan_soc)

    actual_soc = initial_soc
    actual_segment_finals: list[float] = []
    previous_last = None
    discharged_today = 0.0
    daily = {}
    all_actions: list[float] = []
    all_plans: list[float] = []
    hourly_details: list[dict] = []
    rt_solver_calls = 0
    rt_solver_seconds = 0.0
    soc_violation_count = 0
    for delivery_date in ordered_dates:
        positions = eligible[delivery_date]
        starts_segment = previous_last is None or utc[positions[0]] - previous_last != one_hour
        if starts_segment:
            if daily:
                actual_segment_finals.append(actual_soc)
            actual_soc = initial_soc
        discharged_today = 0.0
        components = {
            "da_leg": 0.0,
            "rt_deviation_leg": 0.0,
            "degradation_cost": 0.0,
            "deviation_penalty": 0.0,
            "actual_throughput_mwh": 0.0,
        }
        for hour, position in enumerate(positions):
            max_horizon = int(rt_forecasts.shape[1])
            valid = 1
            for step in range(1, max_horizon):
                future = position + step
                if future >= len(timestamps) or utc[future] - utc[position] != step * one_hour:
                    break
                if timestamps[future].date() not in da_plans:
                    break
                valid += 1
            future_da = np.asarray([
                da_plans[timestamps[position + step].date()][timestamps[position + step].hour]
                for step in range(valid)
            ])
            reset_mask = np.asarray([
                False if step == 0 else (
                    timestamps[position + step].date()
                    != timestamps[position + step - 1].date()
                )
                for step in range(valid)
            ])
            solver_started = perf_counter()
            solved = solve_fixed_da_penalty_lp_window(
                rt_forecasts[position, :valid],
                future_da,
                simulator,
                initial_soc_mwh=actual_soc,
                initial_discharged_mwh=discharged_today,
                cycle_reset_mask=reset_mask,
                terminal_value_weight=terminal_value_weight,
                sample_index=rt_solver_calls,
            )
            rt_solver_seconds += perf_counter() - solver_started
            actual_net = float(solved.net_mwh[0])
            actual_discharge = float(solved.discharge_mwh[0])
            actual_charge = float(solved.charge_mwh[0])
            da_net = float(da_plans[delivery_date][hour])
            actual_soc = float(solved.soc_path_mwh[1])
            soc_violation_count += int(
                actual_soc < cfg.bess_soc_min - 1e-7
                or actual_soc > cfg.bess_soc_max + 1e-7
            )
            discharged_today += actual_discharge
            p_da = float(true_da[position])
            p_rt = float(true_rt[position])
            excess = max(0.0, abs(actual_net - da_net) - 0.03 * abs(da_net))
            components["da_leg"] += da_net * p_da
            components["rt_deviation_leg"] += (actual_net - da_net) * p_rt
            components["degradation_cost"] += cfg.bess_kappa * (
                actual_discharge + actual_charge
            )
            components["deviation_penalty"] += 2.0 * abs(p_rt) * excess
            components["actual_throughput_mwh"] += actual_discharge + actual_charge
            all_actions.append(actual_net / cfg.bess_power_mw)
            all_plans.append(da_net / cfg.bess_power_mw)
            if include_hourly_details:
                da_leg = da_net * p_da
                rt_leg = (actual_net - da_net) * p_rt
                degradation = cfg.bess_kappa * (
                    actual_discharge + actual_charge
                )
                penalty = 2.0 * abs(p_rt) * excess
                hourly_details.append({
                    "timestamp_local": timestamps[position].isoformat(),
                    "delivery_date": str(delivery_date),
                    "hour_local": int(timestamps[position].hour),
                    "predicted_da_price": float(
                        da_price_forecasts[delivery_date][hour]
                    ),
                    "predicted_rt_at_da_price": float(
                        rt_at_da_price_forecasts[delivery_date][hour]
                    ),
                    "predicted_rt_price": float(rt_forecasts[position, 0]),
                    "realized_da_price": p_da,
                    "realized_rt_price": p_rt,
                    "da_plan_mwh": da_net,
                    "actual_dispatch_mwh": actual_net,
                    "actual_discharge_mwh": actual_discharge,
                    "actual_charge_mwh": actual_charge,
                    "deviation_mwh": actual_net - da_net,
                    "excess_deviation_mwh": excess,
                    "da_leg": da_leg,
                    "rt_deviation_leg": rt_leg,
                    "degradation_cost": degradation,
                    "deviation_penalty": penalty,
                    "net_revenue": da_leg + rt_leg - degradation - penalty,
                    "actual_soc_end_mwh": actual_soc,
                })
            rt_solver_calls += 1
        net = (
            components["da_leg"] + components["rt_deviation_leg"]
            - components["degradation_cost"] - components["deviation_penalty"]
        )
        daily[str(delivery_date)] = {
            "net_revenue": net,
            **components,
            "actual_soc_end_mwh": actual_soc,
            "plan_soc_end_mwh": plan_soc_end[delivery_date],
            "plan_clipped_mwh": 0.0,
        }
        previous_last = utc[positions[-1]]
    actual_segment_finals.append(actual_soc)

    values = list(daily.values())
    total_da = sum(row["da_leg"] for row in values)
    total_rt = sum(row["rt_deviation_leg"] for row in values)
    total_degradation = sum(row["degradation_cost"] for row in values)
    total_penalty = sum(row["deviation_penalty"] for row in values)
    total = total_da + total_rt - total_degradation - total_penalty
    result = {
        "contract": "predicted spread-direction continuous DA LP + hourly rolling RT LP/MPC",
        "information_boundary": "forecasts_only_for_decisions; truth_only_after_action",
        "coordination": {"mode": "continuous_penalty_mpc"},
        "solver_calls": {
            "da_joint_penalty_lp": da_solver_calls,
            "rt_lp": rt_solver_calls,
        },
        "solver_timing_seconds": {
            "da_joint_penalty_lp": da_solver_seconds,
            "rt_lp": rt_solver_seconds,
            "total": da_solver_seconds + rt_solver_seconds,
            "da_mean_per_call": da_solver_seconds / da_solver_calls,
            "rt_mean_per_call": rt_solver_seconds / rt_solver_calls,
        },
        "days": len(values),
        "hours": 24 * len(values),
        "segments": len(segment_starts),
        "segment_start_dates": segment_starts,
        "state_resets_due_to_gaps": max(0, len(segment_starts) - 1),
        "total_revenue": total,
        "mean_daily_revenue": total / len(values),
        "positive_day_rate": sum(row["net_revenue"] > 0 for row in values) / len(values),
        "actual_throughput_mwh": sum(
            row["actual_throughput_mwh"] for row in values
        ),
        "soc_violation_count": soc_violation_count,
        "revenue_components": {
            "da_leg": total_da,
            "rt_deviation_leg": total_rt,
            "degradation_cost": total_degradation,
            "deviation_penalty": total_penalty,
        },
        "plan_feasibility": {
            "total_clipped_mwh": 0.0,
            "clipped_days": 0,
            **_soc_summary(initial_soc, plan_segment_finals),
        },
        "actual_state": {
            **_soc_summary(initial_soc, actual_segment_finals),
            "action_digest": _digest(all_actions),
        },
        "da_plan_action_digest": _digest(all_plans),
        "terminal_inventory_accounting": {
            "unit": "MWh",
            "monetized": False,
            "included_in_revenue": False,
            "reporting_rule": "cash revenue and terminal SOC are reported separately",
        },
        "daily": daily,
    }
    if include_hourly_details:
        result["hourly"] = hourly_details
    return result


def locked_continuous_mpc_backtest_by_node(
    node_inputs: Mapping[str, Mapping[str, object]], cfg, **kwargs
) -> dict:
    reports = {}
    for node in sorted(node_inputs):
        item = node_inputs[node]
        reports[node] = locked_continuous_mpc_backtest(
            item["da_price_forecasts"],
            item["rt_at_da_price_forecasts"],
            item["rt_price_forecasts"],
            item["realized_da_price"],
            item["realized_rt_price"],
            item["local_timestamps"],
            cfg,
            **kwargs,
        )
    return {
        "node_count": len(reports),
        "soc_scope": "independent_per_node",
        "macro_mean_daily_revenue": float(np.mean([
            item["mean_daily_revenue"] for item in reports.values()
        ])),
        "node_reports": reports,
    }
