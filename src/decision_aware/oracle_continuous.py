"""Exact dual-settlement MILP Oracle for continuous multi-day segments.

The one-day Oracle in :mod:`decision_aware.policy` is useful for unit tests and
small diagnostics, but resetting SOC at every midnight does not match the
joint-system backtest.  This module keeps separate planned and actual SOC
trajectories continuous across all days in one data segment, while resetting
the discharge-throughput limit at each delivery day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import BESSSimulator, MILPOracleSolveError, MILPSolveStatus


@dataclass(frozen=True)
class ContinuousDualMILPDispatch:
    """Auditable dispatch for one continuous segment.

    Hourly arrays use shape ``[days, hours_per_day]``.  SOC paths are flat and
    include the segment's initial value, so each has ``days*hours+1`` entries.
    """

    da_net_mwh: np.ndarray
    rt_net_mwh: np.ndarray
    da_soc_path_mwh: np.ndarray
    rt_soc_path_mwh: np.ndarray
    deviation_mwh: np.ndarray
    tolerance_mwh: np.ndarray
    excess_mwh: np.ndarray
    hourly_revenue: np.ndarray
    daily_revenue: np.ndarray
    daily_da_leg: np.ndarray
    daily_rt_deviation_leg: np.ndarray
    daily_degradation_cost: np.ndarray
    daily_deviation_penalty: np.ndarray


@dataclass(frozen=True)
class ContinuousDualMILPOracleResult:
    """Certified optimum and dispatch for one continuous data segment."""

    revenue: float
    status: MILPSolveStatus
    dispatch: ContinuousDualMILPDispatch


def _finite_price_matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name}必须是[days,hours]二维非空数组")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}包含NaN或无穷值")
    return array


def solve_dual_penalty_milp_segment(
    price_da,
    price_rt,
    simulator: BESSSimulator,
    *,
    initial_plan_soc_mwh: float | None = None,
    initial_actual_soc_mwh: float | None = None,
    segment_index: int = -1,
    mip_rel_gap: float = 0.0,
) -> ContinuousDualMILPOracleResult:
    """Solve one globally optimal continuous multi-day segment.

    This is the exact perfect-foresight upper bound for the physical and cash
    settlement contract used by ``settle_joint_episode``:

    * planned and actual SOC are two independent physical trajectories;
    * both SOC trajectories continue through midnight;
    * the discharge-throughput cap resets once per delivery day;
    * deviation tolerance is ``3%*abs(u_DA)`` and excess is charged at
      ``2*abs(price_RT)``;
    * terminal SOC is neither constrained nor monetised.

    A sparse recurrence formulation is used so a months-long segment does not
    create the quadratic dense SOC matrix used by the small one-day helper.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    da_matrix = _finite_price_matrix(price_da, "price_da")
    rt_matrix = _finite_price_matrix(price_rt, "price_rt")
    if da_matrix.shape != rt_matrix.shape:
        raise ValueError("price_da和price_rt必须同形")

    days, hours_per_day = da_matrix.shape
    horizon = int(days * hours_per_day)
    da = da_matrix.reshape(-1)
    rt = rt_matrix.reshape(-1)

    power_energy = float(simulator.P * simulator.dt)
    eta = float(simulator.eta)
    soc_min = float(simulator.s_min)
    soc_max = float(simulator.s_max)
    cycle_limit = float(simulator.e_cyc)
    kappa = float(simulator.kappa)
    default_initial = float(simulator.E * simulator.init_soc_frac)
    initial_plan = float(
        default_initial if initial_plan_soc_mwh is None else initial_plan_soc_mwh
    )
    initial_actual = float(
        default_initial
        if initial_actual_soc_mwh is None
        else initial_actual_soc_mwh
    )
    if power_energy <= 0.0 or eta <= 0.0:
        raise ValueError("MILP Oracle要求正的功率、步长和效率")
    if cycle_limit < 0.0 or soc_min > soc_max:
        raise ValueError("MILP Oracle收到无效的循环或SOC边界")
    for label, value in (
        ("initial_plan_soc_mwh", initial_plan),
        ("initial_actual_soc_mwh", initial_actual),
    ):
        if not soc_min - 1e-9 <= value <= soc_max + 1e-9:
            raise ValueError(f"{label}超出SOC上下限")
    if mip_rel_gap < 0.0:
        raise ValueError("mip_rel_gap不能为负")

    # Nine H-sized blocks: four non-negative dispatch blocks, excess,
    # planned/actual end-of-hour SOC, and two binary mode blocks.
    dd = slice(0, horizon)
    dc = slice(horizon, 2 * horizon)
    rd = slice(2 * horizon, 3 * horizon)
    rc = slice(3 * horizon, 4 * horizon)
    excess_var = slice(4 * horizon, 5 * horizon)
    da_soc = slice(5 * horizon, 6 * horizon)
    rt_soc = slice(6 * horizon, 7 * horizon)
    da_mode = slice(7 * horizon, 8 * horizon)
    rt_mode = slice(8 * horizon, 9 * horizon)
    variable_count = 9 * horizon

    spread = da - rt
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[dd] = -spread
    objective[dc] = spread
    objective[rd] = -rt + kappa
    objective[rc] = rt + kappa
    objective[excess_var] = 2.0 * np.abs(rt)

    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.empty(variable_count, dtype=np.float64)
    for block in (dd, dc, rd, rc):
        upper_bounds[block] = power_energy
    upper_bounds[excess_var] = 2.0 * power_energy
    lower_bounds[da_soc] = soc_min
    upper_bounds[da_soc] = soc_max
    lower_bounds[rt_soc] = soc_min
    upper_bounds[rt_soc] = soc_max
    upper_bounds[da_mode] = 1.0
    upper_bounds[rt_mode] = 1.0
    integrality = np.zeros(variable_count, dtype=np.int8)
    integrality[da_mode] = 1
    integrality[rt_mode] = 1

    row_ids: list[int] = []
    col_ids: list[int] = []
    coefficients: list[float] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(terms, lower=-np.inf, upper=np.inf):
        row = len(row_lower)
        for column, coefficient in terms:
            if coefficient != 0.0:
                row_ids.append(row)
                col_ids.append(int(column))
                coefficients.append(float(coefficient))
        row_lower.append(float(lower))
        row_upper.append(float(upper))

    for step in range(horizon):
        # s_t - s_(t-1) + discharge/eta - eta*charge = 0.
        plan_terms = [
            (da_soc.start + step, 1.0),
            (dd.start + step, 1.0 / eta),
            (dc.start + step, -eta),
        ]
        actual_terms = [
            (rt_soc.start + step, 1.0),
            (rd.start + step, 1.0 / eta),
            (rc.start + step, -eta),
        ]
        if step == 0:
            add_constraint(plan_terms, initial_plan, initial_plan)
            add_constraint(actual_terms, initial_actual, initial_actual)
        else:
            plan_terms.append((da_soc.start + step - 1, -1.0))
            actual_terms.append((rt_soc.start + step - 1, -1.0))
            add_constraint(plan_terms, 0.0, 0.0)
            add_constraint(actual_terms, 0.0, 0.0)

        # One direction per hour for each trajectory.
        add_constraint(
            [
                (dd.start + step, 1.0),
                (da_mode.start + step, -power_energy),
            ],
            upper=0.0,
        )
        add_constraint(
            [
                (dc.start + step, 1.0),
                (da_mode.start + step, power_energy),
            ],
            upper=power_energy,
        )
        add_constraint(
            [
                (rd.start + step, 1.0),
                (rt_mode.start + step, -power_energy),
            ],
            upper=0.0,
        )
        add_constraint(
            [
                (rc.start + step, 1.0),
                (rt_mode.start + step, power_energy),
            ],
            upper=power_energy,
        )

        # excess >= |u_RT-u_DA| - 0.03*|u_DA|.
        add_constraint(
            [
                (dd.start + step, 1.03),
                (dc.start + step, -0.97),
                (rd.start + step, -1.0),
                (rc.start + step, 1.0),
                (excess_var.start + step, 1.0),
            ],
            lower=0.0,
        )
        add_constraint(
            [
                (dd.start + step, -0.97),
                (dc.start + step, 1.03),
                (rd.start + step, 1.0),
                (rc.start + step, -1.0),
                (excess_var.start + step, 1.0),
            ],
            lower=0.0,
        )

    # Throughput is a per-delivery-day contract, not a segment-wide cap.
    for day in range(days):
        start = day * hours_per_day
        stop = start + hours_per_day
        add_constraint(
            [(dd.start + step, 1.0) for step in range(start, stop)],
            upper=cycle_limit,
        )
        add_constraint(
            [(rd.start + step, 1.0) for step in range(start, stop)],
            upper=cycle_limit,
        )

    constraint_matrix = coo_matrix(
        (coefficients, (row_ids, col_ids)),
        shape=(len(row_lower), variable_count),
        dtype=np.float64,
    ).tocsc()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            constraint_matrix,
            np.asarray(row_lower, dtype=np.float64),
            np.asarray(row_upper, dtype=np.float64),
        ),
        options={"mip_rel_gap": float(mip_rel_gap)},
    )
    status_code = int(getattr(result, "status", 4))
    certified_optimal = bool(getattr(result, "success", False)) and status_code == 0
    result_fun = getattr(result, "fun", None)
    result_gap = getattr(result, "mip_gap", None)
    result_nodes = getattr(result, "mip_node_count", None)
    status = MILPSolveStatus(
        sample_index=int(segment_index),
        leg="joint_da_rt_with_deviation_penalty_continuous_segment",
        success=certified_optimal,
        status_code=status_code,
        message=str(getattr(result, "message", "MILP solver returned no message")),
        objective=(
            float(result_fun) if certified_optimal and result_fun is not None else None
        ),
        mip_gap=(float(result_gap) if result_gap is not None else None),
        mip_node_count=(int(result_nodes) if result_nodes is not None else None),
    )
    if not certified_optimal:
        raise MILPOracleSolveError(status)

    solution = np.asarray(result.x, dtype=np.float64)

    def clean_energy(values):
        cleaned = np.clip(np.asarray(values, dtype=np.float64), 0.0, power_energy)
        cleaned[np.abs(cleaned) < 1e-9] = 0.0
        return cleaned

    da_discharge = clean_energy(solution[dd])
    da_charge = clean_energy(solution[dc])
    rt_discharge = clean_energy(solution[rd])
    rt_charge = clean_energy(solution[rc])
    da_net = da_discharge - da_charge
    rt_net = rt_discharge - rt_charge
    deviation = rt_net - da_net
    tolerance = 0.03 * (da_discharge + da_charge)
    excess = np.maximum(np.abs(deviation) - tolerance, 0.0)
    degradation = kappa * (rt_discharge + rt_charge)
    penalty = 2.0 * np.abs(rt) * excess
    da_leg = da * da_net
    rt_leg = rt * deviation
    hourly_revenue = da_leg + rt_leg - degradation - penalty

    shape = (days, hours_per_day)
    daily_revenue = hourly_revenue.reshape(shape).sum(axis=1)
    da_soc_values = np.asarray(solution[da_soc], dtype=np.float64)
    rt_soc_values = np.asarray(solution[rt_soc], dtype=np.float64)
    dispatch = ContinuousDualMILPDispatch(
        da_net_mwh=da_net.reshape(shape),
        rt_net_mwh=rt_net.reshape(shape),
        da_soc_path_mwh=np.concatenate(([initial_plan], da_soc_values)),
        rt_soc_path_mwh=np.concatenate(([initial_actual], rt_soc_values)),
        deviation_mwh=deviation.reshape(shape),
        tolerance_mwh=tolerance.reshape(shape),
        excess_mwh=excess.reshape(shape),
        hourly_revenue=hourly_revenue.reshape(shape),
        daily_revenue=daily_revenue,
        daily_da_leg=da_leg.reshape(shape).sum(axis=1),
        daily_rt_deviation_leg=rt_leg.reshape(shape).sum(axis=1),
        daily_degradation_cost=degradation.reshape(shape).sum(axis=1),
        daily_deviation_penalty=penalty.reshape(shape).sum(axis=1),
    )
    return ContinuousDualMILPOracleResult(
        revenue=float(np.sum(daily_revenue)),
        status=status,
        dispatch=dispatch,
    )
