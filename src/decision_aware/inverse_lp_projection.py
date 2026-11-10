"""Inverse-LP projection that preserves a frozen v4 day-ahead decision.

For a fixed RT-at-DA price vector, the joint storage planning problem is a
linear program whose DA prices only parameterize the DA discharge/charge
objective coefficients.  This module projects a causal DA anchor onto the
normal cone of the active v4 solution.  The projected DA is therefore the
same forecast that is scored and passed to the optimizer, while the v4 plan
remains optimal by construction through the KKT stationarity conditions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import BESSSimulator


@dataclass(frozen=True)
class JointLP:
    objective: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray
    bounds: tuple[tuple[float, float], ...]
    horizon: int


@dataclass(frozen=True)
class InverseProjectionResult:
    projected_da: np.ndarray
    target_solution: np.ndarray
    projected_solution: np.ndarray
    target_objective_gap: float
    max_stationarity_residual: float
    active_constraint_count: int
    max_plan_difference_mwh: float


def solve_lexicographic_nearest_solution(
    problem: JointLP,
    reference_solution: np.ndarray,
) -> np.ndarray:
    """Select the primary-optimal solution nearest to a reference in L1.

    The primary objective is imposed as an equality at the reference value.
    Callers must first verify that the reference is primary-optimal.  This
    second LP makes the tie-break explicit instead of relying on HiGHS' choice
    among degenerate primary optima.
    """
    from scipy.optimize import linprog

    reference = np.asarray(reference_solution, dtype=np.float64)
    count = len(problem.objective)
    if reference.shape != (count,) or not np.isfinite(reference).all():
        raise ValueError("reference solution must match the LP variable vector")
    variable_count = 2 * count
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[count:] = 1.0

    a_ub = np.zeros((len(problem.a_ub) + 2 * count, variable_count), dtype=np.float64)
    b_ub = np.zeros(len(a_ub), dtype=np.float64)
    a_ub[: len(problem.a_ub), :count] = problem.a_ub
    b_ub[: len(problem.a_ub)] = problem.b_ub
    for index in range(count):
        row = len(problem.a_ub) + 2 * index
        a_ub[row, index] = 1.0
        a_ub[row, count + index] = -1.0
        b_ub[row] = reference[index]
        a_ub[row + 1, index] = -1.0
        a_ub[row + 1, count + index] = -1.0
        b_ub[row + 1] = -reference[index]

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.concatenate(
            (problem.objective, np.zeros(count, dtype=np.float64))
        )[None, :],
        b_eq=np.asarray([problem.objective @ reference], dtype=np.float64),
        bounds=list(problem.bounds) + [(0.0, None)] * count,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"lexicographic tie-break LP failed: {result.message}")
    selected = np.asarray(result.x[:count], dtype=np.float64)
    # HiGHS returns a solution within feasibility tolerance.  Once the
    # secondary LP has independently proved zero L1 distance, canonicalize the
    # tiny numerical residue so settlement does not depend on solver rounding.
    if np.max(np.abs(selected - reference)) <= 1e-8:
        return reference.copy()
    return selected


def build_joint_lp(
    da_price: np.ndarray,
    rt_price: np.ndarray,
    simulator: BESSSimulator,
    *,
    initial_plan_soc_mwh: float,
    initial_actual_soc_mwh: float | None = None,
) -> JointLP:
    """Build the exact LP used by ``solve_predicted_da_penalty_lp_day``."""
    da = np.asarray(da_price, dtype=np.float64)
    rt = np.asarray(rt_price, dtype=np.float64)
    if da.ndim != 1 or da.shape != rt.shape or not np.isfinite(da).all() or not np.isfinite(rt).all():
        raise ValueError("DA与RT-at-DA必须是一维同形有限数组")
    horizon = len(da)
    power = float(simulator.P * simulator.dt)
    eta = float(simulator.eta)
    soc_min = float(simulator.s_min)
    soc_max = float(simulator.s_max)
    cycle_limit = float(simulator.e_cyc)
    kappa = float(simulator.kappa)
    plan_soc0 = float(np.clip(initial_plan_soc_mwh, soc_min, soc_max))
    actual_soc0 = float(
        plan_soc0 if initial_actual_soc_mwh is None else initial_actual_soc_mwh
    )
    actual_soc0 = float(np.clip(actual_soc0, soc_min, soc_max))
    variable_count = 5 * horizon

    def trajectory_rows(d_start: int, c_start: int, initial_soc: float):
        rows: list[np.ndarray] = []
        upper: list[float] = []
        for step in range(horizon):
            row = np.zeros(variable_count, dtype=np.float64)
            row[d_start:d_start + step + 1] = -1.0 / eta
            row[c_start:c_start + step + 1] = eta
            rows.extend((row, -row))
            upper.extend((soc_max - initial_soc, initial_soc - soc_min))
        row = np.zeros(variable_count, dtype=np.float64)
        row[d_start:d_start + horizon] = 1.0
        rows.append(row)
        upper.append(cycle_limit)
        return rows, upper

    dd = slice(0, horizon)
    dc = slice(horizon, 2 * horizon)
    rd = slice(2 * horizon, 3 * horizon)
    rc = slice(3 * horizon, 4 * horizon)
    excess = slice(4 * horizon, 5 * horizon)
    spread = da - rt
    direction_discharge = spread > 1e-8
    direction_charge = spread < -1e-8

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[dd] = -spread
    objective[dc] = spread
    objective[rd] = -rt + kappa
    objective[rc] = rt + kappa
    objective[excess] = 2.0 * np.abs(rt)

    rows, upper = trajectory_rows(dd.start, dc.start, plan_soc0)
    actual_rows, actual_upper = trajectory_rows(rd.start, rc.start, actual_soc0)
    rows.extend(actual_rows)
    upper.extend(actual_upper)
    for step in range(horizon):
        row = np.zeros(variable_count, dtype=np.float64)
        for column, coefficient in (
            (dd.start + step, 1.03),
            (dc.start + step, -0.97),
            (rd.start + step, -1.0),
            (rc.start + step, 1.0),
            (excess.start + step, 1.0),
        ):
            row[column] = -coefficient
        rows.append(row)
        upper.append(0.0)
        row = np.zeros(variable_count, dtype=np.float64)
        for column, coefficient in (
            (dd.start + step, -0.97),
            (dc.start + step, 1.03),
            (rd.start + step, 1.0),
            (rc.start + step, -1.0),
            (excess.start + step, 1.0),
        ):
            row[column] = -coefficient
        rows.append(row)
        upper.append(0.0)

    bounds: list[tuple[float, float]] = []
    bounds.extend((0.0, power if allowed else 0.0) for allowed in direction_discharge)
    bounds.extend((0.0, power if allowed else 0.0) for allowed in direction_charge)
    bounds.extend([(0.0, power)] * (2 * horizon))
    bounds.extend([(0.0, 2.0 * power)] * horizon)
    return JointLP(
        objective=objective,
        a_ub=np.asarray(rows, dtype=np.float64),
        b_ub=np.asarray(upper, dtype=np.float64),
        bounds=tuple(bounds),
        horizon=horizon,
    )


def solve_joint_lp(problem: JointLP) -> np.ndarray:
    from scipy.optimize import linprog

    result = linprog(
        problem.objective,
        A_ub=problem.a_ub,
        b_ub=problem.b_ub,
        bounds=problem.bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"joint LP failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _active_normals(
    problem: JointLP,
    solution: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    normals: list[np.ndarray] = []
    slack = problem.b_ub - problem.a_ub @ solution
    normals.extend(problem.a_ub[slack <= tolerance])
    count = len(solution)
    for column, (lower, upper) in enumerate(problem.bounds):
        if solution[column] - lower <= tolerance:
            row = np.zeros(count, dtype=np.float64)
            row[column] = -1.0
            normals.append(row)
        if upper - solution[column] <= tolerance:
            row = np.zeros(count, dtype=np.float64)
            row[column] = 1.0
            normals.append(row)
    if not normals:
        raise ValueError("target solution has no active constraints")
    return np.asarray(normals, dtype=np.float64)


def project_da_to_target_normal_cone(
    anchor_da: np.ndarray,
    v4_da: np.ndarray,
    rt_price: np.ndarray,
    simulator: BESSSimulator,
    *,
    initial_plan_soc_mwh: float,
    initial_actual_soc_mwh: float | None = None,
    active_tolerance: float = 1e-7,
    direction_margin: float = 1e-6,
) -> InverseProjectionResult:
    """Find the closest L1 DA forecast that keeps the v4 LP solution optimal."""
    from scipy.optimize import linprog

    anchor = np.asarray(anchor_da, dtype=np.float64)
    v4 = np.asarray(v4_da, dtype=np.float64)
    rt = np.asarray(rt_price, dtype=np.float64)
    if anchor.shape != v4.shape or v4.shape != rt.shape or anchor.ndim != 1:
        raise ValueError("anchor、v4 DA与RT-at-DA必须一维同形")
    original = build_joint_lp(
        v4,
        rt,
        simulator,
        initial_plan_soc_mwh=initial_plan_soc_mwh,
        initial_actual_soc_mwh=initial_actual_soc_mwh,
    )
    target = solve_joint_lp(original)
    normals = _active_normals(original, target, tolerance=active_tolerance)
    horizon = original.horizon
    dual_count = normals.shape[0]
    forward_count = len(target)
    # inverse variables: projected DA, nonnegative active-set multipliers, L1 slacks
    da_slice = slice(0, horizon)
    dual_slice = slice(horizon, horizon + dual_count)
    slack_slice = slice(horizon + dual_count, horizon + dual_count + horizon)
    inverse_count = horizon + dual_count + horizon

    c = np.zeros(inverse_count, dtype=np.float64)
    c[slack_slice] = 1.0
    c[dual_slice] = 1e-10

    a_eq = np.zeros((forward_count, inverse_count), dtype=np.float64)
    b_eq = np.zeros(forward_count, dtype=np.float64)
    # KKT stationarity c(price) + G^T lambda = 0.
    a_eq[:, dual_slice] = normals.T
    for step in range(horizon):
        a_eq[step, da_slice.start + step] = -1.0
        b_eq[step] = -rt[step]
        a_eq[horizon + step, da_slice.start + step] = 1.0
        b_eq[horizon + step] = rt[step]
    b_eq[2 * horizon:] = -original.objective[2 * horizon:]

    rows: list[np.ndarray] = []
    upper: list[float] = []
    for step in range(horizon):
        # |projected - anchor| <= slack
        row = np.zeros(inverse_count, dtype=np.float64)
        row[da_slice.start + step] = 1.0
        row[slack_slice.start + step] = -1.0
        rows.append(row)
        upper.append(anchor[step])
        row = np.zeros(inverse_count, dtype=np.float64)
        row[da_slice.start + step] = -1.0
        row[slack_slice.start + step] = -1.0
        rows.append(row)
        upper.append(-anchor[step])

        spread = v4[step] - rt[step]
        row = np.zeros(inverse_count, dtype=np.float64)
        if spread > 1e-8:
            row[da_slice.start + step] = -1.0
            rows.append(row)
            upper.append(-(rt[step] + direction_margin))
        elif spread < -1e-8:
            row[da_slice.start + step] = 1.0
            rows.append(row)
            upper.append(rt[step] - direction_margin)
        else:
            row[da_slice.start + step] = 1.0
            rows.append(row)
            upper.append(rt[step] + 0.5 * direction_margin)
            rows.append(-row)
            upper.append(-rt[step] + 0.5 * direction_margin)

    bounds = [(-5000.0, 5000.0)] * horizon
    bounds += [(0.0, None)] * dual_count
    bounds += [(0.0, None)] * horizon
    inverse = linprog(
        c,
        A_ub=np.asarray(rows, dtype=np.float64),
        b_ub=np.asarray(upper, dtype=np.float64),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not inverse.success:
        raise RuntimeError(f"inverse LP failed: {inverse.message}")
    projected_da = np.asarray(inverse.x[da_slice], dtype=np.float64)
    projected_problem = build_joint_lp(
        projected_da,
        rt,
        simulator,
        initial_plan_soc_mwh=initial_plan_soc_mwh,
        initial_actual_soc_mwh=initial_actual_soc_mwh,
    )
    projected_solution = solve_joint_lp(projected_problem)
    target_value = float(projected_problem.objective @ target)
    solved_value = float(projected_problem.objective @ projected_solution)
    stationarity = a_eq @ inverse.x - b_eq
    target_plan = target[:horizon] - target[horizon:2 * horizon]
    projected_plan = (
        projected_solution[:horizon]
        - projected_solution[horizon:2 * horizon]
    )
    return InverseProjectionResult(
        projected_da=projected_da,
        target_solution=target,
        projected_solution=projected_solution,
        target_objective_gap=float(target_value - solved_value),
        max_stationarity_residual=float(np.max(np.abs(stationarity))),
        active_constraint_count=int(dual_count),
        max_plan_difference_mwh=float(np.max(np.abs(target_plan - projected_plan))),
    )
