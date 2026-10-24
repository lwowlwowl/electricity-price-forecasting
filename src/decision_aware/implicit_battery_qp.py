"""可微双结算电池层：强凸平滑 QP + KKT 隐式微分。

前向问题保留 DA 计划、RT 实际动作、SOC、日放电额度、3% 偏差容忍带和
偏差罚金。与正式 LP 的区别只有两点：去掉依赖预测符号的可行域切换，并加入
很小的二次正则与对数障碍，使最优解唯一且处处可微。反向不展开求解迭代，
而是在收敛点解平滑 KKT 系统。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class BatteryQPConfig:
    horizon: int = 24
    power_mwh: float = 1.0
    energy_mwh: float = 4.0
    soc_min_fraction: float = 0.1
    soc_max_fraction: float = 0.9
    initial_soc_fraction: float = 0.5
    efficiency: float = 0.95
    daily_discharge_limit_mwh: float = 4.0
    deviation_tolerance: float = 0.03
    quadratic_regularization: float = 0.25
    barrier_temperature: float = 0.25
    newton_steps: int = 60
    convergence_tolerance: float = 1e-7
    absolute_failure_tolerance: float = 5e-5
    relative_failure_tolerance: float = 1e-6

    @property
    def variable_count(self) -> int:
        return 5 * self.horizon


def _slices(horizon: int) -> tuple[slice, slice, slice, slice, slice]:
    return (
        slice(0, horizon),
        slice(horizon, 2 * horizon),
        slice(2 * horizon, 3 * horizon),
        slice(3 * horizon, 4 * horizon),
        slice(4 * horizon, 5 * horizon),
    )


def build_constraint_system(
    config: BatteryQPConfig,
    *,
    initial_plan_soc_mwh: float | None = None,
    initial_actual_soc_mwh: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``C, d`` for the fixed feasible set ``C x <= d``."""
    h = config.horizon
    n = config.variable_count
    dd, dc, rd, rc, excess = _slices(h)
    eta = config.efficiency
    soc_min = config.energy_mwh * config.soc_min_fraction
    soc_max = config.energy_mwh * config.soc_max_fraction
    default_soc = config.energy_mwh * config.initial_soc_fraction
    plan_soc0 = default_soc if initial_plan_soc_mwh is None else initial_plan_soc_mwh
    actual_soc0 = default_soc if initial_actual_soc_mwh is None else initial_actual_soc_mwh
    rows: list[np.ndarray] = []
    bounds: list[float] = []

    def add(terms: list[tuple[int, float]], bound: float) -> None:
        row = np.zeros(n, dtype=np.float64)
        for column, coefficient in terms:
            row[column] += coefficient
        rows.append(row)
        bounds.append(float(bound))

    for step in range(h):
        plan_terms: list[tuple[int, float]] = []
        actual_terms: list[tuple[int, float]] = []
        for index in range(step + 1):
            plan_terms.extend(
                ((dd.start + index, -1.0 / eta), (dc.start + index, eta))
            )
            actual_terms.extend(
                ((rd.start + index, -1.0 / eta), (rc.start + index, eta))
            )
        add(plan_terms, soc_max - plan_soc0)
        add([(i, -v) for i, v in plan_terms], plan_soc0 - soc_min)
        add(actual_terms, soc_max - actual_soc0)
        add([(i, -v) for i, v in actual_terms], actual_soc0 - soc_min)

    add([(dd.start + index, 1.0) for index in range(h)], config.daily_discharge_limit_mwh)
    add([(rd.start + index, 1.0) for index in range(h)], config.daily_discharge_limit_mwh)

    tolerance = config.deviation_tolerance
    for step in range(h):
        # excess >= +(u_rt-u_da) - tolerance * (da_discharge + da_charge)
        add(
            [
                (rd.start + step, 1.0),
                (rc.start + step, -1.0),
                (dd.start + step, -(1.0 + tolerance)),
                (dc.start + step, 1.0 - tolerance),
                (excess.start + step, -1.0),
            ],
            0.0,
        )
        # excess >= -(u_rt-u_da) - tolerance * (da_discharge + da_charge)
        add(
            [
                (rd.start + step, -1.0),
                (rc.start + step, 1.0),
                (dd.start + step, 1.0 - tolerance),
                (dc.start + step, -(1.0 + tolerance)),
                (excess.start + step, -1.0),
            ],
            0.0,
        )

    upper = np.concatenate(
        (
            np.full(4 * h, config.power_mwh, dtype=np.float64),
            np.full(h, 2.0 * config.power_mwh, dtype=np.float64),
        )
    )
    for column in range(n):
        add([(column, -1.0)], 0.0)
        add([(column, 1.0)], upper[column])
    return np.asarray(rows), np.asarray(bounds)


def _strictly_feasible_start(config: BatteryQPConfig) -> np.ndarray:
    h = config.horizon
    dd, dc, rd, rc, excess = _slices(h)
    x = np.full(config.variable_count, 0.005, dtype=np.float64)
    x[excess] = 0.02
    # Equal charge/discharge is slightly SOC-negative because eta < 1.  Make
    # charge marginally larger so the starting trajectory remains centered.
    x[dc] = x[dd] / (config.efficiency * config.efficiency)
    x[rc] = x[rd] / (config.efficiency * config.efficiency)
    return x


def _barrier_value(
    x: np.ndarray,
    cost: np.ndarray,
    constraints: np.ndarray,
    bounds: np.ndarray,
    quadratic: float,
    temperature: float,
) -> float:
    slack = bounds - constraints @ x
    if np.any(slack <= 0.0):
        return float("inf")
    return float(
        cost @ x
        + 0.5 * quadratic * (x @ x)
        - temperature * np.log(slack).sum()
    )


def _solve_one(
    cost: np.ndarray,
    constraints: np.ndarray,
    bounds: np.ndarray,
    config: BatteryQPConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = _strictly_feasible_start(config)
    initial_slack = bounds - constraints @ x
    if np.min(initial_slack) <= 0.0:
        raise RuntimeError("implicit battery QP failed to construct a feasible start")
    identity = np.eye(config.variable_count, dtype=np.float64)
    converged = False
    gradient_norm = float("inf")
    iterations = 0
    for iterations in range(1, config.newton_steps + 1):
        slack = bounds - constraints @ x
        inverse_slack = 1.0 / slack
        gradient = (
            cost
            + config.quadratic_regularization * x
            + config.barrier_temperature * constraints.T @ inverse_slack
        )
        weighted = constraints * inverse_slack[:, None]
        hessian = (
            config.quadratic_regularization * identity
            + config.barrier_temperature * weighted.T @ weighted
        )
        gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if gradient_norm <= config.convergence_tolerance:
            converged = True
            break
        step = -np.linalg.solve(hessian, gradient)
        c_step = constraints @ step
        positive = c_step > 0.0
        max_step = 1.0
        if np.any(positive):
            max_step = min(
                max_step,
                0.99 * float(np.min(slack[positive] / c_step[positive])),
            )
        value = _barrier_value(
            x, cost, constraints, bounds,
            config.quadratic_regularization, config.barrier_temperature,
        )
        directional = float(gradient @ step)
        step_size = max_step
        accepted = False
        for _ in range(40):
            candidate = x + step_size * step
            candidate_value = _barrier_value(
                candidate, cost, constraints, bounds,
                config.quadratic_regularization, config.barrier_temperature,
            )
            if candidate_value <= value + 1e-4 * step_size * directional:
                x = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break

    slack = bounds - constraints @ x
    inverse_slack = 1.0 / slack
    weighted = constraints * inverse_slack[:, None]
    hessian = (
        config.quadratic_regularization * identity
        + config.barrier_temperature * weighted.T @ weighted
    )
    cost_scale = max(1.0, float(np.linalg.norm(cost, ord=np.inf)))
    relative_gradient_norm = gradient_norm / cost_scale
    if (
        not converged
        and gradient_norm > config.absolute_failure_tolerance
        and relative_gradient_norm > config.relative_failure_tolerance
    ):
        raise RuntimeError(
            "implicit battery QP Newton solve did not converge: "
            f"iterations={iterations}, gradient_inf={gradient_norm:.3e}, "
            f"relative_gradient_inf={relative_gradient_norm:.3e}, "
            f"minimum_slack={float(np.min(slack)):.3e}"
        )
    return x, hessian, {
        "iterations": float(iterations),
        "gradient_inf": gradient_norm,
        "relative_gradient_inf": relative_gradient_norm,
        "minimum_slack": float(np.min(slack)),
    }


class _ImplicitBatteryQPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cost: torch.Tensor, layer: "ImplicitBatteryQPLayer"):
        if cost.ndim != 2 or cost.shape[1] != layer.config.variable_count:
            raise ValueError("cost must have shape [batch, 5 * horizon]")
        cost_numpy = cost.detach().double().cpu().numpy()
        solutions = []
        hessians = []
        diagnostics = []
        for row in cost_numpy:
            solution, hessian, diagnostic = _solve_one(
                row, layer.constraints, layer.bounds, layer.config
            )
            solutions.append(solution)
            hessians.append(hessian)
            diagnostics.append(diagnostic)
        layer.last_diagnostics = tuple(diagnostics)
        ctx.hessians = tuple(hessians)
        ctx.input_dtype = cost.dtype
        ctx.input_device = cost.device
        return torch.as_tensor(
            np.stack(solutions), dtype=cost.dtype, device=cost.device
        )

    @staticmethod
    def backward(ctx, gradient_output: torch.Tensor):
        gradients = []
        output_numpy = gradient_output.detach().double().cpu().numpy()
        for hessian, row in zip(ctx.hessians, output_numpy):
            # F(x,c)=0, dF/dx=H and dF/dc=I, hence dx/dc=-H^{-1}.
            gradients.append(-np.linalg.solve(hessian.T, row))
        gradient_cost = torch.as_tensor(
            np.stack(gradients),
            dtype=ctx.input_dtype,
            device=ctx.input_device,
        )
        return gradient_cost, None


class ImplicitBatteryQPLayer(nn.Module):
    """Differentiable daily two-settlement storage optimization layer."""

    def __init__(self, config: BatteryQPConfig | None = None) -> None:
        super().__init__()
        self.config = config or BatteryQPConfig()
        self.constraints, self.bounds = build_constraint_system(self.config)
        self.last_diagnostics: tuple[dict[str, float], ...] = ()

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        return _ImplicitBatteryQPFunction.apply(cost, self)

    def split(self, solution: torch.Tensor) -> dict[str, torch.Tensor]:
        dd, dc, rd, rc, excess = _slices(self.config.horizon)
        return {
            "da_discharge": solution[:, dd],
            "da_charge": solution[:, dc],
            "rt_discharge": solution[:, rd],
            "rt_charge": solution[:, rc],
            "optimizer_excess": solution[:, excess],
        }


def predicted_cost_vector(
    predicted_da: torch.Tensor,
    predicted_rt_at_da: torch.Tensor,
    predicted_rt_executed: torch.Tensor,
    *,
    degradation_cost: float,
    absolute_price_smoothing: float = 1.0,
) -> torch.Tensor:
    if not (
        predicted_da.shape == predicted_rt_at_da.shape == predicted_rt_executed.shape
        and predicted_da.ndim == 2
    ):
        raise ValueError("three price tensors must share shape [batch, horizon]")
    spread = predicted_da - predicted_rt_at_da
    smooth_abs_rt = torch.sqrt(
        predicted_rt_executed.square() + absolute_price_smoothing**2
    )
    return torch.cat(
        (
            -spread,
            spread,
            -predicted_rt_executed + degradation_cost,
            predicted_rt_executed + degradation_cost,
            2.0 * smooth_abs_rt,
        ),
        dim=1,
    )


def realized_two_settlement_revenue(
    actions: dict[str, torch.Tensor],
    realized_da: torch.Tensor,
    realized_rt: torch.Tensor,
    *,
    degradation_cost: float,
    deviation_tolerance: float = 0.03,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    plan = actions["da_discharge"] - actions["da_charge"]
    actual = actions["rt_discharge"] - actions["rt_charge"]
    deviation = actual - plan
    excess = torch.relu(
        deviation.abs() - deviation_tolerance * plan.abs()
    )
    da_component = (realized_da * plan).sum(dim=1)
    rt_component = (realized_rt * deviation).sum(dim=1)
    degradation = degradation_cost * (
        actions["rt_discharge"] + actions["rt_charge"]
    ).sum(dim=1)
    penalty = (2.0 * realized_rt.abs() * excess).sum(dim=1)
    revenue = da_component + rt_component - degradation - penalty
    return revenue, {
        "da_component": da_component,
        "rt_component": rt_component,
        "degradation": degradation,
        "deviation_penalty": penalty,
        "excess": excess,
        "plan": plan,
        "actual": actual,
    }

