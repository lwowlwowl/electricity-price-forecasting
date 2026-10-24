import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.implicit_battery_qp import (  # noqa: E402
    BatteryQPConfig,
    ImplicitBatteryQPLayer,
    predicted_cost_vector,
    realized_two_settlement_revenue,
)


def _small_config() -> BatteryQPConfig:
    return BatteryQPConfig(
        horizon=4,
        daily_discharge_limit_mwh=2.0,
        newton_steps=80,
        convergence_tolerance=1e-6,
    )


def test_solution_is_strictly_feasible_and_economically_directional():
    layer = ImplicitBatteryQPLayer(_small_config())
    da = torch.tensor([[10.0, 80.0, 15.0, 90.0]])
    rt_da = torch.tensor([[30.0, 20.0, 40.0, 25.0]])
    rt = torch.tensor([[25.0, 30.0, 35.0, 40.0]])
    solution = layer(predicted_cost_vector(
        da, rt_da, rt, degradation_cost=5.7
    ))
    slack = layer.bounds - layer.constraints @ solution.detach().numpy()[0]
    assert float(slack.min()) > 0.0
    actions = layer.split(solution)
    plan = actions["da_discharge"] - actions["da_charge"]
    assert plan[0, 1] > plan[0, 0]
    assert plan[0, 3] > plan[0, 2]
    diagnostic = layer.last_diagnostics[0]
    assert diagnostic["relative_gradient_inf"] <= max(
        layer.config.relative_failure_tolerance,
        layer.config.absolute_failure_tolerance,
    )


def test_solver_still_rejects_material_absolute_and_relative_residual():
    config = BatteryQPConfig(
        horizon=4,
        daily_discharge_limit_mwh=2.0,
        newton_steps=1,
        convergence_tolerance=0.0,
        absolute_failure_tolerance=0.0,
        relative_failure_tolerance=0.0,
    )
    layer = ImplicitBatteryQPLayer(config)
    cost = torch.linspace(-100.0, 100.0, config.variable_count).reshape(1, -1)
    with pytest.raises(RuntimeError, match="relative_gradient_inf"):
        layer(cost)


def test_implicit_gradient_matches_central_difference():
    torch.manual_seed(3)
    layer = ImplicitBatteryQPLayer(_small_config())
    cost = torch.linspace(-3.0, 4.0, layer.config.variable_count).reshape(1, -1)
    cost.requires_grad_(True)
    weights = torch.randn_like(cost)
    value = (layer(cost) * weights).sum()
    value.backward()
    analytic = float(cost.grad[0, 2])
    epsilon = 1e-3
    with torch.no_grad():
        plus = cost.detach().clone()
        minus = cost.detach().clone()
        plus[0, 2] += epsilon
        minus[0, 2] -= epsilon
        numeric = float(
            ((layer(plus) * weights).sum() - (layer(minus) * weights).sum())
            / (2.0 * epsilon)
        )
    np.testing.assert_allclose(analytic, numeric, rtol=2e-2, atol=2e-3)


def test_realized_revenue_backpropagates_to_all_three_forecasts():
    layer = ImplicitBatteryQPLayer(_small_config())
    da = torch.tensor([[20.0, 60.0, 15.0, 70.0]], requires_grad=True)
    rt_da = torch.tensor([[40.0, 25.0, 45.0, 20.0]], requires_grad=True)
    rt = torch.tensor([[35.0, 30.0, 50.0, 25.0]], requires_grad=True)
    solution = layer(predicted_cost_vector(
        da, rt_da, rt, degradation_cost=5.7
    ))
    revenue, parts = realized_two_settlement_revenue(
        layer.split(solution),
        torch.tensor([[22.0, 75.0, 10.0, 85.0]]),
        torch.tensor([[38.0, 20.0, 55.0, 18.0]]),
        degradation_cost=5.7,
    )
    (-revenue.mean()).backward()
    for prediction in (da, rt_da, rt):
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()
        assert torch.linalg.vector_norm(prediction.grad) > 0
    assert set(parts) == {
        "da_component", "rt_component", "degradation", "deviation_penalty",
        "excess", "plan", "actual",
    }

