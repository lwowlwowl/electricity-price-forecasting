from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.continuous_zo import (  # noqa: E402
    ContinuousMPCObjective,
    continuous_mpc_proxy_loss,
    estimate_continuous_mpc_gradients,
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


def test_continuous_objective_returns_one_realized_revenue_per_day():
    timestamps = tuple(pd.date_range(
        "2023-01-03 00:00", periods=24, freq="h", tz="America/Chicago"
    ))
    objective = ContinuousMPCObjective(
        realized_da=torch.linspace(-30.0, 120.0, 24).reshape(1, 24),
        realized_rt=torch.linspace(-10.0, 90.0, 24).reshape(1, 24),
        local_timestamps=(timestamps,),
        cfg=_cfg(),
    )
    da = torch.cat((torch.full((12,), -20.0), torch.full((12,), 100.0))).reshape(1, 24)
    rt_da = torch.zeros(1, 24)
    rt = torch.tensor([10.0, 15.0, 20.0, 25.0]).repeat(1, 24, 1)
    revenue = objective(da, rt_da, rt)
    assert revenue.shape == (1,)
    assert torch.isfinite(revenue).all()


class _SmoothRevenue:
    """只用于快速验证零阶接口与三模型梯度符号。"""

    def __call__(self, da, rt_at_da, rt):
        spread_target = torch.full_like(da, 3.0)
        rt_target = torch.full_like(rt, 2.0)
        return -(
            (da - rt_at_da - spread_target).square().flatten(1).sum(1)
            + (rt - rt_target).square().flatten(1).sum(1)
        )


def test_coupled_zo_signal_reaches_three_independent_outputs():
    torch.manual_seed(3)
    da = torch.full((1, 24), 8.0, requires_grad=True)
    rt_da = torch.full((1, 24), 1.0, requires_grad=True)
    rt = torch.full((1, 24, 4), 6.0, requires_grad=True)
    grad_spread, grad_rt, diagnostics = estimate_continuous_mpc_gradients(
        da,
        rt_da,
        rt,
        _SmoothRevenue(),
        epsilon_spread=0.5,
        epsilon_rt=0.5,
        directions_spread=2,
        directions_rt=2,
        direction_mode="orthogonal",
    )
    assert diagnostics["spread"]["finite"]
    assert diagnostics["rt"]["finite"]
    assert diagnostics["spread"]["gradient_norm"] > 0.0
    assert diagnostics["rt"]["gradient_norm"] > 0.0

    loss = continuous_mpc_proxy_loss(
        da, rt_da, rt, grad_spread, grad_rt, reduction="global_sum"
    )
    loss.backward()
    assert torch.allclose(da.grad, grad_spread)
    assert torch.allclose(rt_da.grad, -grad_spread)
    assert torch.allclose(rt.grad, grad_rt)
    assert all(value.grad is not None for value in (da, rt_da, rt))
