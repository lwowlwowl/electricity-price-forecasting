"""连续DA/RT MPC的黑盒零阶收益梯度。

SciPy LP本身不在PyTorch计算图中。本模块用成对扰动比较真实结算收益，估计
预测输出的方向梯度，再通过一个线性代理损失把该方向传回三个独立模型。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .continuous_mpc import locked_continuous_mpc_backtest
from .joint_system import estimate_system_zo_gradient, joint_proxy_loss


def _as_day_hour(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != 24:
        raise ValueError(f"{name}必须是[days,24]，实际{tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}包含非有限值")
    return value


@dataclass(frozen=True)
class ContinuousMPCObjective:
    """把预测张量映射为逐日真实净收益的只读黑盒。"""

    realized_da: torch.Tensor
    realized_rt: torch.Tensor
    local_timestamps: tuple[tuple[pd.Timestamp, ...], ...]
    cfg: object
    terminal_value_weight: float = 1.0

    def __post_init__(self) -> None:
        da = _as_day_hour(self.realized_da, "realized_da")
        rt = _as_day_hour(self.realized_rt, "realized_rt")
        if da.shape != rt.shape:
            raise ValueError("realized_da与realized_rt必须同形")
        if len(self.local_timestamps) != da.shape[0]:
            raise ValueError("local_timestamps天数与真值不一致")
        for values in self.local_timestamps:
            if len(values) != 24 or any(pd.Timestamp(item).tzinfo is None for item in values):
                raise ValueError("每天必须提供24个带时区时刻")

    @torch.no_grad()
    def __call__(
        self,
        predicted_da: torch.Tensor,
        predicted_rt_at_da: torch.Tensor,
        predicted_rt: torch.Tensor,
    ) -> torch.Tensor:
        da = _as_day_hour(predicted_da.detach().float().cpu(), "predicted_da")
        rt_da = _as_day_hour(
            predicted_rt_at_da.detach().float().cpu(), "predicted_rt_at_da"
        )
        rt = predicted_rt.detach().float().cpu()
        if da.shape != rt_da.shape or rt.ndim != 3 or rt.shape[:2] != da.shape:
            raise ValueError("三类预测必须按相同的[days,24]对齐")
        if not torch.isfinite(rt).all():
            raise ValueError("predicted_rt包含非有限值")

        dates: list[date] = [pd.Timestamp(values[0]).date() for values in self.local_timestamps]
        if len(set(dates)) != len(dates):
            raise ValueError("一个MPC episode内交付日不能重复")
        da_by_date = {item: da[index].numpy() for index, item in enumerate(dates)}
        rt_da_by_date = {item: rt_da[index].numpy() for index, item in enumerate(dates)}
        timestamps = tuple(item for day in self.local_timestamps for item in day)
        result = locked_continuous_mpc_backtest(
            da_by_date,
            rt_da_by_date,
            rt.numpy().reshape(-1, rt.shape[-1]),
            self.realized_da.detach().float().cpu().numpy().reshape(-1),
            self.realized_rt.detach().float().cpu().numpy().reshape(-1),
            timestamps,
            self.cfg,
            terminal_value_weight=self.terminal_value_weight,
        )
        revenue = [result["daily"][str(item)]["net_revenue"] for item in dates]
        return torch.tensor(revenue, dtype=torch.float32, device=predicted_da.device)


@torch.no_grad()
def estimate_continuous_mpc_gradients(
    predicted_da: torch.Tensor,
    predicted_rt_at_da: torch.Tensor,
    predicted_rt: torch.Tensor,
    objective: ContinuousMPCObjective,
    *,
    epsilon_spread: float,
    epsilon_rt: float,
    directions_spread: int,
    directions_rt: int,
    pred_clamp: tuple[float, float] = (-500.0, 10_000.0),
    direction_mode: str = "orthogonal",
    feedback_mode: str = "episode_scalar",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """分别估计DA价差和滚动RT预测的黑盒收益梯度。

    DA与RT-at-DA保持预测中点不变、沿相反方向扰动，因此价差改变而不会
    人为平移两条曲线。返回的价差梯度会在代理损失中以相反符号流入两个模型。
    """
    da = _as_day_hour(predicted_da, "predicted_da")
    rt_da = _as_day_hour(predicted_rt_at_da, "predicted_rt_at_da")
    if da.shape != rt_da.shape:
        raise ValueError("predicted_da与predicted_rt_at_da必须同形")
    if predicted_rt.ndim != 3 or predicted_rt.shape[:2] != da.shape:
        raise ValueError("predicted_rt必须是[days,24,horizon]")
    midpoint = 0.5 * (da + rt_da)
    spread = da - rt_da

    def spread_objective(candidate: torch.Tensor) -> torch.Tensor:
        return objective(
            midpoint + 0.5 * candidate,
            midpoint - 0.5 * candidate,
            predicted_rt,
        )

    def rt_objective(candidate: torch.Tensor) -> torch.Tensor:
        return objective(da, rt_da, candidate)

    grad_spread, spread_diagnostics = estimate_system_zo_gradient(
        spread,
        spread_objective,
        epsilon=epsilon_spread,
        directions=directions_spread,
        pred_clamp=pred_clamp,
        direction_mode=direction_mode,
        feedback_mode=feedback_mode,
    )
    grad_rt, rt_diagnostics = estimate_system_zo_gradient(
        predicted_rt,
        rt_objective,
        epsilon=epsilon_rt,
        directions=directions_rt,
        pred_clamp=pred_clamp,
        direction_mode=direction_mode,
        feedback_mode=feedback_mode,
    )
    return grad_spread, grad_rt, {
        "spread": spread_diagnostics,
        "rt": rt_diagnostics,
    }


def continuous_mpc_proxy_loss(
    predicted_da: torch.Tensor,
    predicted_rt_at_da: torch.Tensor,
    predicted_rt: torch.Tensor,
    grad_spread: torch.Tensor,
    grad_rt: torch.Tensor,
    *,
    scale: float = 1.0,
    reduction: str = "global_sum",
) -> torch.Tensor:
    """将黑盒方向注入三个模型，同时保持三套参数完全独立。"""
    spread_proxy = joint_proxy_loss(
        predicted_da - predicted_rt_at_da,
        grad_spread,
        scale,
        reduction=reduction,
    )
    rt_proxy = joint_proxy_loss(
        predicted_rt,
        grad_rt,
        scale,
        reduction=reduction,
    )
    return spread_proxy + rt_proxy
