"""三独立价格模型的联合监督辅助损失。

DA计划直接读取 ``p_DA - p_RT|DA``。因此除了分别拟合两条价格曲线，
还需要显式保护价差幅度与经济上重要时段的方向。这里不包含XGBoost，
也不读取验证/测试真值；训练器只在训练batch上调用这些函数。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_scale(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """把逐样本或逐时点标准差扩展到预测张量形状。"""
    scale = value.float()
    while scale.ndim < reference.ndim:
        scale = scale.unsqueeze(-1)
    return scale.clamp_min(1.0e-6)


def _pseudo_huber(error: torch.Tensor, delta: float) -> torch.Tensor:
    if delta <= 0:
        raise ValueError("delta必须为正")
    return delta * delta * (
        torch.sqrt(1.0 + (error / delta).square()) - 1.0
    )


def spread_auxiliary_losses(
    p_da: torch.Tensor,
    p_rt_at_da: torch.Tensor,
    true_da: torch.Tensor,
    true_rt_at_da: torch.Tensor,
    da_std: torch.Tensor,
    rt_std: torch.Tensor,
    *,
    delta: float = 1.0,
    direction_threshold: float = 6.0,
    direction_temperature: float = 25.0,
) -> dict[str, torch.Tensor]:
    """返回标准化价差幅度损失和显著价差方向损失。

    ``direction_threshold``附近对应当前退化成本下接近不值得交易的价差，
    因而方向项只监督绝对真实价差不小于该阈值的时段。
    """
    if p_da.shape != p_rt_at_da.shape or p_da.shape != true_da.shape:
        raise ValueError("DA、RT-at-DA预测和真值形状必须一致")
    if true_rt_at_da.shape != true_da.shape:
        raise ValueError("DA与RT-at-DA真值形状必须一致")
    if direction_threshold < 0 or direction_temperature <= 0:
        raise ValueError("方向阈值不能为负，温度必须为正")

    da_scale = _as_scale(da_std, true_da)
    rt_scale = _as_scale(rt_std, true_rt_at_da)
    spread_scale = torch.sqrt(da_scale.square() + rt_scale.square())
    predicted_spread = p_da - p_rt_at_da
    true_spread = true_da - true_rt_at_da
    magnitude = _pseudo_huber(
        (predicted_spread - true_spread) / spread_scale, delta
    ).mean()

    important = true_spread.abs() >= direction_threshold
    if bool(important.any()):
        logits = predicted_spread[important] / direction_temperature
        labels = (true_spread[important] > 0).to(logits.dtype)
        # 大价差更影响调度，但上限避免单个尖峰支配整个batch。
        denominator = max(direction_threshold, 1.0)
        weights = (true_spread[important].abs() / denominator).clamp(1.0, 10.0)
        elementwise = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        direction = (elementwise * weights).sum() / weights.sum()
        direction_accuracy = (
            (predicted_spread[important] > 0) == (true_spread[important] > 0)
        ).float().mean()
        important_fraction = important.float().mean()
    else:
        direction = predicted_spread.sum() * 0.0
        direction_accuracy = predicted_spread.new_tensor(float("nan"))
        important_fraction = predicted_spread.new_zeros(())

    return {
        "spread_magnitude": magnitude,
        "spread_direction": direction,
        "spread_direction_accuracy": direction_accuracy,
        "important_spread_fraction": important_fraction,
    }
