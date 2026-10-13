"""DA与RT-at-DA的可解释价差分解。

电池日前计划直接由DA-RT价差驱动。这里把第二套预测器的监督目标写成
标准化价差，而不是再独立拟合一条绝对价格曲线。价差中心和尺度只来自
训练期逐节点DA/RT统计量；未来DA、RT真值仅作为训练标签，绝不进入模型
forward。
"""
from __future__ import annotations

import torch


def spread_location_scale(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """返回逐样本DA-RT价差的中心与稳定尺度。"""
    da_mean = batch["price_da_mean"].reshape(-1, 1)
    rt_mean = batch["price_rt_mean"].reshape(-1, 1)
    da_std = batch["price_da_std"].reshape(-1, 1)
    rt_std = batch["price_rt_std"].reshape(-1, 1)
    center = da_mean - rt_mean
    scale = torch.sqrt(da_std.square() + rt_std.square()).clamp_min(1.0e-6)
    return center, scale


def normalized_spread_target(
    batch: dict,
    *,
    da_target_key: str = "price_da_tgt",
    rt_target_key: str = "price_rt_tgt",
) -> torch.Tensor:
    """用训练标签构造标准化DA-RT价差目标。"""
    if da_target_key not in batch or rt_target_key not in batch:
        raise KeyError("价差监督需要同一交付日的DA与RT训练标签")
    if batch[da_target_key].shape != batch[rt_target_key].shape:
        raise ValueError("DA与RT价差标签形状必须一致")
    center, scale = spread_location_scale(batch)
    return (batch[da_target_key] - batch[rt_target_key] - center) / scale


def reconstruct_rt_from_da_and_spread(
    p_da: torch.Tensor,
    normalized_spread: torch.Tensor,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由上游DA预测和价差预测唯一重建RT-at-DA价格。"""
    if p_da.shape != normalized_spread.shape:
        raise ValueError("DA预测与标准化价差预测形状必须一致")
    center, scale = spread_location_scale(batch)
    spread = normalized_spread * scale + center
    return p_da - spread, spread
