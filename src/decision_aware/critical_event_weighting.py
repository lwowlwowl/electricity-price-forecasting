"""训练期逐节点尖峰与关键DA-RT价差加权。

该模块只根据监督标签产生损失权重，不改变模型输入。阈值必须由训练集拟合，
验证和测试阶段既不重新拟合，也不使用加权指标选模。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class CriticalEventThresholds:
    """按全局node id排列的训练期阈值。"""

    price_lower: torch.Tensor
    price_upper: torch.Tensor
    abs_spread: torch.Tensor

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "price_lower": self.price_lower.detach().cpu().tolist(),
            "price_upper": self.price_upper.detach().cpu().tolist(),
            "abs_spread": self.abs_spread.detach().cpu().tolist(),
        }


def validate_critical_event_config(cfg: object) -> None:
    lower = float(getattr(cfg, "global_event_price_lower_quantile", 0.025))
    upper = float(getattr(cfg, "global_event_price_upper_quantile", 0.975))
    spread = float(getattr(cfg, "global_event_spread_quantile", 0.90))
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("价格尾部分位数必须满足0<=lower<upper<=1")
    if not 0.0 <= spread <= 1.0:
        raise ValueError("价差分位数必须位于[0,1]")
    for name in (
        "global_event_price_weight",
        "global_event_spread_weight",
        "global_event_sign_reversal_weight",
        "global_event_rt_first_step_weight",
    ):
        if float(getattr(cfg, name, 0.0)) < 0.0:
            raise ValueError(f"{name}不能为负")
    if float(getattr(cfg, "global_event_max_weight", 4.0)) < 1.0:
        raise ValueError("global_event_max_weight不能小于1")


def fit_critical_event_thresholds(
    dataset: object,
    task_name: str,
    cfg: object,
    *,
    node_count: int,
) -> CriticalEventThresholds:
    """从MultiNodeDataset的训练样本拟合逐节点分位数。

    RT按日包装后包含24个滚动窗口；直接遍历监督样本可使阈值与训练损失
    实际看到的元素分布完全一致。此函数不得传入验证或测试Dataset。
    """
    validate_critical_event_config(cfg)
    if task_name not in {"da", "rt_at_da", "rt"}:
        raise ValueError(f"未知任务{task_name!r}")
    target_key = {
        "da": "price_da_tgt",
        "rt_at_da": "price_rt_at_da_tgt",
        "rt": "price_rt_tgt",
    }[task_name]
    price_by_node: list[list[torch.Tensor]] = [[] for _ in range(node_count)]
    spread_by_node: list[list[torch.Tensor]] = [[] for _ in range(node_count)]
    for index in range(len(dataset)):
        sample: Mapping[str, torch.Tensor] = dataset[index]
        node_id = int(sample["node_id"].item())
        da_key = (
            "price_da_tgt"
            if "price_da_tgt" in sample
            else "price_da_tgt_weighting_only"
        )
        da = sample[da_key].detach().float().reshape(-1)
        rt_key = (
            "price_rt_tgt"
            if "price_rt_tgt" in sample
            else "price_rt_at_da_tgt"
        )
        rt = sample[rt_key].detach().float().reshape(-1)
        target = sample[target_key].detach().float().reshape(-1)
        price_by_node[node_id].append(target)
        spread_by_node[node_id].append((da - rt).abs())

    lower_q = float(getattr(cfg, "global_event_price_lower_quantile", 0.025))
    upper_q = float(getattr(cfg, "global_event_price_upper_quantile", 0.975))
    spread_q = float(getattr(cfg, "global_event_spread_quantile", 0.90))
    lower_values = torch.full((node_count,), float("nan"), dtype=torch.float32)
    upper_values = torch.full((node_count,), float("nan"), dtype=torch.float32)
    spread_values = torch.full((node_count,), float("nan"), dtype=torch.float32)
    for node_id in range(node_count):
        if not price_by_node[node_id]:
            continue
        prices = torch.cat(price_by_node[node_id])
        spreads = torch.cat(spread_by_node[node_id])
        lower_values[node_id] = torch.quantile(prices, lower_q)
        upper_values[node_id] = torch.quantile(prices, upper_q)
        spread_values[node_id] = torch.quantile(spreads, spread_q)
    return CriticalEventThresholds(lower_values, upper_values, spread_values)


def critical_event_weights(
    batch: Mapping[str, torch.Tensor],
    task_name: str,
    thresholds: CriticalEventThresholds,
    cfg: object,
) -> torch.Tensor:
    """返回与任务目标同形状、最小为1且有上限的元素权重。"""
    validate_critical_event_config(cfg)
    if task_name not in {"da", "rt_at_da", "rt"}:
        raise ValueError(f"未知任务{task_name!r}")
    target_key = {
        "da": "price_da_tgt",
        "rt_at_da": "price_rt_at_da_tgt",
        "rt": "price_rt_tgt",
    }[task_name]
    target = batch[target_key]
    da_key = (
        "price_da_tgt"
        if "price_da_tgt" in batch
        else "price_da_tgt_weighting_only"
    )
    da = batch[da_key]
    rt_key = (
        "price_rt_tgt"
        if "price_rt_tgt" in batch
        else "price_rt_at_da_tgt"
    )
    rt = batch[rt_key]
    if target.shape != da.shape or target.shape != rt.shape:
        raise ValueError("关键事件权重要求DA、RT和任务目标形状一致")
    node_ids = batch["node_id"].long()
    lower_table = thresholds.price_lower.to(device=target.device, dtype=target.dtype)
    upper_table = thresholds.price_upper.to(device=target.device, dtype=target.dtype)
    spread_table = thresholds.abs_spread.to(device=target.device, dtype=target.dtype)
    lower = lower_table[node_ids].reshape(-1, *([1] * (target.ndim - 1)))
    upper = upper_table[node_ids].reshape(-1, *([1] * (target.ndim - 1)))
    spread_cutoff = spread_table[node_ids].reshape(-1, *([1] * (target.ndim - 1)))
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()
                and torch.isfinite(spread_cutoff).all()):
        raise ValueError("batch包含没有训练期关键事件阈值的节点")

    weights = torch.ones_like(target)
    price_tail = (target <= lower) | (target >= upper)
    large_spread = (da - rt).abs() >= spread_cutoff
    sign_reversal = (da * rt) < 0.0
    weights = weights + price_tail.to(target.dtype) * float(
        getattr(cfg, "global_event_price_weight", 1.0)
    )
    weights = weights + large_spread.to(target.dtype) * float(
        getattr(cfg, "global_event_spread_weight", 1.0)
    )
    weights = weights + sign_reversal.to(target.dtype) * float(
        getattr(cfg, "global_event_sign_reversal_weight", 0.5)
    )
    if task_name == "rt":
        first_step = torch.zeros_like(target)
        first_step[..., 0] = 1.0
        weights = weights + first_step * float(
            getattr(cfg, "global_event_rt_first_step_weight", 0.5)
        )
    return weights.clamp(max=float(getattr(cfg, "global_event_max_weight", 4.0)))


def weighted_elementwise_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if loss.shape != weights.shape:
        raise ValueError("逐元素损失与关键事件权重形状不一致")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)
