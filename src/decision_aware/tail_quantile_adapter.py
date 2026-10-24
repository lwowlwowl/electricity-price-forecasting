"""用于现有DA Transformer的非交叉分位数后训练头。

基础模型已经输出逐小时标准化点预测与解码表示。本模块不改变骨干：
它以原预测为中位数锚点，再用非负递增间隔构造单调分位数函数。
固定求积分权重将分位数预测转成条件均值，供价格点预测与LP/MPC使用。
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_QUANTILES = (
    0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99,
)


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("分位数初始间隔必须为正")
    return math.log(math.expm1(value))


def quantile_integral_weights(
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
) -> torch.Tensor:
    """梯形积分权重；区间两端使用最外侧分位数常值外推。"""
    levels = tuple(float(value) for value in quantiles)
    if len(levels) < 2 or levels != tuple(sorted(levels)):
        raise ValueError("quantiles必须严格递增且至少有两个值")
    if levels[0] <= 0 or levels[-1] >= 1:
        raise ValueError("quantiles必须位于(0,1)")
    if any(right <= left for left, right in zip(levels, levels[1:])):
        raise ValueError("quantiles不得重复")
    weights = []
    for index, level in enumerate(levels):
        if index == 0:
            weight = level + 0.5 * (levels[1] - level)
        elif index == len(levels) - 1:
            weight = (1.0 - level) + 0.5 * (level - levels[index - 1])
        else:
            weight = 0.5 * (levels[index + 1] - levels[index - 1])
        weights.append(weight)
    result = torch.tensor(weights, dtype=torch.float32)
    if not torch.isclose(result.sum(), torch.tensor(1.0), atol=1.0e-7):
        raise AssertionError("分位数积分权重未归一")
    return result


def multi_quantile_pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """对[B,H,Q]预测计算等权pinball loss。"""
    levels = torch.as_tensor(
        tuple(float(value) for value in quantiles),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    if prediction.ndim != 3 or prediction.shape[-1] != levels.numel():
        raise ValueError("prediction必须是[B,H,Q]")
    if target.shape != prediction.shape[:-1]:
        raise ValueError("target必须是与prediction前两维一致的[B,H]")
    error = target.unsqueeze(-1) - prediction
    loss = torch.maximum(levels * error, (levels - 1.0) * error)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"未知reduction: {reduction}")


class MeanResidualAdapter(nn.Module):
    """只学习条件均值修正的线性对照头。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.delta = nn.Linear(d_model, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self, representation: torch.Tensor, base_normalized: torch.Tensor
    ) -> torch.Tensor:
        if representation.shape[:-1] != base_normalized.shape:
            raise ValueError("表示与基础预测的[B,H]不一致")
        return base_normalized + self.delta(representation).squeeze(-1)


class NonCrossingQuantileAdapter(nn.Module):
    """以原预测为q50锚点的单调分位数后训练头。"""

    def __init__(
        self,
        d_model: int,
        quantiles: Iterable[float] = DEFAULT_QUANTILES,
        *,
        initial_gap: float = 0.10,
    ):
        super().__init__()
        self.quantiles = tuple(float(value) for value in quantiles)
        if 0.5 not in self.quantiles:
            raise ValueError("分位数网格必须包含0.5")
        self.median_index = self.quantiles.index(0.5)
        if self.quantiles != tuple(sorted(self.quantiles)):
            raise ValueError("分位数网格必须严格递增")
        lower_count = self.median_index
        upper_count = len(self.quantiles) - self.median_index - 1
        self.median_delta = nn.Linear(d_model, 1)
        self.lower_gaps = nn.Linear(d_model, lower_count)
        self.upper_gaps = nn.Linear(d_model, upper_count)
        for layer in (self.median_delta, self.lower_gaps, self.upper_gaps):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.median_delta.bias)
        gap_bias = _inverse_softplus(initial_gap)
        nn.init.constant_(self.lower_gaps.bias, gap_bias)
        nn.init.constant_(self.upper_gaps.bias, gap_bias)
        self.register_buffer(
            "integral_weights",
            quantile_integral_weights(self.quantiles),
            persistent=True,
        )

    def forward(
        self, representation: torch.Tensor, base_normalized: torch.Tensor
    ) -> torch.Tensor:
        if representation.shape[:-1] != base_normalized.shape:
            raise ValueError("表示与基础预测的[B,H]不一致")
        median = base_normalized + self.median_delta(representation).squeeze(-1)
        lower = F.softplus(self.lower_gaps(representation))
        upper = F.softplus(self.upper_gaps(representation))

        # lower从靠近q50向左累加，再反转为升序。
        lower_values = median.unsqueeze(-1) - torch.cumsum(lower, dim=-1)
        lower_values = torch.flip(lower_values, dims=(-1,))
        upper_values = median.unsqueeze(-1) + torch.cumsum(upper, dim=-1)
        return torch.cat(
            [lower_values, median.unsqueeze(-1), upper_values], dim=-1
        )

    def conditional_mean(self, quantile_prediction: torch.Tensor) -> torch.Tensor:
        if quantile_prediction.shape[-1] != len(self.quantiles):
            raise ValueError("分位数维度与固定网格不一致")
        weights = self.integral_weights.to(
            dtype=quantile_prediction.dtype,
            device=quantile_prediction.device,
        )
        return (quantile_prediction * weights).sum(dim=-1)


class RankPreservingPeakScaleAdapter(nn.Module):
    """只改变每日中位数以上幅度、不改变小时排序的条件尺度头。

    线性头读取原预测最高小时的解码表示。输出通过softplus转为
    正尺度，且初始严格为1。中位数以下的原预测保持不变。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.logit = nn.Linear(d_model, 1)
        nn.init.zeros_(self.logit.weight)
        # 0.25 + softplus(bias) = 1。
        nn.init.constant_(self.logit.bias, _inverse_softplus(0.75))

    def forward(
        self, representation: torch.Tensor, base_normalized: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if representation.shape[:-1] != base_normalized.shape:
            raise ValueError("表示与基础预测的[B,H]不一致")
        peak_index = base_normalized.argmax(dim=-1)
        gather_index = peak_index.reshape(-1, 1, 1).expand(
            -1, 1, representation.shape[-1]
        )
        peak_representation = representation.gather(1, gather_index).squeeze(1)
        scale = 0.25 + F.softplus(self.logit(peak_representation).squeeze(-1))
        median = base_normalized.median(dim=-1, keepdim=True).values
        positive_deviation = F.relu(base_normalized - median)
        corrected = base_normalized + (scale.reshape(-1, 1) - 1.0) * positive_deviation
        return corrected, scale


def log_peak_excursion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """对逐日正峰值相对中位数的log1p幅度计算Smooth-L1。"""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction和target必须是同形[B,H]")
    pred_median = prediction.median(dim=-1).values
    true_median = target.median(dim=-1).values
    pred_excursion = (prediction.amax(dim=-1) - pred_median).clamp_min(0.0)
    true_excursion = (target.amax(dim=-1) - true_median).clamp_min(0.0)
    return F.smooth_l1_loss(
        torch.log1p(pred_excursion),
        torch.log1p(true_excursion),
    )

