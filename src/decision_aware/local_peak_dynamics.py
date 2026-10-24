"""可解释的日内峰值幅度结构量与训练损失。

该模块不识别某个固定价格阈值，也不改变LP/MPC。它只比较24小时曲线中
``最高价 - 日内中位价``，用于诊断和修复均匀MAE造成的曲线压平。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def daily_peak_excursion(values: torch.Tensor) -> torch.Tensor:
    """返回每条日曲线的``max - median``，输入最后一维必须为24小时。"""
    if values.ndim < 2 or values.shape[-1] != 24:
        raise ValueError("DA日峰值幅度要求最后一维恰好为24小时")
    values_fp32 = values.float()
    ordered = values_fp32.sort(dim=-1).values
    # 24为偶数，采用第12、13个次序统计量的均值，与numpy median和报告一致。
    median = 0.5 * (ordered[..., 11] + ordered[..., 12])
    return values_fp32.amax(dim=-1) - median


def daily_peak_excursion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    historical_scale: torch.Tensor,
    *,
    beta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """对标准化日峰值幅度误差计算Smooth-L1。

    ``historical_scale``必须只来自起报时可见的历史窗口；下限1 USD/MWh，
    防止平稳节点把很小的美元误差放大成不稳定梯度。
    """
    if prediction.shape != target.shape:
        raise ValueError("预测与真值形状必须一致")
    if beta <= 0:
        raise ValueError("beta必须为正")
    pred_excursion = daily_peak_excursion(prediction)
    true_excursion = daily_peak_excursion(target)
    scale = historical_scale.float().reshape(-1).abs().clamp_min(1.0)
    if scale.shape != pred_excursion.shape:
        raise ValueError("historical_scale必须为每条日曲线提供一个尺度")
    normalized_error = (pred_excursion - true_excursion) / scale
    return F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        beta=beta,
        reduction=reduction,
    )


def daily_peak_excursion_errors(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    """返回逐日峰值幅度与绝对误差，供验证集选模和审计。"""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim < 2 or pred.shape[-1] != 24:
        raise ValueError("prediction与target必须同形，且最后一维为24")
    predicted = np.max(pred, axis=-1) - np.median(pred, axis=-1)
    actual = np.max(truth, axis=-1) - np.median(truth, axis=-1)
    return {
        "predicted": predicted,
        "actual": actual,
        "absolute_error": np.abs(predicted - actual),
    }
