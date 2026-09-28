"""独立DA模型的预测损失与零阶决策代理损失。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .zero_order import compute_l_proxy_scaled, estimate_zo_gradient


def da_decision_aware_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    simulator,
    policy,
    alpha: float,
    beta: float,
    cfg,
    epsilon: float,
):
    """返回可反传的DA总损失及训练日志。

    收益口径是DA-only物理执行代理：预测先进入固定HardTopK策略，再由
    BESS模拟器按真实DA价结算。它不是完整DA/RT双结算，避免在计划SOC
    可行性尚未补齐时过度声称。
    """
    prediction_f = prediction.float()
    target_f = target.float()
    pred_loss = F.huber_loss(
        prediction_f, target_f, delta=cfg.huber_delta, reduction="mean"
    )
    normalized_pred_loss = pred_loss / cfg.pred_scale

    if beta > 0.0:
        zo_gradient = estimate_zo_gradient(
            prediction_f,
            target_f,
            simulator,
            policy,
            epsilon=epsilon,
            K=cfg.zo_K,
            pred_clamp=tuple(cfg.pred_clamp),
        )
        proxy_loss = compute_l_proxy_scaled(
            prediction_f, zo_gradient, cfg.proxy_scale
        )
        gradient_norm = float(zo_gradient.norm().detach().cpu())
    else:
        proxy_loss = prediction_f.new_zeros(())
        gradient_norm = 0.0

    total = alpha * normalized_pred_loss + beta * proxy_loss
    with torch.no_grad():
        action = policy(prediction_f)
        revenue = simulator(action, target_f)
        mae = torch.mean(torch.abs(prediction_f - target_f))
        rmse = torch.mean((prediction_f - target_f) ** 2).sqrt()
    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_pred": float(pred_loss.detach().cpu()),
        "loss_proxy": float(proxy_loss.detach().cpu()),
        "revenue": float(revenue.mean().detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "zo_gradient_norm": gradient_norm,
        "alpha": float(alpha),
        "beta": float(beta),
    }
    return total, metrics
