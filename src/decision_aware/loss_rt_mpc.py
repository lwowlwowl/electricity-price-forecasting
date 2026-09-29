"""滚动RT模型的前瞻MPC决策感知损失。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .zero_order import compute_l_proxy_scaled, estimate_zo_gradient


def rt_mpc_decision_aware_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    simulator,
    policy,
    alpha: float,
    beta: float,
    cfg,
    epsilon: float,
):
    """用Huber保留价格尺度，用零阶梯度让预测服务于MPC收益。

    这里只训练独立滚动RT模型。无偏差罚金时，价格接受者的
    RT优化可与已锁定DA头寸分解，因此该局部业务损失不需要
    把DA模型接入同一张计算图。最终checkpoint仍由连续SOC滚动回测选择。
    """
    prediction_f = prediction.float()
    target_f = target.float()
    pred_loss = F.huber_loss(
        prediction_f,
        target_f,
        delta=cfg.huber_delta,
        reduction="mean",
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
            prediction_f,
            zo_gradient,
            cfg.proxy_scale,
        )
        gradient_norm = float(zo_gradient.norm().detach().cpu())
    else:
        proxy_loss = prediction_f.new_zeros(())
        gradient_norm = 0.0

    total = alpha * normalized_pred_loss + beta * proxy_loss
    with torch.no_grad():
        action = policy(prediction_f)
        local_revenue = simulator(action, target_f)
        mae = torch.mean(torch.abs(prediction_f - target_f))
        rmse = torch.mean((prediction_f - target_f) ** 2).sqrt()
    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_pred": float(pred_loss.detach().cpu()),
        "loss_proxy": float(proxy_loss.detach().cpu()),
        "local_horizon_revenue": float(local_revenue.mean().detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "zo_gradient_norm": gradient_norm,
        "alpha": float(alpha),
        "beta": float(beta),
    }
    return total, metrics
