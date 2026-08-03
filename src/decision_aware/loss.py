"""loss.py — 先行版联合损失 + α/β 退火（todo3 #2 决策 B）。

L = α·L_pred(Huber) + β·L_bus(Regret)
退火：α: alpha0→alpha_end，β: beta0→beta_end，线性跨 anneal_epochs，之后恒定。
当 π 可微时 β→1 即纯业务损失驱动（与正式版同口径）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import PilotConfig
from .policy import BESSSimulator, STEPolicy, compute_regret


def huber_loss(pred: torch.Tensor, true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """逐元素 Huber，对电价尖峰更鲁棒（todo3 #3 先行版）。"""
    return F.huber_loss(pred, true, reduction="mean", delta=delta)


def anneal_alpha_beta(epoch: int, cfg: PilotConfig):
    """线性退火 (alpha0,beta0) → (alpha_end,beta_end)，跨 anneal_epochs 后恒定。"""
    frac = min(1.0, max(0.0, epoch / max(1, cfg.anneal_epochs)))
    alpha = cfg.alpha0 + frac * (cfg.alpha_end - cfg.alpha0)
    beta = cfg.beta0 + frac * (cfg.beta_end - cfg.beta0)
    return float(alpha), float(beta)


def total_loss(
    p_da: torch.Tensor,          # [B, H] 真实尺度预测（模型已反归一化 + clamp）
    price_tgt: torch.Tensor,     # [B, H] 真实电价
    simulator: BESSSimulator,
    policy,
    alpha: float,
    beta: float,
    delta: float = 1.0,
    pred_scale: float = 1.0,      # 把 L_pred 归一到 O(1)，与 L_bus 同尺度，防 β→1 梯度爆炸
    bus_scale: float = 1.0,       # 把 L_bus  归一到 O(1)
    oracle: str = "greedy",       # "greedy"(v1) 或 "lp"(v2 真上界)
):
    """返回 (total_loss, metrics_dict)。metrics_dict 全为 .detach().item() 标量，供日志。

    total = α·(L_pred/pred_scale) + β·(L_bus/bus_scale)，两分量均 O(1)，
    α/β 退火才是真正的权衡（否则 regret 尺度 ~10× 于 pred，β=1 直接 NaN）。
    oracle="lp" 时 R* 用 LP 真上界（模型不可能超过，regret≥0）。
    """
    l_pred_raw = huber_loss(p_da, price_tgt, delta=delta)
    R_model, R_star, regret = compute_regret(p_da, price_tgt, simulator, policy, oracle=oracle)
    l_bus_raw = regret.mean()
    l_pred = l_pred_raw / pred_scale
    l_bus = l_bus_raw / bus_scale
    total = alpha * l_pred + beta * l_bus
    metrics = {
        "loss_total": float(total.detach().item()),       # 归一化后的总损失（O(1)）
        "loss_pred": float(l_pred_raw.detach().item()),  # 原始 Huber（$ 尺度，~25）
        "loss_bus": float(l_bus_raw.detach().item()),    # 原始 Regret（$ 尺度，~250）
        "R_model": float(R_model.detach().mean().item()),
        "R_star": float(R_star.detach().mean().item()),
        "regret": float(regret.detach().mean().item()),
        "alpha": float(alpha),
        "beta": float(beta),
    }
    return total, metrics
