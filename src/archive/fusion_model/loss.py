"""
loss.py — ElecFM 联合损失函数
==============================
L_total = lambda_pinball × L_pinball + lambda_spike × L_spike

设计说明（fusion_model_design_v3.md Section 4）：
  - Pinball：9 分位数（q0.1…q0.9）对 24 步求平均，与 TimesFM 训练目标一致
  - BCE Spike：带 pos_weight 处理 P95 类别不平衡（~95:5 → pos_weight≈19）
  - 默认权重 0.8:0.2，确保精度为主目标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 9 个固定分位数水平（与评估框架一致）
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def pinball_loss(
    y_true: torch.Tensor,   # [B, horizon]
    q_pred: torch.Tensor,   # [B, horizon, 9]
) -> torch.Tensor:
    """
    所有分位数的平均 Pinball Loss（标量）。
    对 batch × horizon × 9 个分位数取均值。
    """
    levels = torch.tensor(QUANTILE_LEVELS, dtype=y_true.dtype, device=y_true.device)
    # y_true 扩展到 [B, horizon, 9] 与 q_pred 对齐
    y = y_true.unsqueeze(-1).expand_as(q_pred)          # [B, H, 9]
    diff = y - q_pred                                   # [B, H, 9]
    loss = torch.where(diff >= 0, levels * diff, (levels - 1.0) * diff)
    return loss.mean()


def spike_bce_loss(
    spike_logits: torch.Tensor,   # [B, horizon]
    spike_labels: torch.Tensor,   # [B, horizon]  float 0/1
    pos_weight: float = 19.0,
) -> torch.Tensor:
    """
    带正样本权重的 BCE 损失（P95 正样本率 ~5% → pos_weight ≈ 19）。
    使用 BCEWithLogitsLoss 保证数值稳定（内置 sigmoid）。
    pos_weight 告诉模型：漏报一个尖峰的代价是误报的 pos_weight 倍。
    """
    pw = torch.tensor([pos_weight], dtype=spike_logits.dtype, device=spike_logits.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    return criterion(spike_logits, spike_labels)


def combined_loss(
    y_true: torch.Tensor,          # [B, horizon]  真实电价
    q_pred: torch.Tensor,          # [B, horizon, 9]  分位数预测
    spike_logits: torch.Tensor,    # [B, horizon]  spike logits
    spike_labels: torch.Tensor,    # [B, horizon]  float 0/1
    lambda_pinball: float = 0.8,
    lambda_spike: float = 0.2,
    spike_pos_weight: float = 19.0,
) -> tuple[torch.Tensor, dict]:
    """
    ElecFM 的联合损失。

    Returns
    -------
    (loss_total, loss_dict) 其中 loss_dict 含各分量的数值（用于监控）。
    """
    l_pin = pinball_loss(y_true, q_pred)
    l_spk = spike_bce_loss(spike_logits, spike_labels, pos_weight=spike_pos_weight)
    total = lambda_pinball * l_pin + lambda_spike * l_spk

    return total, {
        "loss_total": total.item(),
        "loss_pinball": l_pin.item(),
        "loss_spike": l_spk.item(),
    }
