"""
spike_head.py — 尖峰检测头
===========================
从中间层 hidden state 预测各未来时刻的尖峰概率。
两个版本，用于 Step 6 敏感性对比：
  V1: 单层 Linear（基线，~31K params）
  V2: Linear-SiLU-Dropout-Linear（默认，~340K params）

设计依据（fusion_model_design_v3.md Section 3）：
  Skip connection 取自新 L5（= 原 TimesFM L7），该层 ΔSpike-F1 = -6.7%，
  是整个模型中最纯粹的"尖峰检测层"。
"""

import torch
import torch.nn as nn


class SpikeHeadV1(nn.Module):
    """单层线性版（基线）。
    Transformer 输出已高度非线性，单层头可能够用。
    """

    def __init__(self, d_model: int = 1280, horizon: int = 24):
        super().__init__()
        self.proj = nn.Linear(d_model, horizon)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, d_model]  来自新 L5 最后一个 patch 位置的 hidden state
        Returns:
            logits: [batch, horizon]  未经 sigmoid 的 spike logits
        """
        return self.proj(h)


class SpikeHeadV2(nn.Module):
    """双层版（默认）。
    Linear(d_model→hidden) → SiLU → Dropout → Linear(hidden→horizon)
    参数量 ~340K，对 133M 总参数可忽略。
    SiLU 激活与 TimesFM backbone 风格一致。
    """

    def __init__(self, d_model: int = 1280, hidden: int = 256, horizon: int = 24,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, d_model]
        Returns:
            logits: [batch, horizon]
        """
        return self.net(h)
