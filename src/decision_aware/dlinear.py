"""最小单变量DLinear基线。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DLinearForecaster(nn.Module):
    """移动平均分解后分别线性外推趋势与季节项。"""

    def __init__(self, context_len: int, horizon: int, kernel_size: int = 25) -> None:
        super().__init__()
        if context_len < 1 or horizon < 1:
            raise ValueError("context_len和horizon必须为正")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("DLinear移动平均核必须为正奇数")
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.kernel_size = int(kernel_size)
        self.seasonal = nn.Linear(self.context_len, self.horizon)
        self.trend = nn.Linear(self.context_len, self.horizon)
        nn.init.constant_(self.seasonal.weight, 1.0 / self.context_len)
        nn.init.constant_(self.trend.weight, 1.0 / self.context_len)
        nn.init.zeros_(self.seasonal.bias)
        nn.init.zeros_(self.trend.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 2 or history.shape[1] != self.context_len:
            raise ValueError(
                f"DLinear输入必须为[B,{self.context_len}]，实际为{tuple(history.shape)}"
            )
        padding = (self.kernel_size - 1) // 2
        padded = F.pad(history.unsqueeze(1), (padding, padding), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1).squeeze(1)
        seasonal = history - trend
        return self.seasonal(seasonal) + self.trend(trend)
