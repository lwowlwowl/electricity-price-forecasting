"""可解释的基础价格与物理压力门控稀缺残差分解。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def canonical_output_decomposition(value: str) -> str:
    normalized = str(value).lower().strip()
    aliases = {
        "": "direct",
        "direct": "direct",
        "physical_scarcity_residual": "physical_scarcity_residual",
    }
    if normalized not in aliases:
        raise ValueError(
            "global_output_decomposition必须是direct或physical_scarcity_residual"
        )
    return aliases[normalized]


class PhysicalScarcityResidual(nn.Module):
    """ ``base + gate(physical pressure) * residual`` 的受限残差头。

    门控只使用四个可审计压力量：目标小时的上周同小时净负荷、最近24小时
    均值、最近24小时最大值与最近24小时净爬坡。四个压力系数经softplus
    约束为非负，因此任何压力量单独上升都不会降低稀缺门控。日历项只描述
    已知的小时/星期/月份效应。残差幅度读取原模型表征，以保留节点差异。

    残差头严格零初始化，使新增结构在加载旧checkpoint后精确等价于原模型；
    训练可以据此区分真实结构收益和随机初始化扰动。
    """

    def __init__(self, d_model: int, calendar_dim: int = 8):
        super().__init__()
        self.pressure_raw_weights = nn.Parameter(torch.zeros(4))
        self.calendar_gate = nn.Linear(calendar_dim, 1, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(()))
        self.residual_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.calendar_gate.weight)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def pressure_features(batch: dict, horizon: int) -> torch.Tensor:
        context = batch["physical_pressure_ctx"].float()
        lag168 = batch["physical_pressure_tgt_lag168"].float()
        if context.ndim == 3 and context.shape[-1] == 1:
            context = context.squeeze(-1)
        if context.ndim != 2 or lag168.ndim != 2:
            raise ValueError("物理压力上下文与lag168目标必须是二维张量")
        if lag168.shape[1] != horizon:
            raise ValueError("物理压力lag168长度必须等于预测时域")
        if context.shape[1] < 24:
            raise ValueError("物理稀缺门控至少需要24小时历史上下文")
        recent = context[:, -24:]
        recent_mean = recent.mean(dim=1, keepdim=True).expand(-1, horizon)
        recent_max = recent.amax(dim=1, keepdim=True).expand(-1, horizon)
        recent_ramp = (recent[:, -1:] - recent[:, :1]).expand(-1, horizon)
        return torch.stack(
            [lag168, recent_mean, recent_max, recent_ramp], dim=-1
        )

    def forward(self, representation: torch.Tensor, batch: dict) -> dict:
        if representation.ndim != 3:
            raise ValueError("representation必须是[B,H,D]")
        horizon = representation.shape[1]
        pressure = self.pressure_features(batch, horizon)
        calendar = batch["cal_tgt"].float()
        if calendar.shape[:2] != representation.shape[:2]:
            raise ValueError("cal_tgt的[B,H]必须与representation一致")
        monotone_weights = F.softplus(self.pressure_raw_weights)
        pressure_logit = (pressure * monotone_weights.view(1, 1, -1)).sum(-1)
        calendar_logit = self.calendar_gate(calendar).squeeze(-1)
        gate = torch.sigmoid(pressure_logit + calendar_logit + self.gate_bias)
        latent_residual = self.residual_head(representation).squeeze(-1)
        gated_residual = gate * latent_residual
        return {
            "normalized_residual": gated_residual,
            "latent_normalized_residual": latent_residual,
            "gate": gate,
            "pressure_features": pressure,
            "monotone_pressure_weights": monotone_weights,
        }
