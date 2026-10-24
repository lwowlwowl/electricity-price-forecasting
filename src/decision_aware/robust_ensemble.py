"""Low-parameter, interpretable calibration of a fixed forecast ensemble."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EnsembleCalibratorConfig:
    members: int = 3
    hours: int = 24
    maximum_center_bias: float = 10.0
    maximum_spread_bias: float = 20.0
    maximum_spread_log_scale: float = 0.40
    maximum_rt_bias: float = 15.0
    maximum_rt_log_scale: float = 0.20

    def to_dict(self) -> dict:
        return asdict(self)


class EconomicEnsembleCalibrator(nn.Module):
    """Convex seed weights plus bounded hourly economic corrections.

    The zero initialization is exactly the equal-weight arithmetic ensemble.
    DA and RT-at-DA are adjusted through a center/spread factorization, making
    the economically relevant intervention directly inspectable.
    """

    task_names = ("da", "rt_at_da", "rt")

    def __init__(self, config: EnsembleCalibratorConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnsembleCalibratorConfig()
        shape = (len(self.task_names), self.config.members)
        self.seed_logits = nn.Parameter(torch.zeros(shape))
        self.center_bias_raw = nn.Parameter(torch.zeros(self.config.hours))
        self.spread_bias_raw = nn.Parameter(torch.zeros(self.config.hours))
        self.spread_log_scale_raw = nn.Parameter(torch.zeros(self.config.hours))
        self.rt_bias_raw = nn.Parameter(torch.zeros(self.config.hours))
        self.rt_log_scale_raw = nn.Parameter(torch.zeros(self.config.hours))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.seed_logits, dim=-1)

    def calibrate_da_pair(
        self,
        da_members: torch.Tensor,
        rt_at_da_members: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if da_members.ndim != 3 or rt_at_da_members.shape != da_members.shape:
            raise ValueError("DA member tensors must share [batch,members,hours]")
        if da_members.shape[1:] != (self.config.members, self.config.hours):
            raise ValueError("member/hour dimensions do not match calibrator config")
        weights = self.weights()
        da = torch.einsum("m,bmh->bh", weights[0], da_members)
        rt_at_da = torch.einsum("m,bmh->bh", weights[1], rt_at_da_members)
        center = 0.5 * (da + rt_at_da)
        spread = da - rt_at_da
        center = center + self.config.maximum_center_bias * torch.tanh(
            self.center_bias_raw
        )
        spread = spread * torch.exp(
            self.config.maximum_spread_log_scale
            * torch.tanh(self.spread_log_scale_raw)
        ) + self.config.maximum_spread_bias * torch.tanh(self.spread_bias_raw)
        return center + 0.5 * spread, center - 0.5 * spread

    def calibrate_rt(self, rt_members: torch.Tensor) -> torch.Tensor:
        if rt_members.ndim != 4:
            raise ValueError("RT members must have [batch,members,hours,horizon]")
        if rt_members.shape[1:3] != (self.config.members, self.config.hours):
            raise ValueError("member/hour dimensions do not match calibrator config")
        rt = torch.einsum("m,bmhk->bhk", self.weights()[2], rt_members)
        rt_scale = torch.exp(
            self.config.maximum_rt_log_scale * torch.tanh(self.rt_log_scale_raw)
        )[None, :, None]
        rt_bias = (
            self.config.maximum_rt_bias * torch.tanh(self.rt_bias_raw)
        )[None, :, None]
        return rt * rt_scale + rt_bias

    def forward(
        self,
        da_members: torch.Tensor,
        rt_at_da_members: torch.Tensor,
        rt_members: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rt_members.ndim != 4 or rt_members.shape[:3] != da_members.shape:
            raise ValueError("RT members must have [batch,members,hours,horizon]")
        calibrated_da, calibrated_rt_at_da = self.calibrate_da_pair(
            da_members, rt_at_da_members
        )
        calibrated_rt = self.calibrate_rt(rt_members)
        return calibrated_da, calibrated_rt_at_da, calibrated_rt

    def regularization(self) -> torch.Tensor:
        weights = self.weights().clamp_min(1e-8)
        uniform_kl = (weights * torch.log(weights * self.config.members)).sum()
        corrections = sum(
            parameter.square().mean()
            for name, parameter in self.named_parameters()
            if name != "seed_logits"
        )
        return uniform_kl + corrections

    @torch.no_grad()
    def interpretation(self) -> dict:
        weights = self.weights().cpu()
        return {
            "seed_weights": {
                task: weights[index].tolist()
                for index, task in enumerate(self.task_names)
            },
            "center_bias_usd_per_mwh": (
                self.config.maximum_center_bias * torch.tanh(self.center_bias_raw)
            ).cpu().tolist(),
            "spread_bias_usd_per_mwh": (
                self.config.maximum_spread_bias * torch.tanh(self.spread_bias_raw)
            ).cpu().tolist(),
            "spread_scale": torch.exp(
                self.config.maximum_spread_log_scale
                * torch.tanh(self.spread_log_scale_raw)
            ).cpu().tolist(),
            "rt_bias_usd_per_mwh": (
                self.config.maximum_rt_bias * torch.tanh(self.rt_bias_raw)
            ).cpu().tolist(),
            "rt_scale": torch.exp(
                self.config.maximum_rt_log_scale
                * torch.tanh(self.rt_log_scale_raw)
            ).cpu().tolist(),
        }
