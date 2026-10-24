"""可解释的DA/RT-at-DA经济价差分解。

模块不在两套专家之间做路由，而是联合读取两个任务的解码表征，将价格
分解为共同水平、三状态经济方向和超过固定盈亏平衡门槛的条件幅度。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


NEUTRAL = 0
DA_HIGH = 1
RT_HIGH = 2


def economic_spread_regime(
    spread: torch.Tensor, threshold: float = 6.0
) -> torch.Tensor:
    """把DA-RT价差映射为中性、DA高和RT高三个互斥状态。"""
    if threshold <= 0:
        raise ValueError("threshold必须为正")
    result = torch.zeros_like(spread, dtype=torch.long)
    result = torch.where(spread > threshold, DA_HIGH, result)
    result = torch.where(spread < -threshold, RT_HIGH, result)
    return result


def conditional_excess_target(
    spread: torch.Tensor, threshold: float = 6.0
) -> torch.Tensor:
    """返回超过经济门槛的非负幅度；中性小时严格为零。"""
    if threshold <= 0:
        raise ValueError("threshold必须为正")
    return (spread.abs() - float(threshold)).clamp_min(0.0)


@dataclass(frozen=True)
class EconomicSpreadLoss:
    total: torch.Tensor
    regime: torch.Tensor
    conditional_excess: torch.Tensor
    reconstruction: torch.Tensor
    tradeable_count: int


class EconomicSpreadFactorizationHead(nn.Module):
    """联合DA/RT表征的三状态与条件幅度头。

    头部以基础预测为残差锚点。所有学习层零初始化时，硬输出严格等于基础
    DA/RT预测，便于安全加载已有checkpoint并明确测量新增机制的贡献。
    """

    def __init__(
        self,
        d_model: int,
        *,
        threshold: float = 6.0,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        prior_strength: float = 2.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model必须为正")
        if threshold <= 0:
            raise ValueError("threshold必须为正")
        if prior_strength <= 0:
            raise ValueError("prior_strength必须为正")
        width = int(hidden_dim or d_model)
        self.d_model = int(d_model)
        self.threshold = float(threshold)
        self.prior_strength = float(prior_strength)
        input_dim = 3 * self.d_model + 1
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.regime_residual = nn.Linear(width, 3)
        self.excess_residual = nn.Linear(width, 2)
        nn.init.zeros_(self.regime_residual.weight)
        nn.init.zeros_(self.regime_residual.bias)
        nn.init.zeros_(self.excess_residual.weight)
        nn.init.zeros_(self.excess_residual.bias)

    @staticmethod
    def _validate(
        da_representation: torch.Tensor,
        rt_representation: torch.Tensor,
        base_da_price: torch.Tensor,
        base_rt_price: torch.Tensor,
    ) -> None:
        if da_representation.ndim != 3 or rt_representation.ndim != 3:
            raise ValueError("DA和RT表征必须是[B,H,D]")
        if da_representation.shape != rt_representation.shape:
            raise ValueError("DA和RT表征必须同形")
        expected = da_representation.shape[:2]
        if base_da_price.shape != expected or base_rt_price.shape != expected:
            raise ValueError("基础DA和RT价格必须是与表征对应的[B,H]")

    def forward(
        self,
        da_representation: torch.Tensor,
        rt_representation: torch.Tensor,
        base_da_price: torch.Tensor,
        base_rt_price: torch.Tensor,
        *,
        spread_scale: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate(
            da_representation, rt_representation, base_da_price, base_rt_price
        )
        base_spread = base_da_price - base_rt_price
        base_level = 0.5 * (base_da_price + base_rt_price)
        if spread_scale is None:
            scale = torch.full_like(base_spread, self.threshold)
        else:
            scale = torch.as_tensor(
                spread_scale, dtype=base_spread.dtype, device=base_spread.device
            )
            while scale.ndim < base_spread.ndim:
                scale = scale.unsqueeze(-1)
            scale = torch.broadcast_to(scale, base_spread.shape).clamp_min(1.0e-6)

        features = torch.cat(
            [
                da_representation,
                rt_representation,
                da_representation - rt_representation,
                (base_spread / scale).unsqueeze(-1),
            ],
            dim=-1,
        )
        fused = self.fusion(features)

        base_regime = economic_spread_regime(base_spread, self.threshold)
        prior_logits = F.one_hot(base_regime, num_classes=3).to(
            dtype=base_spread.dtype
        ) * self.prior_strength
        regime_logits = prior_logits + self.regime_residual(fused)
        regime_probability = torch.softmax(regime_logits, dim=-1)
        regime_class = regime_logits.argmax(dim=-1)

        base_excess = conditional_excess_target(base_spread, self.threshold)
        excess_residual = self.excess_residual(fused)
        conditional_excess = (
            base_excess.unsqueeze(-1) + excess_residual
        ).clamp_min(0.0)

        neutral_spread = base_spread.clamp(-self.threshold, self.threshold)
        da_high_spread = self.threshold + conditional_excess[..., 0]
        rt_high_spread = -self.threshold - conditional_excess[..., 1]
        state_spreads = torch.stack(
            [neutral_spread, da_high_spread, rt_high_spread], dim=-1
        )
        hard_weights = F.one_hot(regime_class, num_classes=3).to(
            dtype=regime_probability.dtype
        )
        # 训练仍沿三状态硬决策的直通梯度传播；部署价差保留基础模型的方向，
        # 只扩张基础DA_HIGH的条件幅度，避免低置信RT_HIGH误判变成反向交易。
        straight_through_weights = (
            hard_weights + regime_probability - regime_probability.detach()
        )
        soft_spread = (straight_through_weights * state_spreads).sum(dim=-1)
        hard_spread = state_spreads.gather(
            -1, regime_class.unsqueeze(-1)
        ).squeeze(-1)

        soft_da = base_level + 0.5 * soft_spread
        soft_rt = base_level - 0.5 * soft_spread
        deployment_spread = base_spread.clone()
        base_da_high = base_regime == DA_HIGH
        deployment_spread[base_da_high] = torch.maximum(
            base_spread[base_da_high], da_high_spread[base_da_high]
        )
        hard_da = base_da_price
        hard_rt = base_da_price - deployment_spread
        return {
            "base_da_price": base_da_price,
            "base_rt_at_da_price": base_rt_price,
            "base_price_level": base_level,
            "base_spread": base_spread,
            "base_regime_class": base_regime,
            "regime_logits": regime_logits,
            "regime_probability": regime_probability,
            "regime_straight_through_weights": straight_through_weights,
            "regime_class": regime_class,
            "conditional_excess": conditional_excess,
            "state_spreads": state_spreads,
            "soft_spread": soft_spread,
            "hard_spread": deployment_spread,
            "factorized_spread": hard_spread,
            "soft_da_price": soft_da,
            "soft_rt_at_da_price": soft_rt,
            "hard_da_price": hard_da,
            "hard_rt_at_da_price": hard_rt,
            "da_price_adjustment": hard_da - base_da_price,
            "rt_at_da_price_adjustment": hard_rt - base_rt_price,
        }


def economic_spread_factorization_loss(
    output: dict[str, torch.Tensor],
    true_da_price: torch.Tensor,
    true_rt_at_da_price: torch.Tensor,
    *,
    threshold: float = 6.0,
    price_scale: torch.Tensor | float | None = None,
    class_weight: torch.Tensor | None = None,
) -> EconomicSpreadLoss:
    """计算预注册的等权、量纲归一化三分量损失。"""
    if true_da_price.shape != true_rt_at_da_price.shape:
        raise ValueError("DA与RT-at-DA真值必须同形")
    if true_da_price.shape != output["soft_spread"].shape:
        raise ValueError("真值形状必须与模型小时输出一致")
    if threshold <= 0:
        raise ValueError("threshold必须为正")

    true_spread = true_da_price - true_rt_at_da_price
    target_regime = economic_spread_regime(true_spread, threshold)
    regime_loss = F.cross_entropy(
        output["regime_logits"].reshape(-1, 3),
        target_regime.reshape(-1),
        weight=class_weight,
    )

    target_excess = conditional_excess_target(true_spread, threshold)
    tradeable = target_regime != NEUTRAL
    selected_index = torch.where(
        target_regime == DA_HIGH,
        torch.zeros_like(target_regime),
        torch.ones_like(target_regime),
    )
    selected_excess = output["conditional_excess"].gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    if bool(tradeable.any()):
        magnitude_loss = F.smooth_l1_loss(
            selected_excess[tradeable] / threshold,
            target_excess[tradeable] / threshold,
        )
    else:
        magnitude_loss = selected_excess.sum() * 0.0

    if price_scale is None:
        scale = torch.full_like(true_da_price, float(threshold))
    else:
        scale = torch.as_tensor(
            price_scale,
            dtype=true_da_price.dtype,
            device=true_da_price.device,
        )
        while scale.ndim < true_da_price.ndim:
            scale = scale.unsqueeze(-1)
        scale = torch.broadcast_to(scale, true_da_price.shape).clamp_min(1.0e-6)
    reconstruction_loss = 0.5 * (
        F.smooth_l1_loss(
            output["soft_da_price"] / scale, true_da_price / scale
        )
        + F.smooth_l1_loss(
            output["soft_rt_at_da_price"] / scale,
            true_rt_at_da_price / scale,
        )
    )
    total = regime_loss + magnitude_loss + reconstruction_loss
    return EconomicSpreadLoss(
        total=total,
        regime=regime_loss,
        conditional_excess=magnitude_loss,
        reconstruction=reconstruction_loss,
        tradeable_count=int(tradeable.sum().detach().cpu()),
    )


class EconomicSpreadFactorizedTransformer(nn.Module):
    """包装两套现有Transformer，并用联合分解头还原DA/RT-at-DA。"""

    def __init__(
        self,
        da_model: nn.Module,
        rt_at_da_model: nn.Module,
        *,
        d_model: int,
        threshold: float = 6.0,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.da_model = da_model
        self.rt_at_da_model = rt_at_da_model
        self.factorization_head = EconomicSpreadFactorizationHead(
            d_model,
            threshold=threshold,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def set_trainable_scope(self, scope: str) -> None:
        """冻结训练范围；正式首轮固定使用decoder_head。"""
        normalized = scope.lower().strip()
        if normalized not in {"head", "decoder_head", "all"}:
            raise ValueError("scope必须是head、decoder_head或all")
        for parameter in self.parameters():
            parameter.requires_grad = normalized == "all"
        for parameter in self.factorization_head.parameters():
            parameter.requires_grad = True
        if normalized == "decoder_head":
            for model in (self.da_model, self.rt_at_da_model):
                for module_name in ("decoder", "price_head"):
                    module = getattr(model, module_name, None)
                    if module is None:
                        raise AttributeError(f"基础模型缺少{module_name}")
                    for parameter in module.parameters():
                        parameter.requires_grad = True

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        da_output = self.da_model(batch)
        rt_output = self.rt_at_da_model(batch)
        factorized = self.factorization_head(
            da_output["rep"],
            rt_output["rep"],
            da_output["p_da"],
            rt_output["p_rt_at_da"],
        )
        factorized["da_representation"] = da_output["rep"]
        factorized["rt_at_da_representation"] = rt_output["rep"]
        return factorized

