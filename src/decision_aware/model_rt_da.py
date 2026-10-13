"""DA截止时点的独立RT价格Transformer。"""
from __future__ import annotations

import torch

from .config import PilotConfig
from .model_da import DecisionAwareDAForecaster, anchored_normalized_price


class DecisionAwareRTAtDAForecaster(DecisionAwareDAForecaster):
    """D-1 10:00一次预测交付日24个RT价格。

    该类复用DA模型的模块代码，但每次实例化都建立独立参数，checkpoint也
    独立保存。它只使用历史五流和目标日历，不读取交付日真实DA价格。
    """

    def __init__(self, cfg: PilotConfig):
        super().__init__(
            cfg,
            fusion_mode=getattr(
                cfg, "rt_at_da_fusion_mode", "source_attention"
            ),
        )
        self.parameterization = str(
            getattr(cfg, "rt_at_da_parameterization", "direct_price")
        ).lower()
        if self.parameterization not in {
            "direct_price", "da_minus_spread", "economic_regime"
        }:
            raise ValueError(
                "rt_at_da_parameterization必须是direct_price、da_minus_spread"
                "或economic_regime"
            )
        if (
            self.parameterization == "da_minus_spread"
            and self.scarcity_residual is not None
        ):
            raise ValueError(
                "显式价差参数化不能与价格稀缺残差分解混用"
            )
        self.economic_regime_head = None
        if self.parameterization == "economic_regime":
            # 额外一维是基础模型预测的标准化DA-RT价差。零初始化时argmax
            # 固定落在类别0（中性），最终价格精确保持基础模型不变。
            self.economic_regime_head = torch.nn.Linear(cfg.d_model + 1, 3)
            torch.nn.init.zeros_(self.economic_regime_head.weight)
            torch.nn.init.zeros_(self.economic_regime_head.bias)

    def forward(self, batch: dict) -> dict:
        representation, memory, source_weights = self._encode_and_decode(batch)
        normalized_residual = self.price_head(representation).squeeze(-1)
        mean = batch["price_rt_mean"].reshape(-1, 1)
        std = batch["price_rt_std"].reshape(-1, 1)
        spread = None
        latent_spread = None
        normalized_spread = None
        regime_logits = None
        regime_class = None
        base_price = None
        if self.parameterization in {"direct_price", "economic_regime"}:
            base_normalized_price = anchored_normalized_price(
                normalized_residual,
                batch,
                anchor_mode=self.residual_anchor,
                anchor_key="price_rt_lag168",
                mean_key="price_rt_mean",
                std_key="price_rt_std",
            )
            scarcity = None
            normalized_price = base_normalized_price
            if self.parameterization == "direct_price" and self.scarcity_residual is not None:
                scarcity = self.scarcity_residual(representation, batch)
                normalized_price = (
                    base_normalized_price + scarcity["normalized_residual"]
                )
            base_price = normalized_price * std + mean
            price = base_price
            if self.parameterization == "economic_regime":
                if "p_da_upstream" not in batch:
                    raise KeyError("economic_regime需要同一信息时点的p_da_upstream")
                upstream_da = batch["p_da_upstream"]
                if upstream_da.shape != base_price.shape:
                    raise ValueError("p_da_upstream必须与基础RT价格形状一致")
                da_std = batch["price_da_std"].reshape(-1, 1)
                spread_scale = torch.sqrt(da_std.square() + std.square()).clamp_min(
                    1.0e-6
                )
                base_spread = upstream_da - base_price
                normalized_base_spread = base_spread / spread_scale
                regime_input = torch.cat(
                    [representation, normalized_base_spread.unsqueeze(-1)], dim=-1
                )
                regime_logits = self.economic_regime_head(regime_input)
                regime_class = regime_logits.argmax(dim=-1)
                base_sign = torch.where(
                    base_spread >= 0,
                    torch.ones_like(base_spread),
                    -torch.ones_like(base_spread),
                )
                selected_sign = torch.where(
                    regime_class == 1,
                    torch.ones_like(base_spread),
                    torch.where(
                        regime_class == 2,
                        -torch.ones_like(base_spread),
                        base_sign,
                    ),
                )
                spread = selected_sign * base_spread.abs()
                price = upstream_da - spread
                normalized_price = (price - mean) / std.clamp_min(1.0e-6)
        else:
            if self.residual_anchor != "none":
                raise ValueError(
                    "da_minus_spread参数化不与价格lag168残差锚点混用"
                )
            if "p_da_upstream" not in batch:
                raise KeyError(
                    "da_minus_spread参数化需要同一信息时点产生的p_da_upstream"
                )
            upstream_da = batch["p_da_upstream"]
            if upstream_da.shape != normalized_residual.shape:
                raise ValueError("p_da_upstream必须与24小时价差预测形状一致")
            da_mean = batch["price_da_mean"].reshape(-1, 1)
            da_std = batch["price_da_std"].reshape(-1, 1)
            spread_center = da_mean - mean
            spread_scale = torch.sqrt(da_std.square() + std.square()).clamp_min(
                1.0e-6
            )
            normalized_spread = normalized_residual
            latent_spread = normalized_spread * spread_scale + spread_center
            price = upstream_da - latent_spread
            normalized_price = (price - mean) / std.clamp_min(1.0e-6)
        lo, hi = self.cfg.pred_clamp
        price = price.clamp(float(lo), float(hi))
        if latent_spread is not None:
            # 经济调度读取的是截断后的RT价格；对外报告的价差必须与该价格
            # 保持严格代数一致，未截断值另存用于监督诊断。
            spread = upstream_da - price
        elif self.parameterization == "economic_regime":
            spread = upstream_da - price
        result = {
            "p_rt_at_da": price,
            "p_rt_at_da_normalized": normalized_price,
            "p_rt_at_da_normalized_residual": normalized_residual,
            "rep": representation,
            "memory": memory,
            "source_weights": source_weights,
        }
        if spread is not None:
            result["p_da_minus_rt_at_da"] = spread
            result["p_da_minus_rt_at_da_unclamped"] = latent_spread
            result["p_da_minus_rt_at_da_normalized"] = normalized_spread
        if self.parameterization == "direct_price" and scarcity is not None:
            result.update({
                "p_rt_at_da_base_normalized": base_normalized_price,
                "p_rt_at_da_scarcity_normalized_residual": scarcity[
                    "normalized_residual"
                ],
                "scarcity_gate": scarcity["gate"],
                "scarcity_pressure_features": scarcity["pressure_features"],
                "scarcity_monotone_pressure_weights": scarcity[
                    "monotone_pressure_weights"
                ],
            })
        if self.parameterization == "economic_regime":
            result.update({
                "p_rt_at_da_base": base_price,
                "economic_regime_logits": regime_logits,
                "economic_regime_class": regime_class,
            })
        return result
