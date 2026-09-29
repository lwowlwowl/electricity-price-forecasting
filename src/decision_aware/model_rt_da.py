"""DA截止时点的独立RT价格Transformer。"""
from __future__ import annotations

from .config import PilotConfig
from .model_da import DecisionAwareDAForecaster


class DecisionAwareRTAtDAForecaster(DecisionAwareDAForecaster):
    """D-1 10:00一次预测交付日24个RT价格。

    该类复用DA模型的模块代码，但每次实例化都建立独立参数，checkpoint也
    独立保存。它只使用历史五流和目标日历，不读取交付日真实DA价格。
    """

    def __init__(self, cfg: PilotConfig):
        if getattr(cfg, "rt_at_da_fusion_mode", "source_attention") != "source_attention":
            raise ValueError("RT-at-DA模型当前只支持source_attention融合")
        super().__init__(cfg)

    def forward(self, batch: dict) -> dict:
        representation, memory, source_weights = self._encode_and_decode(batch)
        normalized_price = self.price_head(representation).squeeze(-1)
        mean = batch["price_rt_mean"].reshape(-1, 1)
        std = batch["price_rt_std"].reshape(-1, 1)
        price = normalized_price * std + mean
        lo, hi = self.cfg.pred_clamp
        price = price.clamp(float(lo), float(hi))
        return {
            "p_rt_at_da": price,
            "p_rt_at_da_normalized": normalized_price,
            "rep": representation,
            "memory": memory,
            "source_weights": source_weights,
        }
