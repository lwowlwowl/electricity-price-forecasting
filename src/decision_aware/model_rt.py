"""独立实时（RT）滚动Transformer。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import PilotConfig
from .model import QueryDecoder, StreamEncoder
from .model_da import DA_STREAM_DIMS, SourceInteractionFusion


class DecisionAwareRTForecaster(nn.Module):
    """每小时预测接下来H个RT价格；部署时只执行第一个小时动作。"""

    stream_names = tuple(DA_STREAM_DIMS)

    def __init__(self, cfg: PilotConfig):
        super().__init__()
        if cfg.horizon_rt < 1:
            raise ValueError("horizon_rt必须至少为1")
        if getattr(cfg, "rt_fusion_mode", "source_attention") != "source_attention":
            raise ValueError("独立RT模型当前只支持source_attention融合")
        d = cfg.d_model
        n_layers = max(1, cfg.n_layers_enc)
        self.cfg = cfg
        self.encoders = nn.ModuleDict({
            "price_da": StreamEncoder(1, d, "transformer", cfg.n_heads_enc,
                                      cfg.dim_ff, cfg.dropout, cfg.use_rope, n_layers),
            "price_rt": StreamEncoder(1, d, "transformer", cfg.n_heads_enc,
                                      cfg.dim_ff, cfg.dropout, cfg.use_rope, n_layers),
            "load": StreamEncoder(1, d, "transformer", cfg.n_heads_enc,
                                  cfg.dim_ff, cfg.dropout, cfg.use_rope, n_layers),
            "system": StreamEncoder(2, d, "transformer", cfg.n_heads_enc,
                                    cfg.dim_ff, cfg.dropout, cfg.use_rope, n_layers),
            "cal": StreamEncoder(8, d, "mlp", cfg.n_heads_enc,
                                 cfg.dim_ff, cfg.dropout, cfg.use_rope, n_layers),
        })
        self.fusion = SourceInteractionFusion(
            d, cfg.n_heads_fusion, cfg.dim_ff, cfg.dropout,
            cfg.n_layers_fusion, len(self.stream_names),
        )
        # RT起报时可合法使用目标小时的已公布DA价和确定的日历。
        self.target_context = nn.Sequential(
            nn.Linear(9, d), nn.GELU(), nn.LayerNorm(d)
        )
        self.decoder = QueryDecoder(
            d, cfg.n_heads_enc, cfg.dim_ff, cfg.dropout,
            cfg.horizon_rt, use_memory_context=False,
        )
        self.price_head = nn.Linear(d, 1)

    @staticmethod
    def _as_3d(value, name):
        if value.ndim == 2:
            value = value.unsqueeze(-1)
        if value.ndim != 3:
            raise ValueError(f"{name}形状错误: {tuple(value.shape)}")
        return value

    def forward(self, batch: dict):
        inputs = {
            "price_da": self._as_3d(batch["price_da_ctx"], "price_da_ctx"),
            "price_rt": self._as_3d(batch["price_rt_ctx"], "price_rt_ctx"),
            "load": self._as_3d(batch["load_ctx"], "load_ctx"),
            "system": self._as_3d(batch["system_ctx"], "system_ctx"),
            "cal": self._as_3d(batch["cal_ctx"], "cal_ctx"),
        }
        encoded = {name: self.encoders[name](inputs[name]) for name in self.stream_names}
        memory, source_weights = self.fusion(encoded)
        known_future = torch.cat([
            batch["price_da_tgt_known"].unsqueeze(-1), batch["cal_tgt"]
        ], dim=-1)
        query_context = self.target_context(known_future)
        representation = self.decoder(memory, query_context=query_context)
        normalized = self.price_head(representation).squeeze(-1)
        price = normalized * batch["price_rt_std"].reshape(-1, 1)
        price = price + batch["price_rt_mean"].reshape(-1, 1)
        lo, hi = self.cfg.pred_clamp
        price = price.clamp(float(lo), float(hi))
        return {
            "p_rt": price,
            "p_rt_normalized": normalized,
            "rep": representation,
            "memory": memory,
            "source_weights": source_weights,
        }
