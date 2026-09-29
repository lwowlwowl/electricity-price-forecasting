"""独立日前（DA）Transformer。

与旧联合模型最重要的区别：五条输入流不会被线性拼成840个token。
本模型先在每个物理小时内做五来源交互，再汇聚成一个时间token，因此
Cross-Attention看到的是168个时间token，同时仍可审计五个来源的权重。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .config import PilotConfig
from .model import QueryDecoder, StreamEncoder, TransformerBlock


DA_STREAM_DIMS = {
    "price_da": 1,
    "price_rt": 1,
    "load": 1,
    "system": 2,
    "cal": 8,
}


class SourceInteractionFusion(nn.Module):
    """同一时刻内做来源交互，然后得到一条时间序列。

    输入字典中每条流都是[B,T,D]。先变为[B*T,5,D]，让五个来源在
    同一小时互相注意；之后用可学习权重汇聚来源，输出[B,T,D]。
    最后的时间Transformer只处理168个按时间排列的token。
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float,
                 n_layers: int, n_sources: int):
        super().__init__()
        self.n_sources = n_sources
        self.modality_embedding = nn.Parameter(
            torch.randn(n_sources, d_model) * 0.02
        )
        self.source_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, dim_ff, dropout, use_rope=False
            )
            for _ in range(max(1, n_layers))
        ])
        self.source_norm = nn.LayerNorm(d_model)
        self.source_score = nn.Linear(d_model, 1)
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, dim_ff, dropout, use_rope=True
            )
            for _ in range(max(1, n_layers))
        ])
        self.temporal_norm = nn.LayerNorm(d_model)

    def forward(self, encoded: Dict[str, torch.Tensor]):
        values = list(encoded.values())
        if len(values) != self.n_sources:
            raise ValueError(
                f"期望{self.n_sources}条输入流，实际收到{len(values)}条"
            )
        shapes = {tuple(value.shape[:2]) for value in values}
        if len(shapes) != 1:
            raise ValueError(f"五条流的[B,T]必须一致，实际为{sorted(shapes)}")
        batch_size, steps = values[0].shape[:2]
        # [B,T,S,D]：S维明确表示来源，而不是把来源排进时间维。
        source_tokens = torch.stack(values, dim=2)
        source_tokens = source_tokens + self.modality_embedding.view(
            1, 1, self.n_sources, -1
        )
        source_tokens = source_tokens.reshape(
            batch_size * steps, self.n_sources, -1
        )
        for block in self.source_blocks:
            source_tokens = block(source_tokens)
        source_tokens = self.source_norm(source_tokens)
        weights = torch.softmax(self.source_score(source_tokens), dim=1)
        memory = (source_tokens * weights).sum(dim=1).reshape(
            batch_size, steps, -1
        )
        for block in self.temporal_blocks:
            memory = block(memory)
        memory = self.temporal_norm(memory)
        source_weights = weights.reshape(batch_size, steps, self.n_sources)
        return memory, source_weights


class DecisionAwareDAForecaster(nn.Module):
    """D-1 10:00起报、一次输出交付日24小时DA价格的独立模型。"""

    stream_names = tuple(DA_STREAM_DIMS)

    def __init__(self, cfg: PilotConfig):
        super().__init__()
        if cfg.horizon_da != 24:
            raise ValueError("独立DA第一版要求horizon_da=24")
        if getattr(cfg, "da_fusion_mode", "source_attention") != "source_attention":
            raise ValueError("独立DA模型当前只支持source_attention融合")

        self.cfg = cfg
        d_model = cfg.d_model
        n_layers_enc = max(1, cfg.n_layers_enc)
        self.encoders = nn.ModuleDict({
            "price_da": StreamEncoder(
                1, d_model, "transformer", cfg.n_heads_enc, cfg.dim_ff,
                cfg.dropout, cfg.use_rope, n_layers_enc,
            ),
            "price_rt": StreamEncoder(
                1, d_model, "transformer", cfg.n_heads_enc, cfg.dim_ff,
                cfg.dropout, cfg.use_rope, n_layers_enc,
            ),
            "load": StreamEncoder(
                1, d_model, "transformer", cfg.n_heads_enc, cfg.dim_ff,
                cfg.dropout, cfg.use_rope, n_layers_enc,
            ),
            "system": StreamEncoder(
                2, d_model, "transformer", cfg.n_heads_enc, cfg.dim_ff,
                cfg.dropout, cfg.use_rope, n_layers_enc,
            ),
            "cal": StreamEncoder(
                8, d_model, "mlp", cfg.n_heads_enc, cfg.dim_ff,
                cfg.dropout, cfg.use_rope, n_layers_enc,
            ),
        })
        self.fusion = SourceInteractionFusion(
            d_model=d_model,
            n_heads=cfg.n_heads_fusion,
            dim_ff=cfg.dim_ff,
            dropout=cfg.dropout,
            n_layers=cfg.n_layers_fusion,
            n_sources=len(self.stream_names),
        )
        self.target_calendar = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.decoder = QueryDecoder(
            d_model=d_model,
            n_heads=cfg.n_heads_enc,
            dim_ff=cfg.dim_ff,
            dropout=cfg.dropout,
            n_queries=cfg.horizon_da,
            use_memory_context=False,
        )
        self.price_head = nn.Linear(d_model, 1)

    @staticmethod
    def _three_dimensional(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(-1)
        if value.ndim != 3:
            raise ValueError(f"{name}必须是[B,T]或[B,T,C]，实际为{tuple(value.shape)}")
        return value

    def _encode_and_decode(self, batch: dict):
        """共享日级骨干；DA与RT-at-DA实例只共享代码，不共享权重。"""
        inputs = {
            "price_da": self._three_dimensional(batch["price_da_ctx"], "price_da_ctx"),
            "price_rt": self._three_dimensional(batch["price_rt_ctx"], "price_rt_ctx"),
            "load": self._three_dimensional(batch["load_ctx"], "load_ctx"),
            "system": self._three_dimensional(batch["system_ctx"], "system_ctx"),
            "cal": self._three_dimensional(batch["cal_ctx"], "cal_ctx"),
        }
        encoded = {
            name: self.encoders[name](inputs[name]) for name in self.stream_names
        }
        memory, source_weights = self.fusion(encoded)
        calendar_context = self.target_calendar(batch["cal_tgt"])
        representation = self.decoder(memory, query_context=calendar_context)
        return representation, memory, source_weights

    def forward(self, batch: dict) -> dict:
        representation, memory, source_weights = self._encode_and_decode(batch)
        normalized_price = self.price_head(representation).squeeze(-1)

        mean = batch["price_da_mean"].reshape(-1, 1)
        std = batch["price_da_std"].reshape(-1, 1)
        price = normalized_price * std + mean
        lo, hi = self.cfg.pred_clamp
        price = price.clamp(float(lo), float(hi))
        return {
            "p_da": price,
            "p_da_normalized": normalized_price,
            "rep": representation,
            "memory": memory,
            "source_weights": source_weights,
        }
