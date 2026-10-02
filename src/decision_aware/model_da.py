"""独立日前（DA）Transformer及F0/F1/F1b/F2五来源融合候选。

F0是强残差MLP基线，F1在同一小时内做来源注意力后再做时间注意力，
F1b保留F1的来源注意力、但用concat投影和残差MLP替代softmax加权汇聚；
F2保留全部来源×时间token做全局注意力。F2显式加入独立的来源和时间
embedding，并关闭全局层的一维RoPE，避免把840个token误当成一条时间轴。
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


class ResidualMLPBlock(nn.Module):
    """不含注意力的pre-LN残差MLP。"""

    def __init__(self, d_model: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.ffn(self.norm(value))


class SourceAttentionConcatFusion(nn.Module):
    """F1b：来源注意力后保留全部五个token，再做concat残差融合。

    它与F1唯一关键差异是移除标量softmax加权求和，避免五个来源过早压成
    一个加权平均。来源注意力、时间Transformer、来源/时间身份和输出宽度
    均保持不变。只使用一层残差MLP，使参数预算仍与F0/F1接近。
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float,
                 n_layers: int, n_sources: int):
        super().__init__()
        depth = max(1, n_layers)
        self.n_sources = n_sources
        self.modality_embedding = nn.Parameter(
            torch.randn(n_sources, d_model) * 0.02
        )
        self.source_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, dim_ff, dropout, use_rope=False
            )
            for _ in range(depth)
        ])
        self.source_norm = nn.LayerNorm(d_model)
        self.concat_norm = nn.LayerNorm(n_sources * d_model)
        self.concat_projection = nn.Linear(n_sources * d_model, d_model)
        # concat投影已经增加参数；使用半宽残差层，把F1b与F0/F1总参数差控制
        # 在10%以内，避免把额外容量误当成汇聚方式的收益。
        residual_width = max(1, d_model // 2)
        self.residual_mlp = ResidualMLPBlock(
            d_model, residual_width, dropout
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, dim_ff, dropout, use_rope=True
            )
            for _ in range(depth)
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
        memory = source_tokens.reshape(batch_size, steps, -1)
        memory = self.concat_projection(self.concat_norm(memory))
        memory = self.residual_mlp(memory)
        memory = self.fusion_norm(memory)
        for block in self.temporal_blocks:
            memory = block(memory)
        return self.temporal_norm(memory), None


class SimpleResidualFusion(nn.Module):
    """F0：同小时concat投影 + 残差MLP + 时间Transformer。

    这个基线保留各流独立编码、非线性残差容量和时间建模，只移除来源间
    attention，因此不会因为被故意做弱而让F1/F2轻易获胜。
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float,
                 n_layers: int, n_sources: int):
        super().__init__()
        depth = max(1, n_layers)
        self.n_sources = n_sources
        self.modality_embedding = nn.Parameter(
            torch.randn(n_sources, d_model) * 0.02
        )
        self.input_norm = nn.LayerNorm(n_sources * d_model)
        self.input_projection = nn.Linear(n_sources * d_model, d_model)
        self.residual_mlp = nn.ModuleList([
            ResidualMLPBlock(d_model, dim_ff, dropout) for _ in range(depth)
        ])
        self.fusion_norm = nn.LayerNorm(d_model)
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout, use_rope=True)
            for _ in range(depth)
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
        tokens = torch.stack(values, dim=2)
        tokens = tokens + self.modality_embedding.view(
            1, 1, self.n_sources, -1
        )
        batch_size, steps, _, width = tokens.shape
        memory = tokens.reshape(batch_size, steps, self.n_sources * width)
        memory = self.input_projection(self.input_norm(memory))
        for block in self.residual_mlp:
            memory = block(memory)
        memory = self.fusion_norm(memory)
        for block in self.temporal_blocks:
            memory = block(memory)
        return self.temporal_norm(memory), None


class GlobalTokenFusion(nn.Module):
    """F2：来源×时间全局注意力，使用显式二维身份而非840位置RoPE。"""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float,
                 n_layers: int, n_sources: int, max_steps: int):
        super().__init__()
        self.n_sources = n_sources
        self.max_steps = max_steps
        self.modality_embedding = nn.Parameter(
            torch.randn(n_sources, d_model) * 0.02
        )
        self.time_embedding = nn.Parameter(torch.randn(max_steps, d_model) * 0.02)
        # F1每层包含一个来源block和一个时间block。F2使用2倍层数，使可训练
        # block数量和参数预算近似一致；全局层明确关闭RoPE。
        self.global_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout, use_rope=False)
            for _ in range(2 * max(1, n_layers))
        ])
        self.final_norm = nn.LayerNorm(d_model)

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
        if steps > self.max_steps:
            raise ValueError(
                f"F2时间长度{steps}超过time_embedding上限{self.max_steps}"
            )
        tokens = torch.stack(values, dim=2)  # [B,T,S,D]
        tokens = tokens + self.modality_embedding.view(
            1, 1, self.n_sources, -1
        )
        tokens = tokens + self.time_embedding[:steps].view(1, steps, 1, -1)
        memory = tokens.reshape(batch_size, steps * self.n_sources, -1)
        for block in self.global_blocks:
            memory = block(memory)
        return self.final_norm(memory), None


FUSION_MODE_ALIASES = {
    "f0": "f0_residual_mlp",
    "simple_residual_mlp": "f0_residual_mlp",
    "f0_residual_mlp": "f0_residual_mlp",
    "f1": "source_attention",
    "source_attention": "source_attention",
    "f1b": "source_attention_concat",
    "source_attention_concat": "source_attention_concat",
    "f2": "f2_global_attention",
    "global_attention": "f2_global_attention",
    "f2_global_attention": "f2_global_attention",
}


def canonical_da_fusion_mode(value: str) -> str:
    try:
        return FUSION_MODE_ALIASES[value.lower()]
    except KeyError as exc:
        raise ValueError(
            f"未知DA融合方式{value!r}；允许F0/F1/F1b/F2或对应正式名称"
        ) from exc


class DecisionAwareDAForecaster(nn.Module):
    """D-1 10:00起报、一次输出交付日24小时DA价格的独立模型。"""

    stream_names = tuple(DA_STREAM_DIMS)

    def __init__(self, cfg: PilotConfig, *, fusion_mode: str | None = None):
        super().__init__()
        if cfg.horizon_da != 24:
            raise ValueError("独立DA第一版要求horizon_da=24")
        self.cfg = cfg
        self.fusion_mode = canonical_da_fusion_mode(
            fusion_mode
            if fusion_mode is not None
            else getattr(cfg, "da_fusion_mode", "source_attention")
        )
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
        fusion_kwargs = {
            "d_model": d_model,
            "n_heads": cfg.n_heads_fusion,
            "dim_ff": cfg.dim_ff,
            "dropout": cfg.dropout,
            "n_layers": cfg.n_layers_fusion,
            "n_sources": len(self.stream_names),
        }
        if self.fusion_mode == "f0_residual_mlp":
            self.fusion = SimpleResidualFusion(**fusion_kwargs)
        elif self.fusion_mode == "source_attention":
            self.fusion = SourceInteractionFusion(**fusion_kwargs)
        elif self.fusion_mode == "source_attention_concat":
            self.fusion = SourceAttentionConcatFusion(**fusion_kwargs)
        else:
            self.fusion = GlobalTokenFusion(
                **fusion_kwargs, max_steps=cfg.context_len
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
