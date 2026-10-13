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
from .entity_conditioning import EntityConditioner
from .model import QueryDecoder, StreamEncoder, TransformerBlock
from .scarcity_decomposition import (
    PhysicalScarcityResidual,
    canonical_output_decomposition,
)


DA_STREAM_DIMS = {
    "price_da": 1,
    "price_rt": 1,
    "load": 1,
    "system": 2,
    "cal": 8,
}


def canonical_residual_anchor(value: str) -> str:
    """规范化Global价格残差锚点，并拒绝静默拼写错误。"""
    normalized = str(value).lower().strip()
    aliases = {"none": "none", "": "none", "lag168": "lag168"}
    if normalized not in aliases:
        raise ValueError("global_residual_anchor必须是none或lag168")
    return aliases[normalized]


def anchored_normalized_price(
    residual: torch.Tensor,
    batch: dict,
    *,
    anchor_mode: str,
    anchor_key: str,
    mean_key: str,
    std_key: str,
) -> torch.Tensor:
    """把网络输出解释为标准化残差，并加回合法历史锚点。"""
    if anchor_mode == "none":
        return residual
    if anchor_key not in batch:
        raise KeyError(f"启用lag168锚点时batch缺少{anchor_key}")
    mean = batch[mean_key].reshape(-1, 1)
    std = batch[std_key].reshape(-1, 1)
    anchor = batch[anchor_key].float()
    return residual + (anchor - mean) / std


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
        self.entity_conditioning_mode = str(
            getattr(cfg, "global_entity_conditioning_mode", "post_fusion")
        ).lower()
        if self.entity_conditioning_mode not in {"post_fusion", "pre_encoder"}:
            raise ValueError(
                "global_entity_conditioning_mode必须是post_fusion或pre_encoder"
            )
        self.residual_anchor = canonical_residual_anchor(
            getattr(cfg, "global_residual_anchor", "none")
        )
        self.output_decomposition = canonical_output_decomposition(
            getattr(cfg, "global_output_decomposition", "direct")
        )
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
        self.scarcity_residual = (
            PhysicalScarcityResidual(d_model)
            if self.output_decomposition == "physical_scarcity_residual"
            else None
        )
        if self.residual_anchor == "lag168":
            # 从接近季节锚点的位置开始，而不是一开始随机破坏它。
            nn.init.normal_(self.price_head.weight, std=1.0e-3)
            nn.init.zeros_(self.price_head.bias)
        self.entity_conditioner = (
            EntityConditioner(
                d_model=d_model,
                num_nodes=cfg.global_num_nodes,
                num_markets=cfg.global_num_markets,
            )
            if cfg.global_use_entity_embeddings
            else None
        )

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
        identity = None
        if (
            self.entity_conditioner is not None
            and self.entity_conditioning_mode == "pre_encoder"
        ):
            identity = self.entity_conditioner.identity(
                batch, inputs["price_da"].shape[0], inputs["price_da"].device
            )
        encoded = {
            name: self.encoders[name](inputs[name], conditioning=identity)
            for name in self.stream_names
        }
        memory, source_weights = self.fusion(encoded)
        calendar_context = self.target_calendar(batch["cal_tgt"])
        if self.entity_conditioner is not None:
            if self.entity_conditioning_mode == "post_fusion":
                memory, calendar_context = self.entity_conditioner(
                    batch, memory, calendar_context
                )
            else:
                calendar_context = calendar_context + identity.unsqueeze(1)
        representation = self.decoder(memory, query_context=calendar_context)
        return representation, memory, source_weights

    def forward(self, batch: dict) -> dict:
        representation, memory, source_weights = self._encode_and_decode(batch)
        normalized_residual = self.price_head(representation).squeeze(-1)
        base_normalized_price = anchored_normalized_price(
            normalized_residual,
            batch,
            anchor_mode=self.residual_anchor,
            anchor_key="price_da_lag168",
            mean_key="price_da_mean",
            std_key="price_da_std",
        )
        scarcity = None
        normalized_price = base_normalized_price
        if self.scarcity_residual is not None:
            scarcity = self.scarcity_residual(representation, batch)
            normalized_price = (
                base_normalized_price + scarcity["normalized_residual"]
            )

        mean = batch["price_da_mean"].reshape(-1, 1)
        std = batch["price_da_std"].reshape(-1, 1)
        price = normalized_price * std + mean
        lo, hi = self.cfg.pred_clamp
        price = price.clamp(float(lo), float(hi))
        result = {
            "p_da": price,
            "p_da_normalized": normalized_price,
            "p_da_normalized_residual": normalized_residual,
            "rep": representation,
            "memory": memory,
            "source_weights": source_weights,
        }
        if scarcity is not None:
            result.update({
                "p_da_base_normalized": base_normalized_price,
                "p_da_scarcity_normalized_residual": scarcity["normalized_residual"],
                "scarcity_gate": scarcity["gate"],
                "scarcity_pressure_features": scarcity["pressure_features"],
                "scarcity_monotone_pressure_weights": scarcity[
                    "monotone_pressure_weights"
                ],
            })
        return result
