"""独立实时（RT）滚动Transformer。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import PilotConfig
from .entity_conditioning import EntityConditioner
from .model import QueryDecoder, StreamEncoder
from .model_da import (
    DA_STREAM_DIMS,
    GlobalTokenFusion,
    SimpleResidualFusion,
    SourceAttentionConcatFusion,
    SourceInteractionFusion,
    canonical_da_fusion_mode,
    canonical_residual_anchor,
    anchored_normalized_price,
)
from .scarcity_decomposition import (
    PhysicalScarcityResidual,
    canonical_output_decomposition,
)


class DecisionAwareRTForecaster(nn.Module):
    """每小时预测接下来H个RT价格；部署时只执行第一个小时动作。"""

    stream_names = tuple(DA_STREAM_DIMS)

    def __init__(self, cfg: PilotConfig):
        super().__init__()
        if cfg.horizon_rt < 1:
            raise ValueError("horizon_rt必须至少为1")
        d = cfg.d_model
        n_layers = max(1, cfg.n_layers_enc)
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
            getattr(cfg, "rt_fusion_mode", "source_attention")
        )
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
        fusion_kwargs = {
            "d_model": d,
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
        # RT起报时可合法使用目标小时的已公布DA价和确定的日历。
        self.target_context = nn.Sequential(
            nn.Linear(9, d), nn.GELU(), nn.LayerNorm(d)
        )
        self.decoder = QueryDecoder(
            d, cfg.n_heads_enc, cfg.dim_ff, cfg.dropout,
            cfg.horizon_rt, use_memory_context=False,
        )
        self.price_head = nn.Linear(d, 1)
        self.scarcity_residual = (
            PhysicalScarcityResidual(d)
            if self.output_decomposition == "physical_scarcity_residual"
            else None
        )
        if self.residual_anchor == "lag168":
            nn.init.normal_(self.price_head.weight, std=1.0e-3)
            nn.init.zeros_(self.price_head.bias)
        self.entity_conditioner = (
            EntityConditioner(
                d_model=d,
                num_nodes=cfg.global_num_nodes,
                num_markets=cfg.global_num_markets,
            )
            if cfg.global_use_entity_embeddings
            else None
        )

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
        known_future = torch.cat([
            batch["price_da_tgt_known"].unsqueeze(-1), batch["cal_tgt"]
        ], dim=-1)
        query_context = self.target_context(known_future)
        if self.entity_conditioner is not None:
            if self.entity_conditioning_mode == "post_fusion":
                memory, query_context = self.entity_conditioner(
                    batch, memory, query_context
                )
            else:
                query_context = query_context + identity.unsqueeze(1)
        representation = self.decoder(memory, query_context=query_context)
        normalized_residual = self.price_head(representation).squeeze(-1)
        base_normalized = anchored_normalized_price(
            normalized_residual,
            batch,
            anchor_mode=self.residual_anchor,
            anchor_key="price_rt_lag168",
            mean_key="price_rt_mean",
            std_key="price_rt_std",
        )
        scarcity = None
        normalized = base_normalized
        if self.scarcity_residual is not None:
            scarcity = self.scarcity_residual(representation, batch)
            normalized = base_normalized + scarcity["normalized_residual"]
        price = normalized * batch["price_rt_std"].reshape(-1, 1)
        price = price + batch["price_rt_mean"].reshape(-1, 1)
        lo, hi = self.cfg.pred_clamp
        price = price.clamp(float(lo), float(hi))
        result = {
            "p_rt": price,
            "p_rt_normalized": normalized,
            "p_rt_normalized_residual": normalized_residual,
            "rep": representation,
            "memory": memory,
            "source_weights": source_weights,
        }
        if scarcity is not None:
            result.update({
                "p_rt_base_normalized": base_normalized,
                "p_rt_scarcity_normalized_residual": scarcity[
                    "normalized_residual"
                ],
                "scarcity_gate": scarcity["gate"],
                "scarcity_pressure_features": scarcity["pressure_features"],
                "scarcity_monotone_pressure_weights": scarcity[
                    "monotone_pressure_weights"
                ],
            })
        return result
