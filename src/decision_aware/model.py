"""model.py — DecisionAwareTSFM（先行版架构）.

严格对应 docs/model_architecture_v2.drawio「先行版-模型架构」：
  6 混合编码器 (Price/Load=Transformer+RoPE, Weather/System=GRU, Econ=MLP, Calendar=Linear)
  → Cross-Attn 融合 (1 层 single-head) → Shared Memory
  → 24 Learnable Queries + Cross-Attn Decoder → Forecast Representation
  → Head_DA (24 点) / Head_RT (前 6 点)，共享 Decoder 分叉 Head（todo3 #9）。

输出为真实电价尺度（用 batch 的 price_mean/std 反归一化）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PilotConfig


# ── RoPE（轻量自实现，与正式版对齐；无外部依赖）──────────────────────────────
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """预计算 cos/sin 表，apply 时对 [..., d] 张量做旋转。"""

    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_len).float()
        freqs = torch.outer(pos, inv_freq)                    # [T, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)               # [T, d]
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, T, d] → 应用 RoPE（按 T 取 cos/sin）。"""
        T = x.shape[2]
        return x * self.cos[:, :, :T, :] + _rotate_half(x) * self.sin[:, :, :T, :]


# ── 带 RoPE 的多头自注意力（用 SDPA，batch_first）──────────────────────────────
class RotaryMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model={d_model} 不能被 n_heads={num_heads} 整除"
        self.h = num_heads
        self.dh = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.dh) if use_rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # [B, H, T, dh]
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.out(o)


class TransformerBlock(nn.Module):
    """pre-LN: x = x + attn(LN(x)); x = x + ffn(LN(x))."""

    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float, use_rope: bool = True):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = RotaryMHA(d_model, num_heads, dropout, use_rope)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


# ── 跨模态融合：concat 6 流 token → 1 层 TransformerBlock (single-head) ─────────
class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = TransformerBlock(d_model, n_heads, dim_ff, dropout, use_rope=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, 6*T, d] → 全跨模态自注意力（覆盖 Weather→Load/Load→Price/... 所有对）
        return self.block(self.norm(tokens))


# ── Query Decoder：24 learnable queries + cross-attn + FFN ─────────────────────
class QueryDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float, n_queries: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.n1 = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.norm_q = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)       # [B, 24, d]
        q2 = self.norm_q(q)
        cq, _ = self.cross(q2, memory, memory)               # Q=queries, K/V=memory
        q = q + cq
        q = q + self.ffn(self.n2(q))
        return q                                              # [B, 24, d]


# ── 各流编码器 ─────────────────────────────────────────────────────────────────
class StreamEncoder(nn.Module):
    """in_dim → d_model；按类型选 Transformer / GRU / MLP / Linear。"""

    def __init__(self, in_dim: int, d_model: int, kind: str, n_heads: int,
                 dim_ff: int, dropout: float, use_rope: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.kind = kind
        if kind == "transformer":
            self.block = TransformerBlock(d_model, n_heads, dim_ff, dropout, use_rope)
        elif kind == "gru":
            self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        elif kind == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
            )
        elif kind == "linear":
            self.block = None
        else:
            raise ValueError(f"未知 encoder kind: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                                     # [B, T, d]
        if self.kind == "transformer":
            return self.block(x)
        if self.kind == "gru":
            # GRU 权重是 fp32；autocast 下上游 Linear 输出 fp16 会失配。
            # 局部转 fp32 跑 GRU，再转回 autocast dtype，保持 AMP 对 Transformer 部分生效。
            out, _ = self.gru(x.float())
            return out.to(x.dtype)
        if self.kind == "mlp":
            return x + self.mlp(x)                           # 残差 MLP
        return x                                             # linear


# ── 顶层模型 ───────────────────────────────────────────────────────────────────
class DecisionAwareTSFM(nn.Module):
    def __init__(self, cfg: PilotConfig):
        super().__init__()
        self.cfg = cfg
        self.data_version = getattr(cfg, "data_version", "v12")
        d, ff, dp = cfg.d_model, cfg.dim_ff, cfg.dropout
        he, hf = cfg.n_heads_enc, cfg.n_heads_fusion

        if self.data_version == "v3":
            # v3: 4 流（Price DA+RT / Load / System / Calendar）
            self.enc_price_da  = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope)
            self.enc_price_rt  = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope)
            self.enc_load      = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope)
            self.enc_system    = StreamEncoder(2, d, "gru",        he, ff, dp)
            self.enc_calendar  = StreamEncoder(6, d, "linear",     he, ff, dp)
            self.n_streams = 5  # DA + RT + Load + System + Calendar
        else:
            # v1/v2: 6 流（Price / Load / Weather / System / Econ / Calendar）
            self.enc_price    = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope)
            self.enc_load     = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope)
            self.enc_weather  = StreamEncoder(1, d, "gru",        he, ff, dp)
            self.enc_system   = StreamEncoder(7, d, "gru",        he, ff, dp)
            self.enc_econ     = StreamEncoder(4, d, "mlp",        he, ff, dp)
            self.enc_calendar = StreamEncoder(5, d, "linear",     he, ff, dp)
            self.n_streams = 6

        self.fusion = CrossModalFusion(d, hf, ff, dp)
        self.decoder = QueryDecoder(d, hf, ff, dp, n_queries=cfg.horizon_da)
        self.head_da = nn.Linear(d, 1)
        self.head_rt = nn.Linear(d, 1)
        self._init_heads()

    def _init_heads(self):
        for m in (self.head_da, self.head_rt):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> dict:
        if self.data_version == "v3":
            # v3: 5 流
            h_da  = self.enc_price_da(batch["price_da_ctx"].unsqueeze(-1))
            h_rt  = self.enc_price_rt(batch["price_rt_ctx"].unsqueeze(-1))
            h_load = self.enc_load(batch["load_ctx"])
            h_sys  = self.enc_system(batch["system_ctx"])
            h_cal  = self.enc_calendar(batch["cal_ctx"])
            tokens = torch.cat([h_da, h_rt, h_load, h_sys, h_cal], dim=1)
        else:
            # v1/v2: 6 流
            price = batch["price_ctx"].unsqueeze(-1)
            h_price    = self.enc_price(price)
            h_load     = self.enc_load(batch["load_ctx"])
            h_weather  = self.enc_weather(batch["weather_ctx"])
            h_system   = self.enc_system(batch["system_ctx"])
            h_econ     = self.enc_econ(batch["econ_ctx"])
            h_calendar = self.enc_calendar(batch["cal_ctx"])
            tokens = torch.cat([h_price, h_load, h_weather, h_system, h_econ, h_calendar], dim=1)

        memory = self.fusion(tokens)
        rep = self.decoder(memory)                          # [B, 24, d]

        p_da_n = self.head_da(rep).squeeze(-1)              # [B, 24] normalized
        p_rt_n = self.head_rt(rep[:, : self.cfg.horizon_rt]).squeeze(-1)  # [B, 6]

        # 反归一化到真实电价尺度
        lo, hi = self.cfg.pred_clamp
        if self.data_version == "v3":
            # v3: DA 和 RT 分别用自己的 mean/std 反归一化
            da_mean = batch["price_da_mean"].unsqueeze(-1)
            da_std = batch["price_da_std"].unsqueeze(-1)
            rt_mean = batch["price_rt_mean"].unsqueeze(-1)
            rt_std = batch["price_rt_std"].unsqueeze(-1)
            p_da = (p_da_n * da_std + da_mean).clamp(lo, hi)
            p_rt = (p_rt_n * rt_std + rt_mean).clamp(lo, hi)
        else:
            # v1/v2: 共用 price mean/std
            mean = batch["price_mean"].unsqueeze(-1)
            std = batch["price_std"].unsqueeze(-1)
            p_da = (p_da_n * std + mean).clamp(lo, hi)
            p_rt = (p_rt_n * std + mean).clamp(lo, hi)
        return {"p_da": p_da, "p_rt": p_rt, "rep": rep}

    # ── checkpoint（镜像归档：torch.save state_dict，weights_only load）────────
    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, cfg: PilotConfig, path: str, map_location="cpu"):
        m = cls(cfg)
        m.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
        return m
