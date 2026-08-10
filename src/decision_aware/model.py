"""model.py — DecisionAwareTSFM（先行版架构，架构缺陷修复后）.

架构修复（对应 docs/todo3_2.md「架构缺陷」清单 + N1 final norm）：
  A1 融合层关 RoPE（位置先验反事实）
  A2 每条流加可学习 modality embedding
  B1 编码器可配多层（n_layers_enc）+ 每栈出口 final LayerNorm（pre-LN 必备，N1）
  B2 QueryDecoder 加 query 间 self-attn
  B3 query 加可学习位置编码
  B4 非 dual_split 模式 RT 用独立 query（不再复用 DA 前 4 个）
  C2 GRU 加残差连接
  C3 去掉 CrossModalFusion 冗余输入 LayerNorm（各流出口已 final_norm）
  C4 Calendar 流 linear → mlp（捕获周期性）
  C5 v3 system 流 GRU → Transformer（统一表示，C2/C6 自动失效）
  C6 GRU fp32 workaround 仅保留给 v1/v2 向后兼容

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


# ── 跨模态融合：concat N 流 token → 1 层 TransformerBlock（多头，无 RoPE）──────
# A1: 融合层关 RoPE —— 流间 token 的「物理时刻」对齐靠同时间步跨流注意力，
#     RoPE 会给「位置相近=同流」的错误先验，抑制跨模态注意力。
# A2: modality embedding 在 DecisionAwareTSFM.forward 里加（见下）。
# C3: 去掉冗余输入 LayerNorm —— 各 StreamEncoder 出口已有 final_norm，
#     且 TransformerBlock 内部 pre-LN(n1) 会再归一化，连续三次 LN 多余。
class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        # A1: use_rope=False（融合层不做位置旋转）
        # C3: 无 self.norm，直接进 block（block 内 n1 归一化）
        self.block = TransformerBlock(d_model, n_heads, dim_ff, dropout, use_rope=False)
        self.final_norm = nn.LayerNorm(d_model)   # N1: pre-LN 栈出口 final norm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N*T, d] → 全跨模态自注意力（覆盖 Weather→Load/Load→Price/... 所有对）
        return self.final_norm(self.block(tokens))


# ── Query Decoder：N learnable queries + query self-attn + cross-attn + FFN ────
# B2: cross-attn 前加 query 间 self-attn（相邻小时电价强相关，避免预测曲线跳变）
# B3: query 加可学习位置编码（否则模型须从零学会 query 顺序↔时间对应）
class QueryDecoder(nn.Module):
    """N learnable queries + query self-attn + cross-attn + FFN.

    B2: cross-attn 前加 query 间 self-attn（相邻小时电价强相关，避免预测曲线跳变；
        RT decoder 同样受益 → 24 个滚动窗口的重叠小时获得一致性结构）。
    B3: query 加可学习位置编码 + 输入条件化：
        - 位置编码 query_pos 告诉模型哪个 query 对应第几小时；
        - 条件化 ctx = Proj(mean(memory)) 让 query 依赖当前输入样本，
          不再是全局共享、与样本无关的固定槽（pretrain 阶段更快对齐 query↔时间）。
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float, n_queries: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        # B3: 可学习 query 位置编码
        self.query_pos = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        # B3: 输入条件化 —— memory 全局摘要投影到 query 偏置（[B,1,d] 广播）
        self.ctx_norm = nn.LayerNorm(d_model)
        self.ctx_proj = nn.Linear(d_model, d_model)
        # B2: query 间 self-attn（用 RoPE 给「相邻 query 更强关注」的归纳偏置，
        #     24 个 query 对应 24 小时，相对位置有意义）
        self.q_norm1 = nn.LayerNorm(d_model)
        self.q_self = RotaryMHA(d_model, n_heads, dropout, use_rope=True)
        # cross-attn：Q=queries, K/V=memory（标准 cross-attn 不对 K/V 加 RoPE）
        self.norm_q = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)   # N1: pre-LN 栈出口 final norm

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        # B3: query = 内容 query + 位置 query + 输入条件化摘要
        ctx = self.ctx_proj(self.ctx_norm(memory.mean(dim=1, keepdim=True)))   # [B, 1, d]
        q = (self.queries + self.query_pos).unsqueeze(0).expand(B, -1, -1) + ctx  # [B, N, d]
        # B2: query 间 self-attn（query 彼此感知 → 预测曲线平滑）
        q = q + self.q_self(self.q_norm1(q))
        # cross-attn：从 memory 取信息
        q2 = self.norm_q(q)
        cq, _ = self.cross(q2, memory, memory)
        q = q + cq
        q = q + self.ffn(self.n2(q))
        return self.final_norm(q)                       # [B, N, d]


# ── 各流编码器 ─────────────────────────────────────────────────────────────────
class StreamEncoder(nn.Module):
    """in_dim → d_model；按类型选 Transformer / GRU / MLP / Linear。

    B1: transformer 可配多层（n_layers）；每栈出口加 final LayerNorm（pre-LN 必备，N1）。
    C2: GRU 加残差（return x + gru(x)）。
    C4: linear 改 mlp 由 DecisionAwareTSFM 在构造时选 kind="mlp"。
    """

    def __init__(self, in_dim: int, d_model: int, kind: str, n_heads: int,
                 dim_ff: int, dropout: float, use_rope: bool = True, n_layers: int = 1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.kind = kind
        if kind == "transformer":
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, dim_ff, dropout, use_rope)
                for _ in range(max(1, n_layers))
            ])
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
        # N1: 各流出口 final LayerNorm —— 统一 transformer/gru/mlp/linear 输出尺度，
        #     供融合层 concat 时跨流方差一致（也让 C3 删融合输入 norm 安全）。
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                                     # [B, T, d]
        if self.kind == "transformer":
            for blk in self.blocks:
                x = blk(x)
        elif self.kind == "gru":
            # GRU 权重是 fp32；autocast 下上游 Linear 输出 fp16 会失配（C6 技术债，仅 v1/v2 保留）。
            out, _ = self.gru(x.float())
            x = x + out.to(x.dtype)                          # C2: GRU 残差
        elif self.kind == "mlp":
            x = x + self.mlp(x)                              # 残差 MLP
        # linear: 不做非线性变换
        return self.final_norm(x)


# ── 顶层模型 ───────────────────────────────────────────────────────────────────
class DecisionAwareTSFM(nn.Module):
    def __init__(self, cfg: PilotConfig):
        super().__init__()
        self.cfg = cfg
        self.data_version = getattr(cfg, "data_version", "v12")
        d, ff, dp = cfg.d_model, cfg.dim_ff, cfg.dropout
        he, hf = cfg.n_heads_enc, cfg.n_heads_fusion
        # B1: 编码器层数（默认 1，v7 设 2）
        n_layers = getattr(cfg, "n_layers_enc", 1)

        if self.data_version == "v3":
            # v3: 5 流（Price DA+RT / Load / System / Calendar）
            # C5: system 流 GRU → Transformer（统一表示，C2/C6 对 v3 自动失效）
            # C4: Calendar 流 linear → mlp（捕获 hour 与价格的非线性周期关系）
            self.enc_price_da  = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope, n_layers)
            self.enc_price_rt  = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope, n_layers)
            self.enc_load      = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope, n_layers)
            self.enc_system    = StreamEncoder(2, d, "transformer", he, ff, dp, cfg.use_rope, n_layers)
            self.enc_calendar  = StreamEncoder(6, d, "mlp",        he, ff, dp)
            self.n_streams = 5  # DA + RT + Load + System + Calendar
        else:
            # v1/v2: 6 流（Price / Load / Weather / System / Econ / Calendar）
            self.enc_price    = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope, n_layers)
            self.enc_load     = StreamEncoder(1, d, "transformer", he, ff, dp, cfg.use_rope, n_layers)
            self.enc_weather  = StreamEncoder(1, d, "gru",        he, ff, dp)
            self.enc_system   = StreamEncoder(7, d, "gru",        he, ff, dp)
            self.enc_econ     = StreamEncoder(4, d, "mlp",        he, ff, dp)
            self.enc_calendar = StreamEncoder(5, d, "mlp",        he, ff, dp)   # C4: linear→mlp
            self.n_streams = 6

        # A2: 每条流一个可学习 modality embedding [n_streams, d]，
        #     加到该流所有 token 上，让融合层知道 token 属于哪条流。
        self.modality_emb = nn.Parameter(torch.randn(self.n_streams, d) * 0.02)

        # C1: 融合层多头（n_heads_fusion 默认已改 4）
        self.fusion = CrossModalFusion(d, hf, ff, dp)

        # w10 §2: 日前联合预测 48h，输出 pDA + pRT|DA 两条曲线
        # use_dual_split 模式下用 48 queries + 双输出 head
        self.use_dual_split = getattr(cfg, "use_dual_split", False)
        self.horizon_rt = getattr(cfg, "horizon_rt", 4)   # w10 §2: RT 滚动 H=4
        if self.use_dual_split:
            self.horizon_joint = cfg.horizon_da * 2  # 48h
            self.decoder = QueryDecoder(d, hf, ff, dp, n_queries=self.horizon_joint)
            self.head_da = nn.Linear(d, 2)  # 输出 pDA + pRT|DA 两条曲线
            # w10 §2/§3: RT 滚动预测 — 24 个窗口各预测 H=4 小时
            # fD/fRT 结构分离（w10 §3）：RT 用独立 decoder，与 DA 解耦。
            # 注：本地简化版用同一上下文一次性出 24 窗口；完整信息截止分离
            #     （每小时用更新信息重新预测）需逐小时前向，属云 GPU 版本。
            self.rt_decoder = QueryDecoder(d, hf, ff, dp, n_queries=cfg.horizon_da)
            self.head_rt_action = nn.Linear(d, 1)
            self.head_rt_windows = nn.Linear(d, self.horizon_rt)
        else:
            # B4: RT 用独立 query —— decoder 出 horizon_da + horizon_rt 个 query，
            #     DA 用前 24 个、RT 用后 horizon_rt 个，不再共享 DA 的前 4 个 token。
            self.horizon_joint = cfg.horizon_da + self.horizon_rt
            self.decoder = QueryDecoder(d, hf, ff, dp, n_queries=self.horizon_joint)
            self.head_da = nn.Linear(d, 1)
            self.head_rt = nn.Linear(d, 1)
        self._init_heads()

    def _init_heads(self):
        heads = [self.head_da]
        if self.use_dual_split:
            heads += [self.head_rt_action, self.head_rt_windows]
        else:
            heads += [self.head_rt]
        for m in heads:
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def _fuse(self, streams: list) -> torch.Tensor:
        """A2: 给每条流加 modality embedding → concat → 融合。"""
        B = streams[0].shape[0]
        tagged = [h + self.modality_emb[s].view(1, 1, -1)
                  for s, h in enumerate(streams)]
        tokens = torch.cat(tagged, dim=1)          # [B, N*T, d]
        return self.fusion(tokens)                  # [B, N*T, d]

    def forward(self, batch: dict) -> dict:
        if self.data_version == "v3":
            # v3: 5 流
            h_da  = self.enc_price_da(batch["price_da_ctx"].unsqueeze(-1))
            h_rt  = self.enc_price_rt(batch["price_rt_ctx"].unsqueeze(-1))
            h_load = self.enc_load(batch["load_ctx"])
            h_sys  = self.enc_system(batch["system_ctx"])
            h_cal  = self.enc_calendar(batch["cal_ctx"])
            streams = [h_da, h_rt, h_load, h_sys, h_cal]
        else:
            # v1/v2: 6 流
            price = batch["price_ctx"].unsqueeze(-1)
            streams = [
                self.enc_price(price),
                self.enc_load(batch["load_ctx"]),
                self.enc_weather(batch["weather_ctx"]),
                self.enc_system(batch["system_ctx"]),
                self.enc_econ(batch["econ_ctx"]),
                self.enc_calendar(batch["cal_ctx"]),
            ]

        memory = self._fuse(streams)
        rep = self.decoder(memory)                          # [B, H_joint, d]

        lo, hi = self.cfg.pred_clamp
        if self.use_dual_split:
            # w10 §3: fD 输出两条 48h 曲线（pDA + pRT|DA）
            out_2d = self.head_da(rep)                      # [B, 48, 2]
            p_da_n = out_2d[..., 0]                         # [B, 48] pDA
            p_rt_da_n = out_2d[..., 1]                      # [B, 48] pRT|DA

            # w10 §2/§3: RT 滚动预测 — 24 窗口 × H_rt 小时
            rt_rep = self.rt_decoder(memory)                # [B, 24, d]
            p_rt_action_n = self.head_rt_action(rt_rep).squeeze(-1)        # [B, 24] 每窗口第一步
            p_rt_windows_n = self.head_rt_windows(rt_rep)                  # [B, 24, H_rt] 完整窗口

            if self.data_version == "v3":
                da_mean = batch["price_da_mean"].unsqueeze(-1)
                da_std = batch["price_da_std"].unsqueeze(-1)
                rt_mean_1d = batch["price_rt_mean"]
                rt_std_1d = batch["price_rt_std"]
                # 2D 曲线用 [..., None] 广播，3D 窗口用 [..., None, None]
                rt_mean = rt_mean_1d.unsqueeze(-1)
                rt_std = rt_std_1d.unsqueeze(-1)
                rt_mean_w = rt_mean_1d.unsqueeze(-1).unsqueeze(-1)
                rt_std_w = rt_std_1d.unsqueeze(-1).unsqueeze(-1)
                p_da = (p_da_n * da_std + da_mean).clamp(lo, hi)              # [B, 48]
                p_rt_da = (p_rt_da_n * rt_std + rt_mean).clamp(lo, hi)        # [B, 48]
                p_rt = (p_rt_action_n * rt_std + rt_mean).clamp(lo, hi)       # [B, 24]
                p_rt_w = (p_rt_windows_n * rt_std_w + rt_mean_w).clamp(lo, hi)  # [B, 24, H_rt]
            else:
                p_da = p_da_n.clamp(lo, hi)
                p_rt_da = p_rt_da_n.clamp(lo, hi)
                p_rt = p_rt_action_n.clamp(lo, hi)
                p_rt_w = p_rt_windows_n.clamp(lo, hi)
            return {"p_da": p_da, "p_rt_da": p_rt_da, "p_rt": p_rt,
                    "p_rt_windows": p_rt_w, "rep": rep}
        else:
            # B4: DA 用前 horizon_da 个 query，RT 用后 horizon_rt 个（独立 query）
            p_da_n = self.head_da(rep[:, : self.cfg.horizon_da]).squeeze(-1)        # [B, 24]
            rt_rep = rep[:, self.cfg.horizon_da : self.cfg.horizon_da + self.horizon_rt]
            p_rt_n = self.head_rt(rt_rep).squeeze(-1)                               # [B, horizon_rt]

            if self.data_version == "v3":
                da_mean = batch["price_da_mean"].unsqueeze(-1)
                da_std = batch["price_da_std"].unsqueeze(-1)
                rt_mean = batch["price_rt_mean"].unsqueeze(-1)
                rt_std = batch["price_rt_std"].unsqueeze(-1)
                p_da = (p_da_n * da_std + da_mean).clamp(lo, hi)
                p_rt = (p_rt_n * rt_std + rt_mean).clamp(lo, hi)
            else:
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
