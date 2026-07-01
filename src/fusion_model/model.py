"""
model.py — ElecFM 融合模型
============================
架构：TimesFM-2.5 骨干（20→15 层，移除最安全的 5 层）+ SpikeHeadV2

参考：fusion_model_design_v3.md Section 2 & 3
Step 1 验证结果：8 层方案退化 +18%（失败），5 层保守方案退化 +4.4%（通过）

层对应关系（新编号 → 原 TimesFM 编号）：
  新L0→原L0  新L1→原L1  新L2→原L2  新L3→原L3  新L4→原L4
  新L5→原L5  新L6→原L7* 新L7→原L10 新L8→原L11 新L9→原L12
  新L10→原L14 新L11→原L16 新L12→原L17 新L13→原L18 新L14→原L19
  * 新L6（原L7）是"尖峰检测层"，ΔSpike-F1 = −6.7%，spike head 在此分叉

移除的层：{L6, L8, L9, L13, L15}（独立移除时 |ΔSMAPE| < 1%）

环境要求：external/timesfm/.venv（PyTorch 2.x，含 timesfm 包）
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 本文件在 timesfm venv 下运行，直接 import timesfm ──────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import timesfm
from timesfm.torch import util as tfm_util

from spike_head import SpikeHeadV2   # 同目录

# ── 剪枝配置（保守方案：仅移除独立测试时 |ΔSMAPE|<1% 的最安全 5 层）──────────
# 原计划移除 8 层，但 Step 1 验证发现累积退化 +18%（超过 10% 阈值）
# 保守方案：只移除 {L6(+0.6%), L8(≈0%), L9(+0.8%), L13(+0.3%), L15(+0.8%)}
LAYERS_TO_REMOVE = {6, 8, 9, 13, 15}              # 原 TimesFM 层号（0-indexed）
# 剪枝后保留 15 层: [0,1,2,3,4,5,7,10,11,12,14,16,17,18,19]
# 原 L7（spike 关键层）是新模型第 6 位（0-indexed）
SPIKE_LAYER_IDX  = 6   # 剪枝后新模型中的层号（= 原 L7，尖峰检测层）

# TimesFM-2.5 内部常量（从 model.p / model.o / model.os / model.q 读取）
PATCH_SIZE    = 32     # m.p
OUTPUT_PATCH  = 128    # m.o   （点预测输出步数）
QUANTILE_OS   = 1024   # m.os  （分位数输出步数）
N_Q           = 10     # m.q   （每步的分位数维度）
D_MODEL       = 1280   # 隐藏维度


class ElecFM(nn.Module):
    """
    ElecFM：电价预测融合模型。

    输入：单节点电价时序 [batch, context_len]（原始值，未归一化）
    输出：
      quant_pred   : [batch, horizon, 9]  分位数预测 q0.1…q0.9（原始价格空间）
      spike_logits : [batch, horizon]     尖峰 logits（未经 sigmoid）
    """

    def __init__(
        self,
        pretrained_id: str = "google/timesfm-2.5-200m-pytorch",
        horizon: int = 24,
        spike_head_hidden: int = 256,
        spike_head_dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon

        # ── 1. 加载 TimesFM（禁用 compile，保持动态图以支持层修改和反向传播）──
        tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            pretrained_id, torch_compile=False)
        module = tfm.model   # TimesFM_2p5_200M_torch_module

        # ── 2. 剪枝：移除 5 个最安全层，保留 15 层 ───────────────────────────
        kept = [(i, layer) for i, layer in enumerate(module.stacked_xf)
                if i not in LAYERS_TO_REMOVE]
        self._original_layer_indices = [i for i, _ in kept]   # 调试用

        # 验证 SPIKE_LAYER_IDX 对应的原始层是 L7
        orig_idx_at_spike = self._original_layer_indices[SPIKE_LAYER_IDX]
        assert orig_idx_at_spike == 7, (
            f"Spike layer idx 映射错误：新 L{SPIKE_LAYER_IDX} = 原 L{orig_idx_at_spike}，"
            f"期望原 L7。LAYERS_TO_REMOVE={LAYERS_TO_REMOVE}，"
            f"保留层顺序={self._original_layer_indices}"
        )

        # ── 3. 注册组件到 ElecFM ────────────────────────────────────────────
        self.tokenizer    = module.tokenizer
        self.layers       = nn.ModuleList([layer for _, layer in kept])
        self.quant_head   = module.output_projection_quantiles   # ResidualBlock(1280→10240)
        self.spike_head   = SpikeHeadV2(D_MODEL, spike_head_hidden, horizon, spike_head_dropout)

        # 释放原始模型（避免重复保留内存）
        del module, tfm

    # ── 前向传播辅助方法 ──────────────────────────────────────────────────────

    def _make_patches(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        把原始序列切成 32 步的 patch。
        左侧用零值 padding 补齐到 PATCH_SIZE 的整数倍。

        Returns
        -------
        inputs : [B, n_patches, PATCH_SIZE]   原始值（未归一化）
        masks  : [B, n_patches, PATCH_SIZE]   bool，True = padding/缺失
        """
        B, T = x.shape
        pad_len = (PATCH_SIZE - T % PATCH_SIZE) % PATCH_SIZE
        if pad_len:
            x = F.pad(x, (pad_len, 0), value=0.0)
            mask_prefix = torch.ones(B, pad_len, dtype=torch.bool, device=x.device)
            mask_suffix = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            mask = torch.cat([mask_prefix, mask_suffix], dim=1)
        else:
            mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        n_patches = x.shape[1] // PATCH_SIZE
        inputs = x.reshape(B, n_patches, PATCH_SIZE)
        masks  = mask.reshape(B, n_patches, PATCH_SIZE)
        return inputs, masks

    def _normalize(
        self, inputs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        逐 patch 因果归一化（与 TimesFM 预训练时的 decode() 一致）。

        Returns
        -------
        normed_inputs  : [B, n_patches, PATCH_SIZE]  归一化后的输入
        context_mu     : [B, n_patches]               每个 patch 对应的 running mean
        context_sigma  : [B, n_patches]               每个 patch 对应的 running std
        """
        B = inputs.shape[0]
        device = inputs.device
        n = torch.zeros(B, device=device)
        mu = torch.zeros(B, device=device)
        sigma = torch.zeros(B, device=device)

        patch_mu, patch_sigma = [], []
        for i in range(inputs.shape[1]):
            (n, mu, sigma), _ = tfm_util.update_running_stats(
                n, mu, sigma, inputs[:, i], masks[:, i])
            patch_mu.append(mu)
            patch_sigma.append(sigma)

        context_mu    = torch.stack(patch_mu,    dim=1)  # [B, n_patches]
        context_sigma = torch.stack(patch_sigma, dim=1)  # [B, n_patches]

        normed = tfm_util.revin(inputs, context_mu, context_sigma, reverse=False)
        normed = torch.where(masks, 0.0, normed)
        return normed, context_mu, context_sigma

    # ── 主前向传播 ────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数
        ----
        x : [batch, context_len]  原始电价序列（未归一化）

        返回
        ----
        quant_pred   : [batch, horizon, 9]  分位数预测 q0.1…q0.9
        spike_logits : [batch, horizon]     尖峰 logits（用 sigmoid 转概率）
        """
        B = x.shape[0]
        device = x.device

        # Step 1: 切 patch + 掩码
        inputs, masks = self._make_patches(x)        # [B, n_patches, 32]

        # Step 2: 逐 patch 因果归一化
        normed, ctx_mu, ctx_sigma = self._normalize(inputs, masks)

        # Step 3: Tokenizer 输入 = [normed_patches | masks]（64 维）
        tok_in = torch.cat([normed, masks.to(normed.dtype)], dim=-1)  # [B, n_patches, 64]
        h = self.tokenizer(tok_in)                   # [B, n_patches, 1280]

        # Step 4: patch_mask 用于 Transformer attention（每 patch 一个 bool）
        # masks[..., -1]：取每个 patch 最后一步的 mask 作为该 patch 的合法性标志
        patch_mask = masks[..., -1]                  # [B, n_patches]，True=padding

        # Step 5: 逐层 Transformer，在 SPIKE_LAYER_IDX 捕获 h_spike
        h_spike: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            h, _ = layer(h, patch_mask, None)        # (B, n_patches, 1280), cache=None
            if i == SPIKE_LAYER_IDX:
                h_spike = h[:, -1, :]                # [B, 1280]，取最后一个 patch 位置

        assert h_spike is not None, "Spike layer 未触发，检查 SPIKE_LAYER_IDX"

        # Step 6: Quantile head → 反归一化 → 取 horizon 步 × 9 分位数
        q_raw     = self.quant_head(h)               # [B, n_patches, 10240]
        q_denorm  = tfm_util.revin(q_raw, ctx_mu, ctx_sigma, reverse=True)
        # 取最后 patch 位置，reshape 为 [B, QUANTILE_OS=1024, N_Q=10]
        q_last    = q_denorm[:, -1, :].reshape(B, QUANTILE_OS, N_Q)
        # 取前 horizon 步，分位数索引 1-9 = q0.1 ~ q0.9
        quant_pred = q_last[:, :self.horizon, 1:10]  # [B, horizon, 9]

        # Step 7: Spike head（从新 L6 = 原 L7 的 h_spike 分叉，在精度"压制"层之前）
        spike_logits = self.spike_head(h_spike)      # [B, horizon]

        return quant_pred, spike_logits

    # ── 便捷推理方法（eval 模式，不计算梯度）────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """
        推理接口，返回 evaluation 需要的 mean/q10/q50/q90 和 spike_prob。
        spike_prob 的阈值 τ* 由 evaluate.py 的验证集搜索决定，不在此处处理。

        参数
        ----
        x : [batch, context_len]  原始电价序列（未归一化）

        返回
        ----
        dict 含 mean/q10/q50/q90 [batch, horizon] 和 spike_prob [batch, horizon]
        """
        self.eval()
        quant_pred, spike_logits = self.forward(x)

        # quant_pred 是 [B, H, 9]，来自原始 10 个分位数中的索引 1-9（q0.1~q0.9）
        # 切片后：index 0=q0.1, 4=q0.5, 8=q0.9
        # .cpu() 是必须的：当 model 在 GPU 上时，tensor 在 CUDA；
        # numpy() 不能直接作用于 CUDA tensor，必须先移回 CPU。
        return {
            "mean":       quant_pred[:, :, 4].cpu().numpy(),  # q0.5 作为点预测
            "q10":        quant_pred[:, :, 0].cpu().numpy(),  # q0.1
            "q50":        quant_pred[:, :, 4].cpu().numpy(),  # q0.5
            "q90":        quant_pred[:, :, 8].cpu().numpy(),  # q0.9
            "spike_prob": torch.sigmoid(spike_logits).cpu().numpy(),
        }

    # ── 权重保存 / 加载 ───────────────────────────────────────────────────────

    def save(self, path: str):
        """保存完整模型权重（包含剪枝后的 backbone + spike head）。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "ElecFM":
        """从 checkpoint 加载 ElecFM（先初始化架构，再加载权重）。"""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model
