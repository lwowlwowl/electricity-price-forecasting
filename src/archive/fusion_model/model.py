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

# ── LoRA 支持 ────────────────────────────────────────────────────────────────
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

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
        use_swiglu_adapter: bool = False,
        swiglu_adapter_dim: int = 64,
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

        # ── V7：可选 SwiGLU adapter（Toto 的 SwiGLU FFN 思路移植）──────────────
        # 插在 h_spike → spike_head 之间，零初始化，不影响初始性能
        self.swiglu_adapter: Optional[SwiGLUAdapter] = (
            SwiGLUAdapter(D_MODEL, swiglu_adapter_dim) if use_swiglu_adapter else None
        )

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
        # 兼容 PeftModel（LoRA 包装后）和普通 ModuleList
        layers = self.layers
        if hasattr(layers, 'base_model'):
            # PeftModel 情况：访问基础模型
            layers = layers.base_model.model if hasattr(layers.base_model, 'model') else layers.base_model
        for i, layer in enumerate(layers):
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

        # Step 7: SwiGLU adapter（可选，V7 Toto 组件移植）→ Spike head
        if self.swiglu_adapter is not None:
            h_spike = self.swiglu_adapter(h_spike)   # zero-init，初始不影响性能
        spike_logits = self.spike_head(h_spike)      # [B, horizon]

        return quant_pred, spike_logits

    # ── 返回中间特征（供 ElecFMV6 使用）─────────────────────────────────────────

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        与 forward() 相同，但返回 h_spike 而不是 spike_logits。
        供 ElecFMV6 在 CrossNodeAttention 之前提取各节点 hidden state。

        返回
        ----
        quant_pred : [batch, horizon, 9]
        h_spike    : [batch, d_model]   新 L6 的 hidden state
        """
        B = x.shape[0]
        inputs, masks = self._make_patches(x)
        normed, ctx_mu, ctx_sigma = self._normalize(inputs, masks)
        tok_in = torch.cat([normed, masks.to(normed.dtype)], dim=-1)
        h = self.tokenizer(tok_in)
        patch_mask = masks[..., -1]

        h_spike: Optional[torch.Tensor] = None
        layers = self.layers
        if hasattr(layers, 'base_model'):
            layers = layers.base_model.model if hasattr(layers.base_model, 'model') else layers.base_model
        for i, layer in enumerate(layers):
            h, _ = layer(h, patch_mask, None)
            if i == SPIKE_LAYER_IDX:
                h_spike = h[:, -1, :]

        assert h_spike is not None
        q_raw    = self.quant_head(h)
        q_denorm = tfm_util.revin(q_raw, ctx_mu, ctx_sigma, reverse=True)
        q_last   = q_denorm[:, -1, :].reshape(B, QUANTILE_OS, N_Q)
        quant_pred = q_last[:, :self.horizon, 1:10]
        # 注意：forward_features 返回 adapter 前的 h_spike（供 V6 CrossNodeAttention 使用）
        # V7 的 adapter 在 spike head 调用处应用，不在这里
        return quant_pred, h_spike

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


# ── LoRA 辅助函数 ────────────────────────────────────────────────────────────

def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    """
    自动识别模型中适合添加 LoRA 的目标模块名称。
    主要匹配 Attention 和 FFN 中的线性投影层。

    TimesFM 命名规范：
    - Attention: attn.qkv_proj (合并 QKV), attn.out (输出投影)
    - FFN: ff0, ff1
    """
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # TimesFM 特定的命名
            if any(key in name for key in ["attn.qkv_proj", "attn.out", "ff0", "ff1"]):
                target_modules.append(name)
            # 其他常见命名（备用）
            elif any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj",
                                            "query", "key", "value", "dense",
                                            "gate_proj", "up_proj", "down_proj",
                                            "fc1", "fc2"]):
                target_modules.append(name)
    return target_modules


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> nn.Module:
    """
    对模型应用 LoRA。

    参数
    ----
    model : 基础模型（ElecFM 或 TimesFM 子模块）
    r : LoRA 低秩维度（默认 8）
    lora_alpha : 缩放因子（默认 16 = 2*r）
    lora_dropout : LoRA dropout 概率（默认 0.05）
    target_modules : 目标模块名称列表，None 则自动识别

    返回
    ----
    应用 LoRA 后的模型
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft 库未安装，请运行: pip install peft")

    if target_modules is None:
        target_modules = get_target_modules_for_lora(model)

    if not target_modules:
        raise ValueError("未找到适合 LoRA 的目标模块，请检查模型结构")

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    lora_model = get_peft_model(model, config)
    return lora_model


def create_elecfm_with_lora(
    horizon: int = 24,
    spike_head_hidden: int = 256,
    spike_head_dropout: float = 0.1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    pretrained_id: str = "google/timesfm-2.5-200m-pytorch",
) -> ElecFM:
    """
    创建带 LoRA 的 ElecFM 模型。

    参数
    ----
    horizon, spike_head_hidden, spike_head_dropout : 模型架构参数
    lora_r : LoRA 低秩维度（默认 8）
    lora_alpha : LoRA 缩放因子（默认 16）
    lora_dropout : LoRA dropout（默认 0.05）
    target_modules : 目标模块列表，None 则使用 TimesFM 默认值
    pretrained_id : TimesFM 预训练模型 ID

    返回
    ----
    ElecFM 模型，其中 layers 已应用 LoRA
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft 库未安装，请运行: pip install peft")

    # 1. 创建基础 ElecFM 模型
    base_model = ElecFM(
        pretrained_id=pretrained_id,
        horizon=horizon,
        spike_head_hidden=spike_head_hidden,
        spike_head_dropout=spike_head_dropout,
    )

    # 2. 对 layers 应用 LoRA
    # 注意：我们需要对 transformer layers 应用 LoRA
    if target_modules is None:
        # TimesFM 默认目标：Transformer 层的投影矩阵
        target_modules = ["attn.qkv_proj", "attn.out", "ff0", "ff1"]

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        # 不指定 task_type，避免需要 prepare_inputs_for_generation
    )

    # 3. 对 layers 应用 LoRA（使用 ModuleList 包装）
    from peft import PeftModel
    # 先包装 ModuleList
    lora_layers = get_peft_model(base_model.layers, config)
    base_model.layers = lora_layers

    # 4. 打印可训练参数信息
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"LoRA 模型参数: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}% 可训练)")

    return base_model


# 向后兼容：保留原始 create_elecfm_model 函数名
create_elecfm_model = create_elecfm_with_lora


# ── V7：SwiGLU Adapter（Toto 的 SwiGLU FFN 思路移植）─────────────────────────

class SwiGLUAdapter(nn.Module):
    """
    零初始化 SwiGLU 残差旁路（V7 新增组件）。
    插在 h_spike → spike_head 之间，给 spike head 的输入做一次 SwiGLU 风格变换。

    设计原则：
    - 零初始化 W3（输出投影）→ 训练起始时 adapter 输出为 0，等同于 identity
    - 不改变骨干任何权重，不影响 SMAPE
    - 消融依据：Toto 移除 SwiGLU FFN 后退化 +419%，FFN 是最关键组件
      本 adapter 将 SwiGLU 逻辑移植到 TimesFM 的 spike 表征上

    参数量：3 × d_model × d_hidden（d_hidden=64 时约 245K）
    """

    def __init__(self, d_model: int = D_MODEL, d_hidden: int = 64):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_hidden, bias=False)  # gate
        self.W2 = nn.Linear(d_model, d_hidden, bias=False)  # value
        self.W3 = nn.Linear(d_hidden, d_model, bias=False)  # output（零初始化）
        nn.init.zeros_(self.W3.weight)  # 零初始化：训练开始时旁路输出 = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : [B, d_model]

        返回
        ----
        [B, d_model]   x + SwiGLU(x)（零初始化时等于 x）
        """
        gate = F.silu(self.W1(x))          # [B, d_hidden]
        val  = self.W2(x)                  # [B, d_hidden]
        return x + self.W3(gate * val)     # 残差连接


# ── V6：跨节点注意力 ─────────────────────────────────────────────────────────

class CrossNodeAttention(nn.Module):
    """
    轻量跨节点注意力（V6 新增组件）。
    在 spike head 分叉点（新 L6 = 原 L7）之后、spike_head 之前，
    让 3 个节点的 hidden state 互相做注意力，引入跨节点价格相关性信息。

    参数来源：Chronos skip_variate 消融（SMAPE 退化 +6.7%）提供动机，
             但此为在 TimesFM 骨干上的移植实验，效果待验证。

    参数量：4 × (1280×64) + 1280×2 ≈ 330K
    """

    def __init__(self, d_model: int = D_MODEL, d_attn: int = 64):
        super().__init__()
        self.q    = nn.Linear(d_model, d_attn, bias=False)
        self.k    = nn.Linear(d_model, d_attn, bias=False)
        self.v    = nn.Linear(d_model, d_attn, bias=False)
        self.out  = nn.Linear(d_attn, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.scale = d_attn ** -0.5

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        h : [B, N, d_model]   N 个节点的 hidden state（N=3）

        返回
        ----
        [B, N, d_model]   注意力增强后的 hidden state
        """
        Q   = self.q(h)                                          # [B, N, d_attn]
        K   = self.k(h)                                          # [B, N, d_attn]
        V   = self.v(h)                                          # [B, N, d_attn]
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # [B, N, N]
        ctx  = attn @ V                                          # [B, N, d_attn]
        out  = self.out(ctx)                                     # [B, N, d_model]
        return self.norm(h + out)                                # residual + norm


class ElecFMV6(nn.Module):
    """
    ElecFM V6：3 节点联合 spike 检测模型。

    骨干（base）完全冻结，权重在 3 个节点间共享。
    CrossNodeAttention 和 spike_head 是唯一可训练的组件。

    输入：[B, 3, context_len]   3 个节点的历史电价
    输出：
      quant_pred   : [B, 3, horizon, 9]   各节点分位数预测
      spike_logits : [B, 3, horizon]       各节点尖峰 logits
    """

    def __init__(self, base: ElecFM, d_attn: int = 64):
        super().__init__()
        self.base       = base
        self.cross_attn = CrossNodeAttention(D_MODEL, d_attn)
        # spike_head 通过 self.base.spike_head 访问（共享参数）

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape   # N = 3

        # 1. 各节点独立跑冻结骨干，提取 h_spike 和 quant_pred
        quant_preds, h_spikes = [], []
        for n in range(N):
            q_n, h_n = self.base.forward_features(x[:, n, :])
            quant_preds.append(q_n)
            h_spikes.append(h_n)

        # 2. 跨节点注意力：[B, N, D_MODEL]
        h_stack   = torch.stack(h_spikes, dim=1)        # [B, N, D_MODEL]
        h_enhanced = self.cross_attn(h_stack)            # [B, N, D_MODEL]

        # 3. 各节点 spike_head（共享权重）
        spike_logits = torch.stack(
            [self.base.spike_head(h_enhanced[:, n, :]) for n in range(N)],
            dim=1)                                       # [B, N, horizon]

        # 4. 堆叠 quant 预测
        quant_pred = torch.stack(quant_preds, dim=1)    # [B, N, horizon, 9]

        return quant_pred, spike_logits

    def save(self, path: str):
        """保存 V6 完整权重（base + cross_attn）。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @torch.no_grad()
    def predict_node(self, x: torch.Tensor, node_idx: int) -> dict:
        """
        单节点推理接口（用于 evaluate.py 兼容）。
        x: [B, context_len] 单节点输入，其余节点用零填充。
        """
        B = x.shape[0]
        ctx = torch.zeros(B, 3, x.shape[1], device=x.device, dtype=x.dtype)
        ctx[:, node_idx, :] = x
        quant_pred, spike_logits = self.forward(ctx)
        q = quant_pred[:, node_idx]      # [B, H, 9]
        s = spike_logits[:, node_idx]    # [B, H]
        return {
            "mean":       q[:, :, 4].cpu().numpy(),
            "q10":        q[:, :, 0].cpu().numpy(),
            "q50":        q[:, :, 4].cpu().numpy(),
            "q90":        q[:, :, 8].cpu().numpy(),
            "spike_prob": torch.sigmoid(s).cpu().numpy(),
        }
