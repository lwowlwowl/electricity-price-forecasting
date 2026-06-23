"""
结构级消融器 ablations.py
==========================
v2.1 手册所有结构消融实验的实现。每个消融器接收已加载的 PyTorch 模型，
in-place 替换指定组件后返回修改后的模型。

设计原则：
- 纯 PyTorch 操作，不依赖本项目其他模块（worker 内部独立使用）。
- 每个消融器是一个函数，签名为 apply_<ablation>(model, config) -> model。
- config 是一个 dict，允许传入额外参数（如目标层索引、模型类型标识等）。
- worker 根据 request.npz 中的 ablation_type 字段选择对应函数执行。

消融类型编码（ablation_type 字段值）：
  skip_attention      : S-A2
  halve_heads         : S-A3
  skip_variate        : S-B1
  skip_time           : S-B2
  remove_rope         : S-G1a
  disable_xpos        : S-G1b (仅 Toto-2.0)
  simplify_patch_emb  : S-G2
  simplify_output_head: S-G3a
  point_only          : S-G3b (仅 TimesFM)
  skip_ffn            : S-G4
  skip_layernorm      : S-G5
  truncate_front_half : S-G6a
  truncate_front_quarter: S-G6b (仅 TimesFM)
  truncate_back_half  : S-G6c
  skip_layer          : S-L (逐层消融，需 config.layer_index)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════════
#  通用 Identity 层（匹配各模型的不同签名）
# ════════════════════════════════════════════════════════════════════════════════

class IdentityLayerToto2(nn.Module):
    """替代 Toto-2.0 SelfAttention: forward(x, seq_ids=None, **kwargs) → Tensor"""
    def forward(self, x, seq_ids=None, **kwargs):
        return x


class IdentityLayerChronos2(nn.Module):
    """替代 Chronos-2 TimeSelfAttention / GroupSelfAttention。

    EncoderBlock.forward 中的调用方式：
      time_self_attn_outputs = self.layer[0](hidden_states, position_ids=..., attention_mask=..., ...)
      hidden_states = time_self_attn_outputs[0]   # 取第一元素

      group_self_attn_outputs = self.layer[1](hidden_states, attention_mask=..., ...)
      hidden_states = group_self_attn_outputs[0]

    TimeSelfAttention.forward 的真实逻辑：
      normed = layer_norm(hidden_states)
      attn_out = self_attention(normed, ...)
      hidden_states = hidden_states + dropout(attn_out[0])  ← 残差
      return AttentionOutput(hidden_states=hidden_states, ...)

    因此 Identity 应模拟"跳过注意力但保留残差"，即直接返回 hidden_states 本身。
    返回值需支持 [0] 索引，以便调用方 `outputs[0]` 取出 hidden_states。
    """
    def forward(self, hidden_states, **kwargs):
        return _Chronos2IdentityOutput(hidden_states)


class _Chronos2IdentityOutput:
    """模拟 Chronos2 AttentionOutput 的 [0] 索引访问。"""
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
        self.attn_weights = None

    def __getitem__(self, idx):
        if idx == 0:
            return self.hidden_states
        return None


class IdentityLayerTimesFM(nn.Module):
    """替代 TimesFM MultiHeadAttention。
    真实签名：forward(inputs_q, *, decode_cache, patch_mask) → (Tensor, cache)
    原 Transformer.forward 解包为: attn_output, decode_cache = layer.attn(...)
    """
    def forward(self, inputs_q, *, decode_cache=None, patch_mask=None, **kwargs):
        return inputs_q, decode_cache


class IdentitySimple(nn.Module):
    """通用单输入 Identity：替代 FFN 等 forward(x) → x"""
    def forward(self, x, **kwargs):
        return x


class IdentityRoPE(nn.Module):
    """TimesFM 专用 RoPE 替换。
    TimesFM rotary_position_embedding.forward(inputs, position) → Tensor（直接旋转后的张量）
    替换后直接返回原张量。
    """
    def forward(self, inputs, position=None, **kwargs):
        return inputs


class IdentityRoPEChronos2(nn.Module):
    """Chronos-2 专用 RoPE 替换。
    rope_embed.forward(x, position_ids) → (cos, sin)
    调用方: cos, sin = rope_embed(value_states, position_ids)
            q, k = RoPE.apply_rotary_pos_emb(q, k, cos, sin)
    令 cos=1, sin=0 使旋转变为恒等变换（不改变 Q/K）。
    """
    def forward(self, x, position_ids=None, **kwargs):
        # x: (batch, n_heads, seq_len, head_dim)
        # 返回 cos=ones, sin=zeros，使 apply_rotary_pos_emb 输出恒等
        bs, n_heads, seq_len, head_dim = x.shape
        cos = torch.ones(bs, seq_len, head_dim, dtype=x.dtype, device=x.device)
        sin = torch.zeros(bs, seq_len, head_dim, dtype=x.dtype, device=x.device)
        return cos, sin


# ════════════════════════════════════════════════════════════════════════════════
#  Toto-2.0 辅助函数
# ════════════════════════════════════════════════════════════════════════════════

def _is_variate_layer_toto2(model, layer_idx: int) -> bool:
    """判断 Toto-2.0 的第 layer_idx 层是否为 variate 层。"""
    cfg = model.config
    layer_group_size = getattr(cfg, "layer_group_size", 1)
    num_variate = getattr(cfg, "num_variate_layers_per_group", 0)
    variate_first = getattr(cfg, "variate_layer_first", False)
    if variate_first:
        return layer_idx % layer_group_size < num_variate
    return layer_idx % layer_group_size >= layer_group_size - num_variate


# ════════════════════════════════════════════════════════════════════════════════
#  消融实现：通用（三模型共用逻辑）
# ════════════════════════════════════════════════════════════════════════════════

def apply_skip_attention(model, model_type: str, config: dict = None):
    """S-A2：跳过所有注意力层（替换为 Identity）。"""
    if model_type == "toto2":
        for layer in model.transformer.layers:
            layer.attn = IdentityLayerToto2()
    elif model_type == "chronos2":
        # EncoderBlock.forward 调用:
        #   time_self_attn_outputs = self.layer[0](hidden_states, position_ids=..., attention_mask=..., ...)
        #   hidden_states = time_self_attn_outputs[0]
        # 需要返回 AttentionOutput-like 对象（支持 [0] 索引返回 hidden_states）
        for block in model.encoder.block:
            block.layer[0] = IdentityLayerChronos2()
            block.layer[1] = IdentityLayerChronos2()
    elif model_type == "timesfm":
        # Transformer.forward: attn_output, decode_cache = self.attn(inputs_q=..., decode_cache=..., patch_mask=...)
        # Identity 必须返回 (Tensor, cache) 元组
        for layer in model.stacked_xf:
            layer.attn = IdentityLayerTimesFM()
    return model


def apply_halve_heads(model, model_type: str, config: dict = None):
    """S-A3：将注意力头数减半。"""
    if model_type == "toto2":
        _halve_heads_toto2(model)
    elif model_type == "chronos2":
        _halve_heads_chronos2(model)
    elif model_type == "timesfm":
        _halve_heads_timesfm(model)
    return model


def apply_skip_variate(model, model_type: str, config: dict = None):
    """S-B1：跳过变量/Group 注意力层。"""
    if model_type == "toto2":
        for idx, layer in enumerate(model.transformer.layers):
            if _is_variate_layer_toto2(model, idx):
                layer.attn = IdentityLayerToto2()
    elif model_type == "chronos2":
        # block.layer[1] = GroupSelfAttention；EncoderBlock.forward:
        #   group_self_attn_outputs = self.layer[1](hidden_states, attention_mask=group_time_mask, ...)
        #   hidden_states = group_self_attn_outputs[0]
        for block in model.encoder.block:
            block.layer[1] = IdentityLayerChronos2()
    # TimesFM 无跨变量机制，此消融不适用
    return model


def apply_skip_time(model, model_type: str, config: dict = None):
    """S-B2：跳过时间注意力层，只保留 Variate/Group。"""
    if model_type == "toto2":
        for idx, layer in enumerate(model.transformer.layers):
            if not _is_variate_layer_toto2(model, idx):
                layer.attn = IdentityLayerToto2()
    elif model_type == "chronos2":
        # block.layer[0] = TimeSelfAttention
        for block in model.encoder.block:
            block.layer[0] = IdentityLayerChronos2()
    return model


def apply_remove_rope(model, model_type: str, config: dict = None):
    """S-G1a：移除所有 RoPE 位置编码。"""
    if model_type == "toto2":
        for idx, layer in enumerate(model.transformer.layers):
            if not _is_variate_layer_toto2(model, idx):
                # qk_proj 包含 RoPE，设为 None 跳过
                if hasattr(layer.attn, "qk_proj"):
                    layer.attn.qk_proj = None
    elif model_type == "chronos2":
        # MHA.forward: if self.use_rope: cos, sin = self.rope_embed(value_states, position_ids)
        #              query_states, key_states = RoPE.apply_rotary_pos_emb(q, k, cos, sin)
        # 方案：将 rope_embed 替换为返回 cos=1/sin=0 的 IdentityRoPEChronos2
        #       同时保持 use_rope=True（不改 assert），因为 IdentityRoPE 可以接受 position_ids
        for block in model.encoder.block:
            for layer_idx in (0, 1):  # TimeSelfAttention, GroupSelfAttention
                sub_attn = block.layer[layer_idx]
                if hasattr(sub_attn, "self_attention"):
                    mha = sub_attn.self_attention
                    if hasattr(mha, "rope_embed"):
                        mha.rope_embed = IdentityRoPEChronos2()
    elif model_type == "timesfm":
        # TimesFM attn.rotary_position_embedding（非 rotary_pos_emb）
        # 调用方: query = self.rotary_position_embedding(query, position)
        # IdentityRoPE.forward(inputs, position) → inputs
        for layer in model.stacked_xf:
            attn = layer.attn
            if hasattr(attn, "rotary_position_embedding"):
                attn.rotary_position_embedding = IdentityRoPE()
                attn.use_rotary_position_embeddings = False  # 跳过整个 if 分支
    return model


def apply_disable_xpos(model, model_type: str, config: dict = None):
    """S-G1b：关闭 xPos 衰减（仅 Toto-2.0）。保留 RoPE 旋转，只禁用距离衰减。"""
    if model_type != "toto2":
        return model
    for idx, layer in enumerate(model.transformer.layers):
        if not _is_variate_layer_toto2(model, idx):
            qk_proj = getattr(layer.attn, "qk_proj", None)
            if qk_proj is not None:
                # 尝试禁用 xPos 衰减因子
                if hasattr(qk_proj, "scale"):
                    qk_proj.scale = None
                if hasattr(qk_proj, "use_xpos"):
                    qk_proj.use_xpos = False
    return model


def apply_simplify_patch_emb(model, model_type: str, config: dict = None):
    """S-G2：将 Patch 嵌入替换为简单线性投影。"""
    if model_type == "toto2":
        cfg = model.config
        patch_size = cfg.patch_size
        d_model = cfg.d_model
        model.patch_proj = nn.Linear(2 * patch_size, d_model)
    elif model_type == "chronos2":
        # input_patch_embedding: ResidualBlock，没有 in_features/out_features 属性
        # 实际维度：residual_layer.weight.shape = (out_dim, in_dim)
        # residual_layer 是跳连从 in_dim → out_dim 的直接映射，最可靠
        old = model.input_patch_embedding
        res_layer = old.residual_layer  # Linear(in_dim → out_dim)
        in_dim = res_layer.weight.shape[1]  # 48 (patch_size * n_channels)
        out_dim = res_layer.weight.shape[0]  # 768 (d_model)
        model.input_patch_embedding = nn.Linear(in_dim, out_dim)
    elif model_type == "timesfm":
        # tokenizer: ResidualBlock，residual_layer.weight.shape = (1280, 64)
        old = model.tokenizer
        res_layer = old.residual_layer
        in_dim = res_layer.weight.shape[1]   # 64 (2 * patch_size)
        out_dim = res_layer.weight.shape[0]  # 1280 (model_dims)
        model.tokenizer = nn.Linear(in_dim, out_dim)
    return model


def apply_simplify_output_head(model, model_type: str, config: dict = None):
    """S-G3a：将输出头替换为直接线性映射。
    Toto2 不适用：QuantileKnotsOutputHead 内部含 FusedPatchedParamProjection，
    forward 签名不兼容简单 nn.Linear 替换，已从 ABLATION_APPLICABILITY 移除。
    """
    if model_type == "chronos2":
        # output_patch_embedding: ResidualBlock，residual_layer.weight = (out_dim, in_dim)
        # out_dim = 336 (output_patch_size * n_quantiles), in_dim = 768 (d_model)
        old = model.output_patch_embedding
        res_layer = old.residual_layer
        in_dim = res_layer.weight.shape[1]   # 768
        out_dim = res_layer.weight.shape[0]  # 336
        model.output_patch_embedding = nn.Linear(in_dim, out_dim)
    elif model_type == "timesfm":
        # output_projection_quantiles: ResidualBlock
        # residual_layer.weight.shape = (10240, 1280) → 替换为 Linear(1280, 10240)
        old = model.output_projection_quantiles
        res_layer = old.residual_layer
        in_dim = res_layer.weight.shape[1]   # 1280
        out_dim = res_layer.weight.shape[0]  # 10240
        model.output_projection_quantiles = nn.Linear(in_dim, out_dim)
        # 同样替换 point head
        old_pt = model.output_projection_point
        res_layer_pt = old_pt.residual_layer
        in_dim_pt = res_layer_pt.weight.shape[1]   # 1280
        out_dim_pt = res_layer_pt.weight.shape[0]  # 1280 (output patch len)
        model.output_projection_point = nn.Linear(in_dim_pt, out_dim_pt)
    return model


def apply_point_only(model, model_type: str, config: dict = None):
    """S-G3b：仅 TimesFM - 使用点预测头，标记忽略分位数头。"""
    # 这个消融在推理逻辑中处理（worker 检查标志只取 point output），
    # 不需要实际修改模型结构。用一个标记属性即可。
    if model_type == "timesfm":
        model._ablation_point_only = True
    return model


def apply_skip_ffn(model, model_type: str, config: dict = None):
    """S-G4：跳过所有 FFN。"""
    if model_type == "toto2":
        for layer in model.transformer.layers:
            layer.ffn = IdentitySimple()
    elif model_type == "chronos2":
        # FeedForward.forward: forwarded = layer_norm(hidden); forwarded = mlp(forwarded); return hidden + dropout(forwarded)
        # 直接把整个 block.layer[2] 换成 IdentitySimple 会跳过残差连接，使 hidden_states 停止流动。
        # 正确做法：只替换 mlp，保留 layer_norm 和残差。
        for block in model.encoder.block:
            block.layer[2].mlp = IdentitySimple()
    elif model_type == "timesfm":
        # Transformer.forward: output = post_ff_ln(ff1(activation(ff0(pre_ff_ln(attn_out))))) + attn_out
        # ff0 和 ff1 均替换为 Identity，消除两个 Linear 的学习参数。
        # 保留路径：pre_ff_ln → Identity → SiLU → Identity → post_ff_ln → residual。
        # SiLU 仍在计算图中，但无线性变换，等价于"跳过 FFN 的特征变换能力"。
        for layer in model.stacked_xf:
            layer.ff0 = nn.Identity()
            layer.ff1 = nn.Identity()
    return model


def apply_skip_layernorm(model, model_type: str, config: dict = None):
    """S-G5：将所有 LayerNorm/RMSNorm 替换为 Identity。
    注意：此消融在 Toto2 和 TimesFM 上可能导致数值爆炸（CRASH），这本身是有效的实验结论。
    """
    if model_type == "toto2":
        # Toto2 的 norm 存在于 τ-rule 残差机制中，名称可能不是 norm1/norm2
        # 遍历所有子模块替换
        for module in model.modules():
            cls_name = type(module).__name__
            if "RMSNorm" in cls_name or "LayerNorm" in cls_name:
                # 用 Identity 替换该模块
                parent = _find_parent(model, module)
                if parent is not None:
                    for name, child in parent.named_children():
                        if child is module:
                            setattr(parent, name, nn.Identity())
                            break
    elif model_type == "chronos2":
        # 精确替换各子层的 layer_norm 属性（pre-norm 模式）
        for block in model.encoder.block:
            for sub_layer in block.layer:
                if hasattr(sub_layer, "layer_norm"):
                    sub_layer.layer_norm = nn.Identity()
        if hasattr(model.encoder, "final_layer_norm"):
            model.encoder.final_layer_norm = nn.Identity()
    elif model_type == "timesfm":
        for layer in model.stacked_xf:
            for norm_attr in ("pre_attn_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln"):
                if hasattr(layer, norm_attr):
                    setattr(layer, norm_attr, nn.Identity())
    return model


def _find_parent(root: nn.Module, target: nn.Module):
    """找到 target 模块在 root 模型树中的父模块。"""
    for parent in root.modules():
        for child in parent.children():
            if child is target:
                return parent
    return None


def apply_truncate_front_half(model, model_type: str, config: dict = None):
    """S-G6a：只保留前 50% 的层。"""
    if model_type == "toto2":
        total = len(model.transformer.layers)
        keep = list(model.transformer.layers)[:total // 2]
        model.transformer.layers = nn.ModuleList(keep)
    elif model_type == "chronos2":
        total = len(model.encoder.block)
        keep = list(model.encoder.block)[:total // 2]
        model.encoder.block = nn.ModuleList(keep)
    elif model_type == "timesfm":
        total = len(model.stacked_xf)
        keep = list(model.stacked_xf)[:total // 2]
        model.stacked_xf = nn.ModuleList(keep)
    return model


def apply_truncate_front_quarter(model, model_type: str, config: dict = None):
    """S-G6b：只保留前 25% 的层（仅 TimesFM 有意义，20→5 层）。"""
    if model_type == "timesfm":
        total = len(model.stacked_xf)
        keep = list(model.stacked_xf)[:total // 4]
        model.stacked_xf = nn.ModuleList(keep)
    elif model_type == "toto2":
        total = len(model.transformer.layers)
        keep = list(model.transformer.layers)[:max(1, total // 4)]
        model.transformer.layers = nn.ModuleList(keep)
    elif model_type == "chronos2":
        total = len(model.encoder.block)
        keep = list(model.encoder.block)[:max(1, total // 4)]
        model.encoder.block = nn.ModuleList(keep)
    return model


def apply_truncate_back_half(model, model_type: str, config: dict = None):
    """S-G6c：只保留后 50% 的层。"""
    if model_type == "toto2":
        total = len(model.transformer.layers)
        keep = list(model.transformer.layers)[total // 2:]
        model.transformer.layers = nn.ModuleList(keep)
    elif model_type == "chronos2":
        total = len(model.encoder.block)
        keep = list(model.encoder.block)[total // 2:]
        model.encoder.block = nn.ModuleList(keep)
    elif model_type == "timesfm":
        total = len(model.stacked_xf)
        keep = list(model.stacked_xf)[total // 2:]
        model.stacked_xf = nn.ModuleList(keep)
    return model


# ════════════════════════════════════════════════════════════════════════════════
#  逐层消融（Per-Layer Ablation）
# ════════════════════════════════════════════════════════════════════════════════

class IdentityBlockToto2(nn.Module):
    """替代 Toto-2.0 整个 SelfAttentionTransformerLayer。
    forward 签名匹配原层：(x, seq_ids=None, cache_layer=None, ...) → x"""
    def forward(self, x, seq_ids=None, cache_layer=None, **kwargs):
        return x


class IdentityBlockChronos2(nn.Module):
    """替代 Chronos-2 整个 EncoderBlock。
    EncoderBlock.forward(hidden_states, *, position_ids, attention_mask, group_time_mask, ...)
    → Chronos2EncoderBlockOutput (dataclass, 支持 [0] 取 hidden_states)。"""
    def forward(self, hidden_states, *, position_ids=None, attention_mask=None,
                group_time_mask=None, output_attentions=False, **kwargs):
        # 返回 tuple-like 对象，支持 layer_outputs[0] = hidden_states
        return _IdentityBlockOutput(hidden_states)


class _IdentityBlockOutput:
    """模拟 Chronos2EncoderBlockOutput 的 [0] 索引访问和 .hidden_states 属性。"""
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
        self.time_self_attn_weights = None
        self.group_self_attn_weights = None

    def __getitem__(self, idx):
        if idx == 0:
            return self.hidden_states
        return None


class IdentityBlockTimesFM(nn.Module):
    """替代 TimesFM 整个 Transformer layer。
    真实签名: forward(input_embeddings, patch_mask, decode_cache=None) → (output_embeddings, new_cache)
    模块对应 stacked_xf[i]，由 module.forward 中调用:
      output_embeddings, new_cache = layer(output_embeddings, masks[..., -1], decode_caches[i])
    """
    def forward(self, input_embeddings, patch_mask=None, decode_cache=None, **kwargs):
        return input_embeddings, decode_cache


def apply_skip_layer(model, model_type: str, config: dict = None):
    """S-L：逐层消融——跳过第 N 层（整层替换为 Identity）。
    config 中须包含 "layer_index": int 指定要跳过的层索引。"""
    config = config or {}
    layer_idx = config.get("layer_index", 0)

    if model_type == "toto2":
        total = len(model.transformer.layers)
        if layer_idx >= total:
            print(f"⚠️  skip_layer: layer_index={layer_idx} 超出范围 (total={total})，跳过。")
            return model
        layers = list(model.transformer.layers)
        layers[layer_idx] = IdentityBlockToto2()
        model.transformer.layers = nn.ModuleList(layers)
    elif model_type == "chronos2":
        total = len(model.encoder.block)
        if layer_idx >= total:
            print(f"⚠️  skip_layer: layer_index={layer_idx} 超出范围 (total={total})，跳过。")
            return model
        blocks = list(model.encoder.block)
        blocks[layer_idx] = IdentityBlockChronos2()
        model.encoder.block = nn.ModuleList(blocks)
    elif model_type == "timesfm":
        total = len(model.stacked_xf)
        if layer_idx >= total:
            print(f"⚠️  skip_layer: layer_index={layer_idx} 超出范围 (total={total})，跳过。")
            return model
        layers = list(model.stacked_xf)
        layers[layer_idx] = IdentityBlockTimesFM()
        model.stacked_xf = nn.ModuleList(layers)
    return model


# ════════════════════════════════════════════════════════════════════════════════
#  Head 减半的具体实现（较复杂，各模型不同）
# ════════════════════════════════════════════════════════════════════════════════

def _halve_heads_toto2(model):
    """Toto-2.0 的 Fused in_proj 头减半。
    in_proj 输出布局: [Q(num_heads*head_dim) | K(num_groups*head_dim) | V(num_groups*head_dim)]
    forward 用 attn._Hq / attn._Hkv / attn._split_sizes 控制分割，
    最终调用 F.scaled_dot_product_attention with enable_gqa=config.heads_per_group > 1。

    注意：本模型 num_groups == num_heads（标准 MHA，heads_per_group=1），GQA 关闭。
    因此必须同时减半 Q 和 KV，保持 _Hq == _Hkv。
    """
    cfg = model.config
    num_heads = cfg.num_heads        # Q 和 KV heads 数（相同）
    num_groups = getattr(cfg, "num_groups", num_heads)
    head_dim = cfg.d_model // num_heads
    half_heads = num_heads // 2      # 4
    new_q_size = half_heads * head_dim   # 256
    new_kv_size = half_heads * head_dim  # 256（KV 也减半以保持 Hq == Hkv）

    for layer in model.transformer.layers:
        attn = layer.attn
        if not hasattr(attn, "in_proj"):
            continue  # 已被 Identity 替换

        has_bias = attn.in_proj.bias is not None
        old_w = attn.in_proj.weight  # [3 * num_heads * head_dim, d_model]
        q_size = num_heads * head_dim    # 512
        kv_size = num_groups * head_dim  # 512（num_groups==num_heads）

        # Q、K、V 同时截取前半 heads
        new_weight = torch.cat([
            old_w[:new_q_size],                        # Q: 前半
            old_w[q_size: q_size + new_kv_size],       # K: 前半
            old_w[q_size + kv_size: q_size + kv_size + new_kv_size],  # V: 前半
        ], dim=0)
        new_in = nn.Linear(cfg.d_model, new_q_size + 2*new_kv_size, bias=has_bias)
        new_in.weight = nn.Parameter(new_weight)
        if has_bias:
            old_b = attn.in_proj.bias
            new_b = torch.cat([
                old_b[:new_q_size],
                old_b[q_size: q_size + new_kv_size],
                old_b[q_size + kv_size: q_size + kv_size + new_kv_size],
            ])
            new_in.bias = nn.Parameter(new_b)
        attn.in_proj = new_in

        # out_proj 输入维度从 q_size → new_q_size（Q heads 的 concat 输出）
        has_out_bias = attn.out_proj.bias is not None
        old_out_w = attn.out_proj.weight  # [d_model, q_size]
        new_out = nn.Linear(new_q_size, cfg.d_model, bias=has_out_bias)
        new_out.weight = nn.Parameter(old_out_w[:, :new_q_size].clone())
        if has_out_bias:
            new_out.bias = nn.Parameter(attn.out_proj.bias.clone())
        attn.out_proj = new_out

        # 更新 forward 内部状态，保持 _Hq == _Hkv
        attn._Hq = half_heads
        attn._Hkv = half_heads
        attn._split_sizes = [new_q_size, new_kv_size, new_kv_size]
        # PerDimScale：维度是 head_dim，不变；qk_proj 用 num_heads 但不影响权重形状


def _halve_heads_chronos2(model):
    """Chronos-2 的 MHA 头减半。投影属性名为 q/k/v/o（非 q_proj）。"""
    for block in model.encoder.block:
        # block.layer[0] = TimeSelfAttention, block.layer[1] = GroupSelfAttention
        for layer_idx in (0, 1):
            sub_attn = block.layer[layer_idx]
            mha = getattr(sub_attn, "self_attention", None)
            if mha is None:
                continue
            _halve_chronos2_mha(mha)


def _halve_chronos2_mha(mha):
    """Chronos-2 MHA 的头减半（投影名为 q/k/v/o）。"""
    q = mha.q
    out_dim = q.weight.shape[0]  # n_heads * kv_proj_dim = 768
    in_dim = q.weight.shape[1]   # d_model = 768
    half_dim = out_dim // 2      # 384

    def _half_proj(linear, in_d, out_d):
        new = nn.Linear(in_d, out_d, bias=linear.bias is not None)
        new.weight = nn.Parameter(linear.weight[:out_d].clone())
        if linear.bias is not None:
            new.bias = nn.Parameter(linear.bias[:out_d].clone())
        return new

    mha.q = _half_proj(mha.q, in_dim, half_dim)
    mha.k = _half_proj(mha.k, in_dim, half_dim)
    mha.v = _half_proj(mha.v, in_dim, half_dim)

    # O 输出投影：输入维度从 out_dim → half_dim
    o = mha.o
    new_o = nn.Linear(half_dim, o.weight.shape[0], bias=o.bias is not None)
    new_o.weight = nn.Parameter(o.weight[:, :half_dim].clone())
    if o.bias is not None:
        new_o.bias = nn.Parameter(o.bias.clone())
    mha.o = new_o

    # 更新 head 数：
    # - n_heads 减半（从 12 → 6）
    # - inner_dim 减半（从 768 → 384），因为 inner_dim = n_heads * kv_proj_dim
    # - kv_proj_dim 保持不变（它是每个 head 的 K/V 维度 = 64，head 减半但每 head 维度不变）
    mha.n_heads = mha.n_heads // 2
    mha.inner_dim = half_dim
    # kv_proj_dim 不变（仍为 64），rearrange 中 d=kv_proj_dim 保持正确


def _halve_heads_timesfm(model):
    """TimesFM 的 fused QKV 头减半。
    model 参数是 internal model（stacked_xf 所在的 Module）。
    需要同时更新：
    1. 每层 attn 的 qkv_proj / out / num_heads
    2. 模块级 self.h（decode() 用此值分配 decode_cache）
    """
    for layer in model.stacked_xf:
        _halve_fused_qkv_timesfm(layer.attn)
    # 更新模块级 num_heads 属性（decode() 用 self.h 分配 decode_cache）
    model.h = model.h // 2


def _halve_fused_qkv_timesfm(attn):
    """TimesFM MultiHeadAttention 头减半（fused qkv_proj）。
    attn 是 MultiHeadAttention 实例，其 num_heads/head_dim 是实例属性（非 nn.Module 属性）。
    """
    # 直接从权重形状推导：qkv_proj.weight = [3*num_heads*head_dim, d_model]
    d_model = attn.qkv_proj.weight.shape[1]        # 1280
    total_qkv = attn.qkv_proj.weight.shape[0]      # 3840 = 3 * 16 * 80
    num_heads = attn.num_heads                      # 16（实例属性，不在 nn.Module 中）
    head_dim = attn.head_dim                        # 80
    half_heads = num_heads // 2                     # 8
    half_inner = half_heads * head_dim              # 640

    # qkv_proj 输出布局: [Q(num_heads*head_dim) | K(num_heads*head_dim) | V(num_heads*head_dim)]
    q_size = num_heads * head_dim  # 1280
    # 截取前半 Q、前半 K、前半 V
    old_w = attn.qkv_proj.weight  # [3840, 1280]
    new_w = torch.cat([
        old_w[:half_inner],                    # Q: 前半 heads
        old_w[q_size: q_size + half_inner],    # K: 前半 heads
        old_w[2*q_size: 2*q_size + half_inner], # V: 前半 heads
    ], dim=0)  # [3*half_inner, 1280]
    new_qkv = nn.Linear(d_model, 3 * half_inner, bias=attn.qkv_proj.bias is not None)
    new_qkv.weight = nn.Parameter(new_w)
    if attn.qkv_proj.bias is not None:
        old_b = attn.qkv_proj.bias
        new_b = torch.cat([
            old_b[:half_inner],
            old_b[q_size: q_size + half_inner],
            old_b[2*q_size: 2*q_size + half_inner],
        ])
        new_qkv.bias = nn.Parameter(new_b)
    attn.qkv_proj = new_qkv

    # out 投影：输入维度从 num_heads*head_dim → half_inner
    old_out = attn.out
    new_out = nn.Linear(half_inner, old_out.weight.shape[0], bias=old_out.bias is not None)
    new_out.weight = nn.Parameter(old_out.weight[:, :half_inner].clone())
    if old_out.bias is not None:
        new_out.bias = nn.Parameter(old_out.bias.clone())
    attn.out = new_out

    # 更新 head 计数和 in_features（forward 中用 self.in_features 做 reshape）
    attn.num_heads = half_heads
    attn.in_features = half_inner  # x.reshape(b, n_patches, self.in_features) 需要更新


# ════════════════════════════════════════════════════════════════════════════════
#  注册表：ablation_type → 函数
# ════════════════════════════════════════════════════════════════════════════════

ABLATION_REGISTRY = {
    "skip_attention":        apply_skip_attention,
    "halve_heads":           apply_halve_heads,
    "skip_variate":          apply_skip_variate,
    "skip_time":             apply_skip_time,
    "remove_rope":           apply_remove_rope,
    "disable_xpos":          apply_disable_xpos,
    "simplify_patch_emb":    apply_simplify_patch_emb,
    "simplify_output_head":  apply_simplify_output_head,
    "point_only":            apply_point_only,
    "skip_ffn":              apply_skip_ffn,
    "skip_layernorm":        apply_skip_layernorm,
    "truncate_front_half":   apply_truncate_front_half,
    "truncate_front_quarter": apply_truncate_front_quarter,
    "truncate_back_half":    apply_truncate_back_half,
    "skip_layer":            apply_skip_layer,
}

# 每种消融对哪些模型有效
ABLATION_APPLICABILITY = {
    "skip_attention":        ["toto2", "chronos2", "timesfm"],
    "halve_heads":           ["toto2", "chronos2", "timesfm"],
    "skip_variate":          ["toto2", "chronos2"],
    "skip_time":             ["toto2", "chronos2"],
    "remove_rope":           ["toto2", "chronos2", "timesfm"],
    "disable_xpos":          ["toto2"],
    "simplify_patch_emb":    ["toto2", "chronos2", "timesfm"],
    "simplify_output_head":  ["chronos2", "timesfm"],  # toto2 不适用（output_head 结构不兼容）
    "point_only":            ["timesfm"],
    "skip_ffn":              ["toto2", "chronos2", "timesfm"],
    "skip_layernorm":        ["toto2", "chronos2", "timesfm"],
    "truncate_front_half":   ["toto2", "chronos2", "timesfm"],
    "truncate_front_quarter": ["toto2", "chronos2", "timesfm"],
    "truncate_back_half":    ["toto2", "chronos2", "timesfm"],
    "skip_layer":            ["toto2", "chronos2", "timesfm"],
}


def apply_ablation(model, model_type: str, ablation_type: str,
                   config: Optional[dict] = None):
    """
    对模型应用指定消融。

    参数
    ----
    model : 已加载的 PyTorch 模型
    model_type : "toto2" / "chronos2" / "timesfm"
    ablation_type : ABLATION_REGISTRY 中的键，或 "none" 表示不消融（对照组）。
                    对于 skip_layer，支持 "skip_layer_N" 格式自动解析 layer_index。
    config : 可选额外配置

    返回
    ----
    修改后的 model（in-place）
    """
    # "none" = 对照组，不做任何修改
    if ablation_type in ("none", "baseline", ""):
        return model

    # 支持 "skip_layer_N" 格式：自动解析 layer_index
    resolved_type = ablation_type
    resolved_config = dict(config) if config else {}
    if ablation_type.startswith("skip_layer_"):
        try:
            layer_idx = int(ablation_type.split("skip_layer_")[1])
            resolved_type = "skip_layer"
            resolved_config["layer_index"] = layer_idx
        except (ValueError, IndexError):
            pass  # 不匹配格式，按原字符串处理

    if resolved_type not in ABLATION_REGISTRY:
        raise ValueError(f"未知消融类型 {ablation_type!r}。"
                         f"可用：{list(ABLATION_REGISTRY) + ['none']}")
    applicable = ABLATION_APPLICABILITY.get(resolved_type, [])
    if model_type not in applicable:
        print(f"⚠️  消融 {ablation_type} 不适用于 {model_type}，跳过。")
        return model
    fn = ABLATION_REGISTRY[resolved_type]
    return fn(model, model_type, resolved_config)
