"""
train.py — ElecFM 两阶段微调训练循环
======================================
Stage 1：冻结 tokenizer + L0-L6（含 spike 关键层 = 新L6 = 原L7），训练 L7-L14 + 双输出头
Stage 2：全层解冻，低 LR Cosine decay 精调；若验证集 Pinball 5 epoch 内反弹，
         可切换至 LoRA 退路（见文档 Section 6.2）

运行环境：external/timesfm/.venv/bin/python
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── 路径 ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from model import ElecFM
from loss  import combined_loss


# ── 训练配置 ─────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Stage 1
    stage1_epochs:    int   = 10
    stage1_lr:        float = 1e-4
    # Stage 2
    stage2_epochs:    int   = 40
    stage2_lr:        float = 5e-6
    stage2_lr_min:    float = 5e-7     # Cosine decay 终点
    # 公共
    batch_size:       int   = 32
    gradient_clip:    float = 1.0
    weight_decay:     float = 0.01
    early_stop_patience: int = 10      # 监控验证集 Pinball
    lambda_pinball:   float = 0.8
    lambda_spike:     float = 0.2
    spike_pos_weight: float = 19.0     # 更新为从数据集统计的实际值
    use_amp:          bool  = True     # fp16 混合精度（需要 GPU）
    checkpoint_dir:   str   = "data/checkpoints/electfm"
    # Stage 2 过拟合退路（观察 5 epoch 后决定）
    stage2_overfit_patience: int = 5
    # LoRA 配置
    use_lora:         bool  = False    # 是否使用 LoRA
    lora_r:           int   = 8        # LoRA 低秩维度
    lora_alpha:       int   = 16       # LoRA 缩放因子
    lora_dropout:     float = 0.05     # LoRA dropout
    # v5：纯 spike head 模式
    spike_head_only:  bool  = False    # 完全冻结骨干，只训练 spike_head
    # v6：跨节点注意力模式
    cross_node_only:  bool  = False    # 冻结骨干，训练 CrossNodeAttention + spike_head


def _freeze(modules: List[nn.Module], lora_only: bool = False):
    """
    冻结给定模块列表的所有参数。

    参数
    ----
    modules : 要冻结的模块列表
    lora_only : 如果为 True，只冻结非 LoRA 参数（用于 LoRA 模式）
    """
    for m in modules:
        for name, p in m.named_parameters():
            if lora_only and "lora_" in name:
                # LoRA 模式下，保留 LoRA 参数可训练
                continue
            p.requires_grad = False


def _unfreeze(modules: List[nn.Module]):
    """解冻给定模块列表的所有参数。"""
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _setup_lora_training(model: nn.Module, train_lora: bool = True):
    """
    设置模型参数的 requires_grad，用于 LoRA 训练。

    参数
    ----
    model : ElecFM 模型
    train_lora : 如果为 True，只训练 LoRA 参数和 spike_head；
                 如果为 False，解冻所有参数
    """
    for name, p in model.named_parameters():
        if train_lora:
            # LoRA 模式：只训练 LoRA 参数和 spike_head
            if "lora_" in name or "spike_head" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        else:
            # 非 LoRA 模式：解冻所有参数（但 quant_head 会单独处理）
            p.requires_grad = True


def _run_epoch(
    model: ElecFM,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: TrainConfig,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    training: bool,
) -> dict:
    """跑一个 epoch，返回平均 loss 字典。"""
    model.train() if training else model.eval()
    totals = {"loss_total": 0.0, "loss_pinball": 0.0, "loss_spike": 0.0}
    n_batches = 0

    ctx_mgr = torch.enable_grad() if training else torch.no_grad()
    with ctx_mgr:
        for ctx, tgt, spike_lb in loader:
            ctx      = ctx.to(device)       # [B, context_len]
            tgt      = tgt.to(device)       # [B, horizon]
            spike_lb = spike_lb.to(device)  # [B, horizon]

            # autocast: CUDA/MPS 支持混合精度，CPU 不支持（enabled=False 为 no-op）
            _amp_ok = cfg.use_amp and device.type in ("cuda", "mps")
            _ac_dev = device.type if device.type in ("cuda", "mps") else "cpu"
            with torch.amp.autocast(_ac_dev, enabled=_amp_ok):
                q_pred, spike_logits = model(ctx)   # [B, H, 9] or [B, 3, H, 9]

                # V6：将 [B, 3, H] 张量展开为 [B*3, H] 再计算损失
                if tgt.dim() == 3:
                    B_loc, N_loc, H_loc = tgt.shape
                    tgt_loss      = tgt.reshape(B_loc * N_loc, H_loc)
                    q_pred_loss   = q_pred.reshape(B_loc * N_loc, H_loc, -1)
                    spike_logits_ = spike_logits.reshape(B_loc * N_loc, H_loc)
                    spike_lb_     = spike_lb.reshape(B_loc * N_loc, H_loc)
                else:
                    tgt_loss = tgt; q_pred_loss = q_pred
                    spike_logits_ = spike_logits; spike_lb_ = spike_lb

                loss, loss_dict = combined_loss(
                    tgt_loss, q_pred_loss, spike_logits_, spike_lb_,
                    lambda_pinball=cfg.lambda_pinball,
                    lambda_spike=cfg.lambda_spike,
                    spike_pos_weight=cfg.spike_pos_weight,
                )

            if training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                    optimizer.step()

            for k, v in loss_dict.items():
                totals[k] += v
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train(
    model: ElecFM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> str:
    """
    两阶段训练主函数。

    Returns
    -------
    path : 最优 checkpoint 路径
    """
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(cfg.checkpoint_dir, "electfm_best.pt")
    _cuda = torch.cuda.is_available()
    _mps  = torch.backends.mps.is_available()
    # GradScaler 只支持 CUDA；MPS 可以用 autocast 但不支持 GradScaler
    scaler = torch.amp.GradScaler("cuda") if cfg.use_amp and _cuda else None

    # ── Stage 1：冻结底层 + quant_head，只训顶层 transformer + spike_head ──────
    # 关键设计决策（v2 训练结果）：
    # quant_head 在全部两个 stage 中永久冻结，理由：
    #   - 预训练分位数校准良好（coverage≈0.775），25K 样本无法维持此质量
    #   - 上一轮训练中 quant_head 被 Pinball loss 损坏（coverage 降至 0.582）
    #   - 我们的创新在 spike_head，不在改进分位数头
    print("=" * 60)
    if cfg.cross_node_only:
        # ── v6：冻结骨干，训练 CrossNodeAttention + spike_head ──────────────
        print("Stage 1 (Cross Node Only)：冻结骨干，训练 CrossNodeAttention + spike_head")
        print("=" * 60)
        _freeze([model.base.tokenizer] + list(model.base.layers) + [model.base.quant_head])
        _unfreeze([model.cross_attn, model.base.spike_head])
        stage1_lr = cfg.stage1_lr
        stage1_epochs = cfg.stage1_epochs
        monitor_key = "loss_spike"
    elif cfg.spike_head_only:
        # ── v5/v7：完全冻结骨干，训练 spike_head（+ swiglu_adapter 如果有）──────
        has_adapter = hasattr(model, "swiglu_adapter") and model.swiglu_adapter is not None
        label = "Spike Head + SwiGLU Adapter" if has_adapter else "Spike Head Only"
        print(f"Stage 1 ({label})：冻结全部骨干 + quant_head，训练 spike_head" +
              (" + swiglu_adapter" if has_adapter else ""))
        print("=" * 60)
        _freeze([model.tokenizer] + list(model.layers) + [model.quant_head])
        trainable_modules = [model.spike_head]
        if has_adapter:
            trainable_modules.append(model.swiglu_adapter)
        _unfreeze(trainable_modules)
        stage1_lr = cfg.stage1_lr
        stage1_epochs = cfg.stage1_epochs
        monitor_key = "loss_spike"
    elif cfg.use_lora:
        print("Stage 1 (LoRA)：训练 LoRA 参数 + spike_head")
        print("=" * 60)
        _setup_lora_training(model, train_lora=True)
        _freeze([model.quant_head])
        stage1_lr = cfg.stage1_lr * 100
        stage1_epochs = max(5, cfg.stage1_epochs // 2)
        monitor_key = "loss_pinball"
    else:
        print("Stage 1：冻结 tokenizer + L0–L6 + quant_head，训练 L7–L14 + spike_head")
        print("=" * 60)
        _freeze([model.tokenizer] + list(model.layers[:7]) + [model.quant_head])
        _unfreeze([model.spike_head] + list(model.layers[7:]))
        stage1_lr = cfg.stage1_lr
        stage1_epochs = cfg.stage1_epochs
        monitor_key = "loss_pinball"
    print(f"  可训练参数：{_count_trainable(model):,}")

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage1_lr, weight_decay=cfg.weight_decay)

    best_monitor = float("inf")
    best_val_pinball = float("inf")   # 仅用于日志打印
    patience_counter = 0

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, opt1, cfg, device, scaler, training=True)
        va = _run_epoch(model, val_loader,   None,  cfg, device, scaler, training=False)
        elapsed = time.time() - t0

        if cfg.spike_head_only or cfg.cross_node_only:
            epoch_label = "S1-SpikeOnly" if cfg.spike_head_only else "S1-CrossNode"
            print(f"  {epoch_label} Epoch {epoch:2d}/{stage1_epochs} | "
                  f"train spike={tr['loss_spike']:.4f} | "
                  f"val spike={va['loss_spike']:.4f} | "
                  f"{elapsed:.0f}s")
        else:
            epoch_label = f"S1{'-LoRA' if cfg.use_lora else ''}"
            print(f"  {epoch_label} Epoch {epoch:2d}/{stage1_epochs} | "
                  f"train pinball={tr['loss_pinball']:.4f} spike={tr['loss_spike']:.4f} | "
                  f"val pinball={va['loss_pinball']:.4f} spike={va['loss_spike']:.4f} | "
                  f"{elapsed:.0f}s")

        monitor_val = va[monitor_key]

        # spike_head_only / cross_node_only 模式：val spike=0 时切换为 train spike 监控
        if (cfg.spike_head_only or cfg.cross_node_only) and monitor_key == "loss_spike" and monitor_val < 1e-3:
            if epoch == 1:
                print("  ⚠️  val spike ≈ 0（验证集无尖峰事件），改用 train spike 监控早停")
            monitor_val = tr["loss_spike"]   # fallback: 用训练 spike loss

        if monitor_val < best_monitor:
            best_monitor = monitor_val
            best_val_pinball = va["loss_pinball"]
            model.save(best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"  Early stop at S1 epoch {epoch}")
                break

    # ── v5/v6：跳过 Stage 2，直接返回 ──────────────────────────────────────────
    if cfg.spike_head_only or cfg.cross_node_only:
        print(f"\n  最优验证集 Spike Loss = {best_monitor:.4f}")
        print(f"  Checkpoint 保存至：{best_ckpt}")
        return best_ckpt

    # ── Stage 1 → Stage 2 过渡：恢复最优 checkpoint ─────────────────────────
    print(f"\n  从 Stage 1 最优 checkpoint 恢复（val {monitor_key}={best_monitor:.4f}）")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    # ── Stage 2：解冻骨干（除 quant_head），低 LR Cosine decay ───────────────
    # quant_head 仍然冻结：load_state_dict 不改变 requires_grad，Stage 1 冻结的
    # quant_head 在 Stage 2 自动保持冻结，无需额外操作。
    print()
    print("=" * 60)
    if cfg.use_lora:
        print("Stage 2 (LoRA)：继续训练 LoRA 参数 + spike_head，LR Cosine decay")
        print("=" * 60)
        # LoRA 模式下保持相同的参数冻结策略
        _setup_lora_training(model, train_lora=True)
        _freeze([model.quant_head])
        stage2_lr = cfg.stage2_lr * 100  # LoRA 可用更高学习率
        stage2_lr_min = cfg.stage2_lr_min * 100
        stage2_epochs = max(20, cfg.stage2_epochs // 2)  # LoRA 收敛更快
    else:
        print("Stage 2：解冻 tokenizer + 所有层（quant_head 仍冻结），LR Cosine decay")
        print("=" * 60)
        _unfreeze([model.tokenizer] + list(model.layers))
        # quant_head 保持冻结（load_state_dict 已保留 requires_grad=False）
        _freeze([model.quant_head])
        stage2_lr = cfg.stage2_lr
        stage2_lr_min = cfg.stage2_lr_min
        stage2_epochs = cfg.stage2_epochs
    print(f"  可训练参数：{_count_trainable(model):,}")

    opt2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage2_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=stage2_epochs, eta_min=stage2_lr_min)

    patience_counter = 0
    overfit_check_loss = float("inf")  # 用于 5 epoch 过拟合侦测

    for epoch in range(1, stage2_epochs + 1):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, opt2, cfg, device, scaler, training=True)
        va = _run_epoch(model, val_loader,   None,  cfg, device, scaler, training=False)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        epoch_label = f"S2{'-LoRA' if cfg.use_lora else ''}"
        print(f"  {epoch_label} Epoch {epoch:2d}/{stage2_epochs} | "
              f"train pinball={tr['loss_pinball']:.4f} | "
              f"val pinball={va['loss_pinball']:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.0f}s")

        # 过拟合早期检测（5 epoch 内）- 非 LoRA 模式下提醒
        if not cfg.use_lora:
            if epoch == 1:
                overfit_check_loss = va["loss_pinball"]
            elif epoch == cfg.stage2_overfit_patience:
                if va["loss_pinball"] > overfit_check_loss * 1.05:
                    print(f"\n  ⚠️  Stage 2 验证集 Pinball 在 {epoch} epoch 内上升 >5%")
                    print("  建议切换至 LoRA 退路（见 fusion_model_design_v3.md Section 6.2）")
                    print("  当前继续训练，但请监控后续 epoch")

        if va["loss_pinball"] < best_val_pinball:
            best_val_pinball = va["loss_pinball"]
            model.save(best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"  Early stop at S2 epoch {epoch}")
                break

    print(f"\n  最优验证集 Pinball = {best_val_pinball:.4f}")
    print(f"  Checkpoint 保存至：{best_ckpt}")
    return best_ckpt
