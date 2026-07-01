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


def _freeze(modules: List[nn.Module]):
    """冻结给定模块列表的所有参数。"""
    for m in modules:
        for p in m.parameters():
            p.requires_grad = False


def _unfreeze(modules: List[nn.Module]):
    """解冻给定模块列表的所有参数。"""
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
                q_pred, spike_logits = model(ctx)   # [B, H, 9], [B, H]
                loss, loss_dict = combined_loss(
                    tgt, q_pred, spike_logits, spike_lb,
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

    # ── Stage 1：冻结底层，只训顶层 + 两个头 ────────────────────────────────
    # Spike layer = 新 L6（原 L7），Stage 1 连同冻结避免扰动尖峰表征
    print("=" * 60)
    print("Stage 1：冻结 tokenizer + L0–L6（含 spike layer），训练 L7–L14 + heads")
    print("=" * 60)
    _freeze([model.tokenizer] + list(model.layers[:7]))
    _unfreeze([model.quant_head, model.spike_head] + list(model.layers[7:]))
    print(f"  可训练参数：{_count_trainable(model):,}")

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.stage1_lr, weight_decay=cfg.weight_decay)

    best_val_pinball = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.stage1_epochs + 1):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, opt1, cfg, device, scaler, training=True)
        va = _run_epoch(model, val_loader,   None,  cfg, device, scaler, training=False)
        elapsed = time.time() - t0

        print(f"  S1 Epoch {epoch:2d}/{cfg.stage1_epochs} | "
              f"train pinball={tr['loss_pinball']:.4f} spike={tr['loss_spike']:.4f} | "
              f"val pinball={va['loss_pinball']:.4f} spike={va['loss_spike']:.4f} | "
              f"{elapsed:.0f}s")

        if va["loss_pinball"] < best_val_pinball:
            best_val_pinball = va["loss_pinball"]
            model.save(best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"  Early stop at S1 epoch {epoch}")
                break

    # ── Stage 1 → Stage 2 过渡：恢复最优 checkpoint ─────────────────────────
    # 若 Stage 1 发生 early stop，当前模型状态不是最优。
    # 必须从 best_ckpt 重新加载，确保 Stage 2 从 Stage 1 最好的状态出发。
    print(f"\n  从 Stage 1 最优 checkpoint 恢复（val pinball={best_val_pinball:.4f}）")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    # ── Stage 2：全层解冻，低 LR Cosine decay ───────────────────────────────
    print()
    print("=" * 60)
    print("Stage 2：全层解冻，LR Cosine decay")
    print("=" * 60)
    _unfreeze([model.tokenizer] + list(model.layers))
    print(f"  可训练参数：{_count_trainable(model):,}")

    opt2 = torch.optim.AdamW(
        model.parameters(), lr=cfg.stage2_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=cfg.stage2_epochs, eta_min=cfg.stage2_lr_min)

    patience_counter = 0
    overfit_check_loss = float("inf")  # 用于 5 epoch 过拟合侦测

    for epoch in range(1, cfg.stage2_epochs + 1):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, opt2, cfg, device, scaler, training=True)
        va = _run_epoch(model, val_loader,   None,  cfg, device, scaler, training=False)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"  S2 Epoch {epoch:2d}/{cfg.stage2_epochs} | "
              f"train pinball={tr['loss_pinball']:.4f} | "
              f"val pinball={va['loss_pinball']:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.0f}s")

        # 过拟合早期检测（5 epoch 内）
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
