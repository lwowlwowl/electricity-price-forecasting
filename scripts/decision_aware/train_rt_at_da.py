#!/usr/bin/env python
"""训练D-1截止时点的独立24小时RT价格Transformer。

单个run只用验证集MAE做早停，不读取测试集。不同seed及其与DA/滚动RT的
最终组合，随后统一用完整双结算验证收益选择。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_rt_da import build_rt_at_da_datasets  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402


def _even_indices(length: int, limit: int | None):
    if limit is None or limit >= length:
        return list(range(length))
    return np.linspace(0, length - 1, limit, dtype=int).tolist()


def _move(batch: dict, device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, loader, device):
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            predictions.append(model(batch)["p_rt_at_da"].float().cpu())
            targets.append(batch["price_rt_at_da_tgt"].float().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return {
        "mae": float(torch.mean(torch.abs(prediction - target))),
        "rmse": float(torch.mean((prediction - target) ** 2).sqrt()),
        "days": int(target.shape[0]),
    }


def main() -> None:
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/rt_at_da_ercot_v1.yaml"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="data/results/rt_at_da_transformer_v1.json")
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.smoke:
        cfg.d_model = 32
        cfg.dim_ff = 64
        cfg.n_layers_enc = 1
        cfg.n_layers_fusion = 1
        cfg.batch_size = 8
        cfg.epochs = 1 if args.epochs is None else args.epochs
        args.max_train = args.max_train or 32
        args.max_val = args.max_val or 16

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats()
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")

    train_ds, val_ds, _, _ = build_rt_at_da_datasets(cfg)
    train_indices = _even_indices(len(train_ds), args.max_train)
    val_indices = _even_indices(len(val_ds), args.max_val)
    train_loader = DataLoader(
        Subset(train_ds, train_indices), cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices), 128, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )

    model = DecisionAwareRTAtDAForecaster(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_mae = float("inf")
    best_state = None
    best_epoch = None
    stale = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                prediction = model(batch)["p_rt_at_da"].float()
                loss = F.huber_loss(
                    prediction, batch["price_rt_at_da_tgt"].float(),
                    delta=cfg.huber_delta, reduction="mean",
                ) / cfg.pred_scale
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        validation = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation["mae"] < best_mae:
            best_mae = validation["mae"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("训练没有产生可保存状态")
    model.load_state_dict(best_state)
    model.to(device)
    validation = evaluate(model, val_loader, device)
    checkpoint = Path(cfg.checkpoint_path("best_mae"))
    if args.smoke:
        checkpoint = checkpoint.with_name(checkpoint.stem + "_smoke.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": cfg.to_dict(),
        "norm_stats": train_ds.norm_stats,
        "selection_metric": "validation.mae",
        "best_validation_mae": best_mae,
    }, checkpoint)
    report = {
        "run_type": "smoke" if args.smoke else "training",
        "contract": "ERCOT D-1 10:00 local -> delivery-day 24 hourly RT prices",
        "training_objective": "supervised Huber only",
        "epoch_selection_metric": "validation MAE (lower is better)",
        "system_selection_note": (
            "seed/checkpoint enters a later validation-only full dual-settlement grid; "
            "test is not read by this script"
        ),
        "device": str(device),
        "mixed_precision": amp_enabled,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "samples": {"train": len(train_indices), "validation": len(val_indices)},
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": validation,
        "checkpoint": str(checkpoint),
        "wall_time_seconds": round(time.time() - started, 2),
        "peak_cuda_memory_mb": (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            if device.type == "cuda" else 0.0
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved report: {output}")
    print(f"saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
