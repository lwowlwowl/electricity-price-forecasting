#!/usr/bin/env python
"""训练独立RT Transformer；按验证集滚动RT-only收益保存模型。"""
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

from decision_aware.backtest_rt import rolling_rt_backtest  # noqa: E402
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_rt import build_rt_datasets  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402


def _indices(length, limit, prefix=False):
    if limit is None or limit >= length:
        return list(range(length))
    if prefix:
        return list(range(limit))
    return np.linspace(0, length - 1, limit, dtype=int).tolist()


def _move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, loader, dataset, indices, device, cfg):
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            predictions.append(model(batch)["p_rt"].float().cpu())
            targets.append(batch["price_rt_tgt"].float().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    dates = [
        dataset.sample_time_contract(index)["executed_target"].date()
        for index in indices
    ]
    rolling = rolling_rt_backtest(prediction, target[:, 0], dates, cfg)
    rolling.pop("actual_actions")
    return {
        "mae_all_horizons": float(torch.mean(torch.abs(prediction - target))),
        "rmse_all_horizons": float(torch.mean((prediction - target) ** 2).sqrt()),
        "mae_executed_hour": float(torch.mean(torch.abs(prediction[:, 0] - target[:, 0]))),
        "rolling_rt_only": rolling,
    }


def main():
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_aware/rt_ercot_v1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="data/results/rt_transformer_v1.json")
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
        cfg.epochs = 2 if args.epochs is None else args.epochs
        args.max_train = args.max_train or 32
        args.max_val = args.max_val or 32
        args.max_test = args.max_test or 32

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_ds, val_ds, test_ds, _ = build_rt_datasets(cfg)
    train_indices = _indices(len(train_ds), args.max_train)
    val_indices = _indices(len(val_ds), args.max_val, prefix=True)
    test_indices = _indices(len(test_ds), args.max_test, prefix=True)
    train_loader = DataLoader(
        Subset(train_ds, train_indices), cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices), 128, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        Subset(test_ds, test_indices), 128, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )

    model = DecisionAwareRTForecaster(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_revenue = -float("inf")
    best_state = None
    history = []
    stale = 0
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                prediction = model(batch)["p_rt"].float()
                loss = F.huber_loss(
                    prediction, batch["price_rt_tgt"].float(),
                    delta=cfg.huber_delta, reduction="mean",
                ) / cfg.pred_scale
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, val_loader, val_ds, val_indices, device, cfg)
        if device.type == "cuda":
            # 验证阶段会临时产生完整预测张量；及时归还缓存，避免下一epoch碎片化。
            torch.cuda.empty_cache()
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "validation": validation}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        revenue = validation["rolling_rt_only"]["mean_daily_revenue"]
        if revenue > best_revenue:
            best_revenue = revenue
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("训练没有产生可保存状态")
    model.load_state_dict(best_state)
    model.to(device)
    validation = evaluate(model, val_loader, val_ds, val_indices, device, cfg)
    test = evaluate(model, test_loader, test_ds, test_indices, device, cfg)
    checkpoint = Path(cfg.checkpoint_path("best_rolling_revenue"))
    if args.smoke:
        checkpoint = checkpoint.with_name(checkpoint.stem + "_smoke.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": cfg.to_dict(),
        "norm_stats": train_ds.norm_stats,
        "selection_metric": "validation.rolling_rt_only.mean_daily_revenue",
    }, checkpoint)
    report = {
        "run_type": "smoke" if args.smoke else "training",
        "contract": "hourly issue; [t-168,t) -> [t,t+4); execute first step",
        "training_objective": "supervised Huber only",
        "selection_metric": "validation RT-only rolling mean daily revenue",
        "revenue_scope": "RT-only diagnostic; excludes DA position and dual settlement",
        "device": str(device),
        "mixed_precision": amp_enabled,
        "batch_size": cfg.batch_size,
        "wall_time_seconds": round(time.time() - started, 2),
        "peak_cuda_memory_mb": (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            if device.type == "cuda" else 0.0
        ),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "config": cfg.to_dict(),
        "samples": {"train": len(train_indices), "validation": len(val_indices),
                    "test": len(test_indices)},
        "history": history,
        "best_epoch": max(
            history,
            key=lambda row: row["validation"]["rolling_rt_only"]["mean_daily_revenue"],
        )["epoch"],
        "best_validation": validation,
        "test": test,
        "checkpoint": str(checkpoint),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved report: {output}")
    print(f"saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
