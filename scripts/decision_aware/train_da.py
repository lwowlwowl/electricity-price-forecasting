#!/usr/bin/env python
"""训练独立DA Transformer；按验证集平均净收益保存最佳checkpoint。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_da import build_da_datasets  # noqa: E402
from decision_aware.loss import anneal_alpha_beta  # noqa: E402
from decision_aware.loss_da import da_decision_aware_loss  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.policy import (  # noqa: E402
    BESSSimulator,
    HardTopKPolicy,
    lp_oracle_revenue,
)
from decision_aware.zero_order import compute_epsilon  # noqa: E402


def _even_indices(length: int, limit: int | None):
    if limit is None or limit >= length:
        return list(range(length))
    return np.linspace(0, length - 1, limit, dtype=int).tolist()


def _to_device(batch: dict, device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def _simulator_and_policy(cfg: PilotConfig, device: torch.device):
    simulator = BESSSimulator(
        cfg.bess_power_mw,
        cfg.bess_energy_mwh,
        cfg.bess_eta,
        cfg.bess_init_soc_frac,
        kappa=cfg.bess_kappa,
        soc_min=cfg.bess_soc_min,
        soc_max=cfg.bess_soc_max,
        e_cyc=cfg.bess_e_cyc,
    ).to(device)
    policy = HardTopKPolicy(
        cfg.topk_k_charge,
        cfg.topk_k_discharge,
        spread_threshold=cfg.resolved_spread_threshold,
    ).to(device)
    return simulator, policy


def evaluate(model, loader, simulator, policy, device, with_oracle: bool):
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            predictions.append(model(batch)["p_da"].float())
            targets.append(batch["price_da_tgt"].float())
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        revenue = simulator(policy(prediction), target)
        report = {
            "mae": float(torch.mean(torch.abs(prediction - target)).cpu()),
            "rmse": float(torch.mean((prediction - target) ** 2).sqrt().cpu()),
            "mean_revenue": float(revenue.mean().cpu()),
            "positive_day_rate": float((revenue > 0).float().mean().cpu()),
            "days": int(target.shape[0]),
        }
        if with_oracle:
            oracle = lp_oracle_revenue(target, simulator)
            report.update({
                "mean_oracle": float(oracle.mean().cpu()),
                "mean_regret": float((oracle - revenue).mean().cpu()),
            })
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_aware/da_ercot_v1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="data/results/da_transformer_v1.json")
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.smoke:
        # smoke只验证端到端链路，不作为正式模型结论。
        cfg.d_model = 32
        cfg.dim_ff = 64
        cfg.n_layers_enc = 1
        cfg.n_layers_fusion = 1
        cfg.batch_size = 8
        cfg.epochs = 1 if args.epochs is None else args.epochs
        args.max_train = args.max_train or 32
        args.max_val = args.max_val or 16
        args.max_test = args.max_test or 16

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds, _ = build_da_datasets(cfg)
    train_subset = Subset(train_ds, _even_indices(len(train_ds), args.max_train))
    val_subset = Subset(val_ds, _even_indices(len(val_ds), args.max_val))
    test_subset = Subset(test_ds, _even_indices(len(test_ds), args.max_test))
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    model = DecisionAwareDAForecaster(cfg).to(device)
    simulator, policy = _simulator_and_policy(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    epsilon = compute_epsilon(train_ds.norm_stats["price_da"]["std"], cfg.zo_rho)
    history = []
    best_revenue = -float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        alpha, beta = anneal_alpha_beta(epoch, cfg)
        epoch_metrics = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            prediction = model(batch)["p_da"]
            loss, metrics = da_decision_aware_loss(
                prediction,
                batch["price_da_tgt"],
                simulator,
                policy,
                alpha,
                beta,
                cfg,
                epsilon,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_metrics.append(metrics)

        validation = evaluate(
            model, val_loader, simulator, policy, device, with_oracle=False
        )
        training = {
            key: float(np.mean([row[key] for row in epoch_metrics]))
            for key in epoch_metrics[0]
        }
        row = {
            "epoch": epoch + 1,
            "alpha": alpha,
            "beta": beta,
            "train": training,
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if validation["mean_revenue"] > best_revenue:
            best_revenue = validation["mean_revenue"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("训练没有产生可保存的模型状态")
    model.load_state_dict(best_state)
    model.to(device)
    validation = evaluate(model, val_loader, simulator, policy, device, True)
    test = evaluate(model, test_loader, simulator, policy, device, True)

    checkpoint = Path(cfg.checkpoint_path("best_revenue"))
    if args.smoke:
        checkpoint = checkpoint.with_name(checkpoint.stem + "_smoke.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": cfg.to_dict(),
        "norm_stats": train_ds.norm_stats,
        "selection_metric": "validation.mean_revenue",
        "best_validation_revenue": best_revenue,
    }, checkpoint)
    report = {
        "run_type": "smoke" if args.smoke else "training",
        "contract": "ERCOT D-1 10:00 local -> delivery day 00:00-23:00",
        "selection_metric": "validation.mean_revenue",
        "revenue_scope": "DA standalone physical-execution proxy; not full dual settlement",
        "device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "samples": {
            "train": len(train_subset),
            "validation": len(val_subset),
            "test": len(test_subset),
        },
        "epsilon": epsilon,
        "history": history,
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
