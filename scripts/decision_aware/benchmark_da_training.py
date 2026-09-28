#!/usr/bin/env python
"""只跑少量真实batch，估算独立DA训练速度；不保存checkpoint。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_da import build_da_datasets  # noqa: E402
from decision_aware.loss_da import da_decision_aware_loss  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.policy import BESSSimulator, HardTopKPolicy  # noqa: E402
from decision_aware.zero_order import compute_epsilon  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_aware/da_ercot_v1.yaml")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    cfg = PilotConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, _, _, _ = build_da_datasets(cfg)
    batch = next(iter(DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)))
    batch = {key: value.to(device) for key, value in batch.items()}
    model = DecisionAwareDAForecaster(cfg).to(device)
    simulator = BESSSimulator(
        cfg.bess_power_mw, cfg.bess_energy_mwh, cfg.bess_eta,
        cfg.bess_init_soc_frac, kappa=cfg.bess_kappa,
        soc_min=cfg.bess_soc_min, soc_max=cfg.bess_soc_max,
        e_cyc=cfg.bess_e_cyc,
    ).to(device)
    policy = HardTopKPolicy(
        cfg.topk_k_charge, cfg.topk_k_discharge,
        spread_threshold=cfg.resolved_spread_threshold,
    ).to(device)
    epsilon = compute_epsilon(train_ds.norm_stats["price_da"]["std"], cfg.zo_rho)

    def run(beta: float):
        durations = []
        for step in range(args.steps + 1):
            started = time.perf_counter()
            prediction = model(batch)["p_da"]
            loss, _ = da_decision_aware_loss(
                prediction, batch["price_da_tgt"], simulator, policy,
                alpha=1.0 - beta, beta=beta, cfg=cfg, epsilon=epsilon,
            )
            model.zero_grad(set_to_none=True)
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if step > 0:  # 第一轮用于预热
                durations.append(elapsed)
        return sum(durations) / len(durations)

    prediction_only = run(beta=0.0)
    decision_aware = run(beta=0.5)
    batches_per_epoch = (len(train_ds) + cfg.batch_size - 1) // cfg.batch_size
    print(json.dumps({
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "train_samples": len(train_ds),
        "batch_size": cfg.batch_size,
        "batches_per_epoch": batches_per_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds_per_prediction_batch": prediction_only,
        "seconds_per_decision_batch": decision_aware,
        "estimated_prediction_epoch_minutes": prediction_only * batches_per_epoch / 60,
        "estimated_decision_epoch_minutes": decision_aware * batches_per_epoch / 60,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
