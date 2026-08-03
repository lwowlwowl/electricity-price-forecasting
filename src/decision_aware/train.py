"""train.py — 先行版训练回路（镜像 src/archive/fusion_model/train.py 脚手架）.

AdamW(requires_grad 过滤) + autocast(MPS 有 autocast 无 scaler，CUDA 才有 scaler)
+ clip_grad_norm_(1.0) + val 上 early-stop + torch.save state_dict(weights_only load)。
α/β 退火每 epoch 更新。日志含 L_pred/L_bus/R/R*/regret/α/β/val MAE。
"""
from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PilotConfig
from .dataset import collate
from .loss import total_loss, anneal_alpha_beta
from .model import DecisionAwareTSFM
from .policy import BESSSimulator, STEPolicy, TopKPolicy


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loaders(cfg: PilotConfig, train_ds, val_ds):
    nw = cfg.num_workers
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=nw)
    return train_loader, val_loader


def _run_epoch(model, loader, simulator, policy, cfg, device,
               optimizer=None, alpha=1.0, beta=0.0, amp_ok=False):
    """训练(optimizer!=None) 或评估(optimizer==None) 一个 epoch。返回平均 metrics。"""
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.amp.autocast(device_type=device.type, enabled=amp_ok)
    agg: Dict[str, float] = {}
    n = 0
    with torch.enable_grad() if is_train else torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with ctx:
                out = model(batch)
                loss, m = total_loss(out["p_da"], batch["price_tgt"],
                                     simulator, policy, alpha, beta, cfg.huber_delta,
                                     cfg.pred_scale, cfg.bus_scale, oracle=cfg.oracle_type)
                mae = (out["p_da"] - batch["price_tgt"]).abs().mean()
                m["mae"] = float(mae.detach().item())
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
            del loss, out
    return {k: v / max(1, n) for k, v in agg.items()}


def train(model: DecisionAwareTSFM, train_ds, val_ds, cfg: PilotConfig,
          device: torch.device = None, verbose: bool = True) -> Dict:
    if device is None:
        device = get_device()
    model = model.to(device)
    simulator = BESSSimulator(cfg.bess_power_mw, cfg.bess_energy_mwh, cfg.bess_eta,
                              cfg.bess_init_soc_frac)
    if cfg.policy_type == "topk":
        policy = TopKPolicy(cfg.topk_k_charge, cfg.topk_k_discharge)
    else:
        policy = STEPolicy(cfg.ste_k)

    train_loader, val_loader = _make_loaders(cfg, train_ds, val_ds)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp_ok = cfg.use_amp and device.type in ("cuda", "mps")

    best_val = float("inf")
    best_path = cfg.checkpoint_path("best")
    patience = 0
    history = []

    for epoch in range(cfg.epochs):
        alpha, beta = anneal_alpha_beta(epoch, cfg)
        t0 = time.time()
        tr = _run_epoch(model, train_loader, simulator, policy, cfg, device,
                        optimizer=optimizer, alpha=alpha, beta=beta, amp_ok=amp_ok)
        va = _run_epoch(model, val_loader, simulator, policy, cfg, device,
                        optimizer=None, alpha=alpha, beta=beta, amp_ok=amp_ok)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "val": va, "alpha": alpha, "beta": beta})
        if verbose:
            print(f"  Epoch {epoch+1:2d}/{cfg.epochs} "
                  f"| tr loss={tr['loss_total']:.2f} pred={tr['loss_pred']:.2f} "
                  f"bus={tr['loss_bus']:.2f} regret={tr['regret']:.1f} mae={tr.get('mae',0):.2f} "
                  f"| val loss={va['loss_total']:.2f} pred={va['loss_pred']:.2f} "
                  f"bus={va['loss_bus']:.2f} regret={va['regret']:.1f} mae={va.get('mae',0):.2f} "
                  f"| α={alpha:.2f} β={beta:.2f} | {elapsed:.0f}s")

        # 监控指标由 cfg.monitor 决定（decision-aware 默认 val regret；可切 "mae"）。
        # 都 lower=better。不能监控 val loss_total——退火下 β=0 的 epoch 1 必然最小，误选。
        monitor = va["regret"] if cfg.monitor == "regret" else va["mae"]
        if monitor < best_val:
            best_val = monitor
            patience = 0
            model.save(best_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                if verbose:
                    print(f"  early stop @ epoch {epoch+1} (patience={cfg.early_stop_patience})")
                break

    # 末次也存 last
    model.save(cfg.checkpoint_path("last"))
    if verbose:
        print(f"  best val {cfg.monitor}={best_val:.3f}  ckpt→{best_path}")
    return {"best_val": best_val, "best_path": best_path, "history": history}


def evaluate(model, test_ds, cfg: PilotConfig, device: torch.device = None,
             ckpt_path: str = None) -> Dict:
    """在 test 段评估：加载 ckpt(可选)，报 MAE/SMAPE/平均 Regret/平均收益。"""
    if device is None:
        device = get_device()
    model = model.to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    simulator = BESSSimulator(cfg.bess_power_mw, cfg.bess_energy_mwh, cfg.bess_eta,
                              cfg.bess_init_soc_frac)
    if cfg.policy_type == "topk":
        policy = TopKPolicy(cfg.topk_k_charge, cfg.topk_k_discharge)
    else:
        policy = STEPolicy(cfg.ste_k)
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=cfg.num_workers)

    model.eval()
    preds, acts = [], []
    R_m, R_s = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            preds.append(out["p_da"].cpu().numpy())
            acts.append(batch["price_tgt"].cpu().numpy())
            from .policy import oracle_revenue, lp_oracle_revenue
            R = simulator(policy(out["p_da"]), batch["price_tgt"])
            R_m.append(R.cpu().numpy())
            if cfg.oracle_type == "lp":
                R_s.append(lp_oracle_revenue(batch["price_tgt"], simulator).cpu().numpy())
            else:
                R_s.append(oracle_revenue(batch["price_tgt"], simulator).cpu().numpy())
    preds = np.concatenate(preds); acts = np.concatenate(acts)
    R_m = np.concatenate(R_m); R_s = np.concatenate(R_s)
    mae = float(np.mean(np.abs(preds - acts)))
    smape = float(np.mean(2 * np.abs(preds - acts) / (np.abs(preds) + np.abs(acts) + 1e-8)))
    return {
        "mae": mae, "smape": smape,
        "R_model": float(R_m.mean()), "R_star": float(R_s.mean()),
        "regret": float((R_s - R_m).mean()),
        "n_samples": int(len(preds)),
    }
