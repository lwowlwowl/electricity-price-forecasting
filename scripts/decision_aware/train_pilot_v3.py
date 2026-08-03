#!/usr/bin/env python
"""train_pilot_v3.py — 先行版 v3 训练入口（6.5年 DA+RT 双结算）.

用法:
  external/chronos-forecasting/.venv/bin/python \\
    scripts/decision_aware/train_pilot_v3.py \\
    --config configs/decision_aware/pilot_ercot_v3.yaml [--epochs N] [--no-early-stop]

v3 改动（相对 v1/v2）:
- 数据: ERCOT 6.5年统一表（DA+RT 双价，真节假日），~56000 训练样本
- 模型: d=256/2层 ~5M 参数（v1/v2: d=128/1层 ~1.1M）
- 收益: 真双结算 pDA·u + pRT·Δu − κ|u|（v1/v2: 单腿 RT）
- BESS: w10 规范参数（η=0.95, κ=27, SOC 0.4-3.6）
- 策略: TopK + LP Oracle
"""
from __future__ import annotations
import argparse, os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
os.chdir(_ROOT)

import torch
from decision_aware.config import PilotConfig
from decision_aware.dataset_v3 import build_datasets_v3, collate_v3
from decision_aware.model import DecisionAwareTSFM
from decision_aware.policy import BESSSimulator, TopKPolicy, STEPolicy, lp_oracle_revenue
from decision_aware.loss import huber_loss, anneal_alpha_beta
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(model, loader, sim, pol, cfg, dev, opt=None, alpha=1.0, beta=0.0, amp_ok=False):
    is_train = opt is not None
    model.train(is_train)
    ctx = torch.amp.autocast(device_type=dev.type, enabled=amp_ok)
    agg, n = {}, 0
    with torch.enable_grad() if is_train else torch.no_grad():
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with ctx:
                out = model(batch)
                # v3: 用 DA 价做预测损失 + 双结算收益
                p_da = out["p_da"]
                price_da_tgt = batch["price_da_tgt"]
                price_rt_tgt = batch["price_rt_tgt"]
                l_pred = huber_loss(p_da, price_da_tgt, delta=cfg.huber_delta)
                # 双结算收益：DA 腿按 DA 价、RT 腿按 RT 价
                u = pol(p_da)
                R_model = sim(u, price_rt_tgt, price_da=price_da_tgt if cfg.use_dual_settlement else None)
                R_star = lp_oracle_revenue(price_rt_tgt, sim)
                regret = (R_star - R_model).mean()
                l_pred_n = l_pred / cfg.pred_scale
                l_bus_n = regret / cfg.bus_scale
                loss = alpha * l_pred_n + beta * l_bus_n
                mae = (p_da - price_da_tgt).abs().mean()
                m = {"loss_total": float(loss.detach().item()),
                     "loss_pred": float(l_pred.detach().item()),
                     "loss_bus": float(regret.detach().item()),
                     "R_model": float(R_model.detach().mean().item()),
                     "R_star": float(R_star.detach().mean().item()),
                     "regret": float(regret.detach().item()),
                     "mae": float(mae.detach().item())}
            if is_train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            for k, v in m.items(): agg[k] = agg.get(k, 0.0) + v
            n += 1; del loss, out
    return {k: v / max(1, n) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decision_aware/pilot_ercot_v3.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-early-stop", action="store_true")
    args = ap.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if args.epochs: cfg.epochs = args.epochs
    if args.no_early_stop: cfg.early_stop_patience = 99
    torch.manual_seed(cfg.seed)

    print("="*70)
    print(f"先行版 v3 训练 | {cfg.market} {cfg.node} | d={cfg.d_model} "
          f"| epochs={cfg.epochs} | dual_settlement={cfg.use_dual_settlement}")
    print("="*70)

    train_ds, val_ds, test_ds, _ = build_datasets_v3(cfg)
    model = DecisionAwareTSFM(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数: {n:,} (~{n/1e6:.2f}M)")

    dev = get_device()
    model = model.to(dev)
    sim = BESSSimulator(cfg.bess_power_mw, cfg.bess_energy_mwh, cfg.bess_eta,
                        cfg.bess_init_soc_frac, kappa=cfg.bess_kappa,
                        soc_min=cfg.bess_soc_min, soc_max=cfg.bess_soc_max)
    pol = TopKPolicy(cfg.topk_k_charge, cfg.topk_k_discharge) if cfg.policy_type == "topk" else STEPolicy(cfg.ste_k)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp_ok = cfg.use_amp and dev.type in ("cuda", "mps")

    tr_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                           collate_fn=collate_v3, num_workers=cfg.num_workers, drop_last=True)
    va_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                           collate_fn=collate_v3, num_workers=cfg.num_workers)

    best_val, patience = float("inf"), 0
    best_path = cfg.checkpoint_path("best")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    for epoch in range(cfg.epochs):
        alpha, beta = anneal_alpha_beta(epoch, cfg)
        t0 = time.time()
        tr = _run_epoch(model, tr_loader, sim, pol, cfg, dev, opt, alpha, beta, amp_ok)
        va = _run_epoch(model, va_loader, sim, pol, cfg, dev, None, alpha, beta, amp_ok)
        el = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{cfg.epochs} "
              f"| tr loss={tr['loss_total']:.2f} pred={tr['loss_pred']:.2f} "
              f"bus={tr['loss_bus']:.2f} regret={tr['regret']:.1f} mae={tr.get('mae',0):.2f} "
              f"| val loss={va['loss_total']:.2f} pred={va['loss_pred']:.2f} "
              f"bus={va['loss_bus']:.2f} regret={va['regret']:.1f} mae={va.get('mae',0):.2f} "
              f"| α={alpha:.2f} β={beta:.2f} | {el:.0f}s")
        monitor = va["regret"] if cfg.monitor == "regret" else va["mae"]
        if monitor < best_val:
            best_val = monitor; patience = 0; model.save(best_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"  early stop @ epoch {epoch+1}"); break

    model.save(cfg.checkpoint_path("last"))
    torch.save(train_ds.norm_stats, cfg.checkpoint_path("stats"))
    print(f"  best val {cfg.monitor}={best_val:.3f}  ckpt→{best_path}")

    # Test 评估
    print("\n" + "="*70); print("Test 段评估"); print("="*70)
    te_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                           collate_fn=collate_v3, num_workers=cfg.num_workers)
    model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
    te = _run_epoch(model, te_loader, sim, pol, cfg, dev, None, 0, 1, amp_ok)
    print(f"  MAE={te['mae']:.3f}  R={te['R_model']:.1f}  R*_LP={te['R_star']:.1f}  "
          f"regret={te['regret']:.1f}  占LP={te['R_model']/te['R_star']*100:.0f}%")
    print("\n✅ v3 跑通")


if __name__ == "__main__":
    main()
