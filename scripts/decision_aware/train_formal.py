#!/usr/bin/env python
"""train_formal.py — 正式版训练入口（Hard TopK + 零阶梯度，w10 §6）.

与 train_pilot_v3.py 的核心区别:
- 策略: STE/soft-TopK → HardTopKPolicy（不可微）
- 损失: regret 直接退火 → L_proxy 零阶梯度注入（L = α·L_pred + β·L_proxy）
- 梯度: 穿过 STE sigmoid → 零阶双点高斯估计（不经过 soft 近似，不会正反馈爆炸）

v3 失败原因: soft TopK/STE 在高波动 DA 价（std=370）上正反馈把预测推到上千/NaN。
正式版修复: 零阶梯度对 p̂ 加 ±εz 扰动 → 跑硬策略仿真 → 估计 ĝ → L_proxy=p̂·ĝ 注入。

用法:
  external/chronos-forecasting/.venv/bin/python \\
    scripts/decision_aware/train_formal.py \\
    --config configs/decision_aware/formal_ercot.yaml [--no-early-stop] [--no-oracle-train]

  --no-oracle-train: 训练时跳过 LP oracle（省时间，regret 只在 val 上算）
"""
from __future__ import annotations
import argparse, os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
os.chdir(_ROOT)

# 强制 stdout 行缓冲（否则管道/文件输出会 block-buffer，看不到实时日志）
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from decision_aware.config import PilotConfig
from decision_aware.dataset_v3 import build_datasets_v3, collate_v3
from decision_aware.model import DecisionAwareTSFM
from decision_aware.policy import BESSSimulator, HardTopKPolicy, lp_oracle_revenue
from decision_aware.loss import huber_loss, anneal_alpha_beta, total_loss_zo
from decision_aware.zero_order import compute_epsilon
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(model, loader, sim, pol_hard, cfg, dev, eps_da, eps_rt,
               opt=None, alpha=1.0, beta=0.0, amp_ok=False, oracle_train=True):
    """训练(opt!=None) 或评估(opt==None) 一个 epoch。

    核心改动（vs v3）: 用 total_loss_zo（零阶梯度 + L_proxy）替代直接 regret。
    """
    is_train = opt is not None
    model.train(is_train)
    ctx = torch.amp.autocast(device_type=dev.type, enabled=amp_ok)
    agg, n = {}, 0
    grad_fn = torch.enable_grad() if is_train else torch.no_grad()
    with grad_fn:
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with ctx:
                out = model(batch)
                price_da_tgt = batch["price_da_tgt"]
                price_rt_tgt = batch["price_rt_tgt"]

                # 正式版损失：零阶梯度 + L_proxy
                loss, m = total_loss_zo(
                    out, price_da_tgt, price_rt_tgt,
                    sim, pol_hard, alpha, beta, cfg, eps_da, eps_rt,
                    oracle_train=oracle_train,
                )

            if is_train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
            del loss, out
    return {k: v / max(1, n) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decision_aware/formal_ercot.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--no-oracle-train", action="store_true",
                    help="训练时跳过 LP oracle（省时间，regret 只在 val 上算）")
    args = ap.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if args.epochs: cfg.epochs = args.epochs
    if args.no_early_stop: cfg.early_stop_patience = 99
    torch.manual_seed(cfg.seed)

    print("=" * 70)
    print(f"正式版训练 | {cfg.market} {cfg.node} | d={cfg.d_model} "
          f"| policy={cfg.policy_type} | zo_K={cfg.zo_K} zo_rho={cfg.zo_rho}")
    print(f"| dual_settlement={cfg.use_dual_settlement} | oracle_train={not args.no_oracle_train}")
    print("=" * 70)

    # 数据
    train_ds, val_ds, test_ds, _ = build_datasets_v3(cfg)
    model = DecisionAwareTSFM(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数: {n_params:,} (~{n_params/1e6:.2f}M)")

    # σ_m：每任务独立（w10 §6.1 ε_m=ρ·σ_m），DA 与 RT 各自的 std
    sigma_da = cfg.zo_sigma if cfg.zo_sigma > 0 else train_ds.norm_stats["price_da"]["std"]
    sigma_rt = cfg.zo_sigma if cfg.zo_sigma > 0 else train_ds.norm_stats["price_rt"]["std"]
    eps_da = compute_epsilon(sigma_da, cfg.zo_rho)
    eps_rt = compute_epsilon(sigma_rt, cfg.zo_rho)
    print(f"σ_DA={sigma_da:.2f}  ε_DA={eps_da:.2f}   σ_RT={sigma_rt:.2f}  ε_RT={eps_rt:.2f}  (ρ={cfg.zo_rho})")

    dev = get_device()
    model = model.to(dev)
    sim = BESSSimulator(cfg.bess_power_mw, cfg.bess_energy_mwh, cfg.bess_eta,
                        cfg.bess_init_soc_frac, kappa=cfg.bess_kappa,
                        soc_min=cfg.bess_soc_min, soc_max=cfg.bess_soc_max,
                        e_cyc=cfg.bess_e_cyc)
    pol_hard = HardTopKPolicy(cfg.topk_k_charge, cfg.topk_k_discharge,
                              spread_threshold=cfg.resolved_spread_threshold)
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

    print(f"\n训练开始（~{len(train_ds)//cfg.batch_size} batch/epoch × {cfg.epochs} epoch）")
    print(f"每 batch 零阶仿真次数: {cfg.zo_K * 2}（K={cfg.zo_K} 成对方向）")
    print()

    for epoch in range(cfg.epochs):
        alpha, beta = anneal_alpha_beta(epoch, cfg)
        t0 = time.time()
        tr = _run_epoch(model, tr_loader, sim, pol_hard, cfg, dev, eps_da, eps_rt,
                        opt, alpha, beta, amp_ok, oracle_train=not args.no_oracle_train)
        va = _run_epoch(model, va_loader, sim, pol_hard, cfg, dev, eps_da, eps_rt,
                        None, alpha, beta, amp_ok, oracle_train=True)  # val 始终算 oracle
        el = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{cfg.epochs} "
              f"| tr loss={tr['loss_total']:.2f} pred={tr['loss_pred']:.2f} "
              f"proxy={tr.get('loss_proxy',0):.4f} g_norm={tr.get('g_norm',0):.1f} "
              f"regret={tr.get('regret',0):.1f} mae={tr.get('mae',0):.2f} "
              f"rmse={tr.get('rmse',0):.2f} "
              f"| val loss={va['loss_total']:.2f} pred={va['loss_pred']:.2f} "
              f"regret={va['regret']:.1f} mae={va.get('mae',0):.2f} "
              f"rmse={va.get('rmse',0):.2f} "
              f"| α={alpha:.2f} β={beta:.2f} | {el:.0f}s")

        monitor = va["regret"] if cfg.monitor == "regret" else va.get("mae", float("inf"))
        if monitor < best_val:
            best_val = monitor
            patience = 0
            model.save(best_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"  early stop @ epoch {epoch+1}")
                break

    model.save(cfg.checkpoint_path("last"))
    torch.save(train_ds.norm_stats, cfg.checkpoint_path("stats"))
    print(f"\n  best val {cfg.monitor}={best_val:.3f}  ckpt→{best_path}")

    # Test 评估
    print("\n" + "=" * 70)
    print("Test 段评估")
    print("=" * 70)
    te_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                           collate_fn=collate_v3, num_workers=cfg.num_workers)
    model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
    te = _run_epoch(model, te_loader, sim, pol_hard, cfg, dev, eps_da, eps_rt,
                    None, 0, 1, amp_ok, oracle_train=True)
    r_star = te["R_star"]
    r_model = te["R_model"]
    # w10 §7: Oracle 非正时不报告 PCR（无意义）
    pcr = (r_model / r_star * 100) if r_star > 0 else None
    pcr_str = f"  PCR={pcr:.0f}%" if pcr is not None else "  PCR=N/A(Oracle≤0)"
    print(f"  MAE={te['mae']:.3f}  RMSE={te.get('rmse',0):.3f}  R_model={r_model:.1f}  "
          f"R*_LP={r_star:.1f}  regret={te['regret']:.1f}{pcr_str}")

    # 验证标准
    print("\n验证标准:")
    print(f"  不爆炸 (val MAE < 100):  {'✅' if te['mae'] < 100 else '❌'} (MAE={te['mae']:.1f})")
    print(f"  R_model 转正 (> 0):      {'✅' if r_model > 0 else '❌'} (R={r_model:.1f})")
    if pcr is not None:
        print(f"  PCR > 0:                 {'✅' if pcr > 0 else '❌'} (PCR={pcr:.1f}%)")
    else:
        print(f"  PCR:                     — (Oracle≤0，w10 §7 不报告)")
    print(f"  训练完成:                ✅")

    print("\n✅ 正式版跑通")


if __name__ == "__main__":
    main()
