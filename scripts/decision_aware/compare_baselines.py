#!/usr/bin/env python
"""compare_baselines.py — B 对比：foundation 零样本 vs 你的 decision-aware 模型，同台比收益。

每个模型（你的 β=1 + TimesFM/Chronos2/Toto/Toto2 零样本）在相同 test 起报点上：
  预测 24h 电价 → 喂进【同一套 BESS 模拟器 + greedy 策略】→ 算 R_model、R*(oracle)、regret。
所有模型过同一个"收益裁判台"，不比 MSE 比 regret——这是 decision-aware 的卖点。

用法:
  external/chronos-forecasting/.venv/bin/python scripts/decision_aware/compare_baselines.py \
      [--config ...] [--n-origins 30] [--ckpt last] [--models timesfm,chronos2,toto,toto2]
"""
from __future__ import annotations
import argparse, os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
sys.path.insert(0, os.path.join(_ROOT, "src", "models"))
os.chdir(_ROOT)

import numpy as np, pandas as pd, torch
import loader
from decision_aware.config import PilotConfig, ALL_COVARIATES_V12 as ALL_COVARIATES
from decision_aware.policy import BESSSimulator, STEPolicy, lp_oracle_revenue as oracle_revenue
from decision_aware.dataset import DecisionAwareDataset
from decision_aware.forecaster import DecisionAwareForecaster
import foundation as F


def build_origins(cfg, wide_df, n):
    """从 test 段均匀抽 n 个起报点，返回 [(ctx_df_full, true_price), ...]。"""
    # 用 dataset 拿 valid_starts（复用其无泄漏划分逻辑），但 norm_stats 用 train 的
    train_ds = DecisionAwareDataset(wide_df, cfg, split="train", stride=cfg.train_stride)
    ds = DecisionAwareDataset(wide_df, cfg, split="test", norm_stats=train_ds.norm_stats,
                              stride=cfg.eval_stride)
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)
    origins = []
    for k in idxs:
        i = ds.valid_starts[k]
        ctx = wide_df.iloc[i: i + cfg.context_len]
        tp = wide_df[cfg.price_col].iloc[i + cfg.context_len: i + cfg.context_len + cfg.horizon_da].to_numpy(dtype=np.float32)
        origins.append((ctx, tp))
    return origins


def revenue_from_forecast(p_hat, true_price, sim, policy):
    """p_hat, true_price: [N,24] ndarray。返回 (R_model[N], R_star[N], regret[N])。
    BESS 模拟器无参数（纯张量运算），用 CPU 即可（N 小）。"""
    P = torch.from_numpy(p_hat).float()
    T = torch.from_numpy(true_price).float()
    u = policy(P)                       # [N,24] STE
    R = sim(u, T)                       # [N]
    Rs = oracle_revenue(T, sim)         # [N]
    return R.detach().cpu().numpy(), Rs.cpu().numpy(), (Rs - R).detach().cpu().numpy()


def run_my_model(cfg, origins, ckpt_tag):
    fc = DecisionAwareForecaster(cfg, ckpt_path=cfg.checkpoint_path(ckpt_tag),
                                 stats_path=cfg.checkpoint_path("stats"))
    preds = []
    for ctx, _ in origins:
        f = fc.predict(ctx, future_covariates=None, horizon=cfg.horizon_da)
        preds.append(np.asarray(f.mean, dtype=float).reshape(-1))
    return np.stack(preds)  # [N,24]


def run_foundations(cfg, origins, model_names):
    """每个 foundation: 零样本 predict_batch（price-only context），返回 {name: [N,24]}。"""
    price_only = [ctx[[cfg.price_col]] for ctx, _ in origins]
    out = {}
    FC = {"timesfm": F.TimesFMForecaster, "chronos2": F.Chronos2Forecaster,
          "toto": F.TotoForecaster, "toto2": F.Toto2Forecaster}
    for name in model_names:
        t0 = time.time()
        try:
            fc = FC[name]()
            fcs = fc.predict_batch(price_only, future_covs=None, horizon=cfg.horizon_da, multivariate=False)
            arr = np.stack([np.asarray(f.mean, dtype=float).reshape(-1) for f in fcs])
            out[name] = (arr, None)  # (preds, error)
            print(f"  [{name}] ✅ {arr.shape}  {time.time()-t0:.0f}s")
        except Exception as e:
            out[name] = (None, str(e)[:300])
            print(f"  [{name}] ❌ {str(e)[:200]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decision_aware/pilot_ercot.yaml")
    ap.add_argument("--n-origins", type=int, default=30)
    ap.add_argument("--ckpt", default="last", help="best 或 last（last=β=1）")
    ap.add_argument("--models", default="timesfm,chronos2,toto,toto2")
    ap.add_argument("--ckpt-dir", default=None, help="覆盖 yaml 的 checkpoint_dir（v1/v2 不同目录）")
    args = ap.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if args.ckpt_dir:
        cfg.checkpoint_dir = args.ckpt_dir
    print("="*70); print(f"B 对比: foundation 零样本 vs DA-TSFM({args.ckpt}) | {args.n_origins} 个 test 起报点 | 收益裁判台"); print("="*70)

    wide_df = loader.load_slice_model_ready(
        market=cfg.market, nodes=[cfg.node], freq=cfg.freq, covariates=ALL_COVARIATES,
        start="2025-01-01", end="2026-06-02", forecast=True)
    origins = build_origins(cfg, wide_df, args.n_origins)
    true_prices = np.stack([tp for _, tp in origins])  # [N,24]
    print(f"  数据: {cfg.node}, {len(origins)} 起报点, 真实电价 mean={true_prices.mean():.1f}")

    sim = BESSSimulator(cfg.bess_power_mw, cfg.bess_energy_mwh, cfg.bess_eta, cfg.bess_init_soc_frac)
    pol = STEPolicy(cfg.ste_k)

    # Oracle（所有模型共用同一上界）
    _, R_star, _ = revenue_from_forecast(true_prices, true_prices, sim, pol)
    print(f"  Oracle R* (均值, 开天眼基准): {R_star.mean():.1f}\n")

    results = []  # (name, R_mean, regret_mean, pct, mae, err)

    # "不充不放"基线：电池不动 R=0，看模型是否至少比不操作强
    results.append(("不充不放(R=0)", 0.0, R_star.mean(), 0.0, float('nan'), None))

    # 我的模型
    print(f"--- 你的模型 (β={'1' if args.ckpt=='last' else '0.5'}, ckpt={args.ckpt}) ---")
    p_mine = run_my_model(cfg, origins, args.ckpt)
    R, _, Reg = revenue_from_forecast(p_mine, true_prices, sim, pol)
    mae = np.mean(np.abs(p_mine - true_prices))
    results.append(("DA-TSFM-"+args.ckpt, R.mean(), Reg.mean(), R.mean()/R_star.mean()*100, mae, None))
    print(f"  ✅ R={R.mean():.1f} regret={Reg.mean():.1f} 占oracle={R.mean()/R_star.mean()*100:.0f}% MAE={mae:.1f}\n")

    # Foundations
    print(f"--- foundation 零样本 ---")
    found = run_foundations(cfg, origins, args.models.split(","))
    for name, (preds, err) in found.items():
        if preds is None:
            results.append((name, float('nan'), float('nan'), float('nan'), float('nan'), err))
            continue
        R, _, Reg = revenue_from_forecast(preds, true_prices, sim, pol)
        mae = np.mean(np.abs(preds - true_prices))
        results.append((name.upper()+"-zeroshot", R.mean(), Reg.mean(), R.mean()/R_star.mean()*100, mae, None))
        print(f"  → R={R.mean():.1f} regret={Reg.mean():.1f} 占oracle={R.mean()/R_star.mean()*100:.0f}% MAE={mae:.1f}")

    # 汇总表
    print("\n" + "="*70); print(f"{'模型':24s} {'R_model':>9s} {'regret':>9s} {'占oracle':>9s} {'MAE':>7s}")
    print("-"*70)
    for name, R, Reg, pct, mae, err in results:
        if err:
            print(f"{name:24s}  ❌ {err[:40]}")
        elif name.startswith("不充不放"):
            print(f"{name:24s} {0.0:9.1f} {Reg:9.1f} {0:8.0f}% {'—':>7s}")
        else:
            print(f"{name:24s} {R:9.1f} {Reg:9.1f} {pct:8.0f}% {mae:7.1f}")
    print("="*70)
    print("注: 所有模型过同一 BESS(1MW/4MWh/η0.9)+STE策略, oracle=LP开天眼(真上界)。")


if __name__ == "__main__":
    main()
