#!/usr/bin/env python
"""compare_baselines_formal.py — 正式版 B 对比：foundation 零样本 vs DA-TSFM v4，同台比收益。

与先行版 compare_baselines.py 的区别：
- 用 HardTopKPolicy（非 STE）
- 用 DA 价（非 RT 价）
- κ=27 + SOC 0.4-3.6 + η=0.95（w10 规范，非简化版）
- 用 v3 数据集（6.5年统一表，非 17 月 model_ready）
- 加"不充不放"基线（R=0），看模型是否至少比不操作好

每个模型在相同 test 起报点上：
  预测 24h DA 价 → 喂进同一套 BESS(HardTopK+κ=27) → 算 R_model、regret。
不比 MAE 比 R——这是 decision-aware 的核心卖点。

用法:
  external/chronos-forecasting/.venv/bin/python scripts/decision_aware/compare_baselines_formal.py \
      --config configs/decision_aware/formal_ercot_v4.yaml [--n-origins 30] [--models timesfm,chronos2]
"""
from __future__ import annotations
import argparse, os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
sys.path.insert(0, os.path.join(_ROOT, "src", "models"))
os.chdir(_ROOT)

import numpy as np, pandas as pd, torch
from decision_aware.config import PilotConfig
from decision_aware.dataset_v3 import build_datasets_v3, collate_v3
from decision_aware.model import DecisionAwareTSFM
from decision_aware.policy import BESSSimulator, HardTopKPolicy, lp_oracle_revenue, lp_oracle_revenue_dual
from decision_aware.loader_v2 import load_ercot_unified
from torch.utils.data import DataLoader


def build_origins_v3(cfg, n):
    """从 v3 test 段均匀抽 n 个起报点，返回 [(ctx_df, true_da_price), ...]。"""
    wide_df = load_ercot_unified(node=cfg.node, start="2020-01-01", end="2026-06-02")
    train_ds, _, test_ds, _ = build_datasets_v3(cfg)
    idxs = np.linspace(0, len(test_ds) - 1, min(n, len(test_ds))).astype(int)

    origins = []
    for k in idxs:
        i = test_ds.valid_starts[k]
        ctx = wide_df.iloc[i: i + cfg.context_len]
        tp = wide_df["price_da"].iloc[i + cfg.context_len: i + cfg.context_len + cfg.horizon_da].to_numpy(dtype=np.float32)
        origins.append((ctx, tp))
    return origins, train_ds.norm_stats


def revenue_from_forecast(p_hat, true_price, sim, pol_hard):
    """p_hat, true_price: [N,24] ndarray。返回 (R_model[N], R_star_LP[N], regret[N])。"""
    P = torch.from_numpy(p_hat).float()
    T = torch.from_numpy(true_price).float()
    with torch.no_grad():
        u = pol_hard(P)
        R = sim(u, T)
        R_star = lp_oracle_revenue(T, sim)
    return R.numpy(), R_star.numpy(), (R_star - R).numpy()


def run_my_model(cfg, origins, norm_stats, ckpt_tag):
    """用 v4 训练好的模型预测。"""
    model = DecisionAwareTSFM(cfg)
    ckpt_path = cfg.checkpoint_path(ckpt_tag)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(dev)

    preds = []
    for ctx, _ in origins:
        # 构建 v3 batch（单样本）
        if ctx.index.tz is None:
            ctx.index = ctx.index.tz_localize("UTC")
        else:
            ctx.index = ctx.index.tz_convert("UTC")
        ctx = ctx.sort_index().iloc[-cfg.context_len:]

        da_st = norm_stats["price_da"]
        rt_st = norm_stats["price_rt"]
        price_da = ctx["price_da"].to_numpy(dtype=np.float32)
        price_rt = ctx["price_rt"].to_numpy(dtype=np.float32)

        batch = {
            "price_da_ctx": torch.from_numpy((price_da - da_st["mean"]) / da_st["std"]).unsqueeze(0).to(dev),
            "price_rt_ctx": torch.from_numpy((price_rt - rt_st["mean"]) / rt_st["std"]).unsqueeze(0).to(dev),
            "price_da_mean": torch.tensor([da_st["mean"]]).to(dev),
            "price_da_std": torch.tensor([da_st["std"]]).to(dev),
            "price_rt_mean": torch.tensor([rt_st["mean"]]).to(dev),
            "price_rt_std": torch.tensor([rt_st["std"]]).to(dev),
        }
        # load/system/calendar 流
        for stream, cols in [("load", ["load"]), ("system", ["wind", "solar"])]:
            st = norm_stats[stream]
            arr = ctx[cols].to_numpy(dtype=np.float32)
            arr_n = (arr - st["mean"]) / st["std"]
            batch[f"{stream}_ctx"] = torch.from_numpy(arr_n).unsqueeze(0).to(dev)
        # calendar
        idx = ctx.index
        cal = np.stack([
            np.sin(2*np.pi*idx.hour/24), np.cos(2*np.pi*idx.hour/24),
            np.sin(2*np.pi*idx.dayofweek/7), np.cos(2*np.pi*idx.dayofweek/7),
            (idx.dayofweek >= 5).astype(float),
            ctx["is_holiday"].astype(float).values if "is_holiday" in ctx else np.zeros(len(ctx)),
        ], axis=-1).astype(np.float32)
        batch["cal_ctx"] = torch.from_numpy(cal).unsqueeze(0).to(dev)

        with torch.no_grad():
            out = model(batch)
        preds.append(out["p_da"][0].cpu().numpy())

    return np.stack(preds)  # [N,24]


def run_foundations(cfg, origins, model_names):
    """每个 foundation: 零样本 predict_batch（price-only context），返回 {name: [N,24]}。"""
    import foundation as F
    # foundation 需要 price__<node> 列
    price_col = f"price__{cfg.node}"
    price_only = []
    for ctx, _ in origins:
        c = ctx.copy()
        c[price_col] = c["price_da"]  # 用 DA 价作为历史价格
        price_only.append(c[[price_col]])

    out = {}
    FC = {"timesfm": F.TimesFMForecaster, "chronos2": F.Chronos2Forecaster,
          "toto": F.TotoForecaster, "toto2": F.Toto2Forecaster}
    for name in model_names:
        t0 = time.time()
        try:
            fc = FC[name]()
            fcs = fc.predict_batch(price_only, future_covs=None, horizon=cfg.horizon_da, multivariate=False)
            arr = np.stack([np.asarray(f.mean, dtype=float).reshape(-1) for f in fcs])
            out[name] = (arr, None)
            print(f"  [{name}] ✅ {arr.shape}  {time.time()-t0:.0f}s")
        except Exception as e:
            out[name] = (None, str(e)[:300])
            print(f"  [{name}] ❌ {str(e)[:200]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decision_aware/formal_ercot_v4.yaml")
    ap.add_argument("--n-origins", type=int, default=30)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--models", default="timesfm,chronos2")
    ap.add_argument("--ckpt-dir", default=None, help="覆盖 yaml 的 checkpoint_dir")
    args = ap.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if args.ckpt_dir:
        cfg.checkpoint_dir = args.ckpt_dir
    print("=" * 70)
    print(f"B 对比（正式版）: foundation 零样本 vs DA-TSFM v4({args.ckpt})")
    print(f"  {args.n_origins} 个 test 起报点 | HardTopK + κ=27 + DA 价")
    print("=" * 70)

    origins, norm_stats = build_origins_v3(cfg, args.n_origins)
    true_prices = np.stack([tp for _, tp in origins])  # [N,24]
    print(f"  数据: {cfg.node}, {len(origins)} 起报点, DA 价 mean={true_prices.mean():.1f} std={true_prices.std():.1f}")
    print(f"  BESS: P={cfg.bess_power_mw}MW E={cfg.bess_energy_mwh}MWh η={cfg.bess_eta} κ={cfg.bess_kappa} SOC[{cfg.bess_soc_min},{cfg.bess_soc_max}]")
    print()

    sim = BESSSimulator(cfg.bess_power_mw, cfg.bess_energy_mwh, cfg.bess_eta,
                        cfg.bess_init_soc_frac, kappa=cfg.bess_kappa,
                        soc_min=cfg.bess_soc_min, soc_max=cfg.bess_soc_max,
                        e_cyc=cfg.bess_e_cyc)
    pol = HardTopKPolicy(cfg.topk_k_charge, cfg.topk_k_discharge,
                         spread_threshold=cfg.resolved_spread_threshold)

    # Oracle（所有模型共用同一上界）
    _, R_star, _ = revenue_from_forecast(true_prices, true_prices, sim, pol)
    print(f"  LP Oracle R* (均值, 开天眼最优): {R_star.mean():.1f}")
    print(f"  '不充不放' R = 0（基准）")
    print()

    results = []

    # 不充不放基线
    results.append(("不充不放(R=0)", 0.0, R_star.mean(), R_star.mean(), 0.0, None))

    # 我的模型
    print(f"--- DA-TSFM v4 (ckpt={args.ckpt}) ---")
    t0 = time.time()
    try:
        p_mine = run_my_model(cfg, origins, norm_stats, args.ckpt)
        R, Rs, Reg = revenue_from_forecast(p_mine, true_prices, sim, pol)
        mae = np.mean(np.abs(p_mine - true_prices))
        results.append((f"DA-TSFM-v4-{args.ckpt}", R.mean(), Rs.mean(), Reg.mean(), mae, None))
        print(f"  ✅ R={R.mean():.1f} regret={Reg.mean():.1f} 占LP={R.mean()/Rs.mean()*100:.0f}% MAE={mae:.1f}  ({time.time()-t0:.0f}s)")
    except Exception as e:
        results.append((f"DA-TSFM-v4-{args.ckpt}", float('nan'), float('nan'), float('nan'), float('nan'), str(e)[:200]))
        print(f"  ❌ {str(e)[:200]}")
    print()

    # Foundations
    print(f"--- foundation 零样本 ---")
    found = run_foundations(cfg, origins, args.models.split(","))
    for name, (preds, err) in found.items():
        if preds is None:
            results.append((name.upper()+"-zeroshot", float('nan'), float('nan'), float('nan'), float('nan'), err))
            continue
        R, Rs, Reg = revenue_from_forecast(preds, true_prices, sim, pol)
        mae = np.mean(np.abs(preds - true_prices))
        results.append((name.upper()+"-zeroshot", R.mean(), Rs.mean(), Reg.mean(), mae, None))
        print(f"  → R={R.mean():.1f} regret={Reg.mean():.1f} 占LP={R.mean()/Rs.mean()*100:.0f}% MAE={mae:.1f}")
    print()

    # 汇总表
    print("=" * 80)
    print(f"{'模型':26s} {'R_model':>9s} {'R*(LP)':>9s} {'regret':>9s} {'占LP':>7s} {'MAE':>7s}")
    print("-" * 80)
    for name, R, Rs, Reg, mae, err in results:
        if err:
            print(f"{name:26s}  ❌ {err[:40]}")
        elif name.startswith("不充不放"):
            print(f"{name:26s} {'0.0':>9s} {Rs:9.1f} {Rs:9.1f} {'0%':>7s} {'—':>7s}")
        else:
            pct = R / Rs * 100 if Rs != 0 else float('nan')
            print(f"{name:26s} {R:9.1f} {Rs:9.1f} {Reg:9.1f} {pct:6.0f}% {mae:7.1f}")
    print("=" * 80)
    print(f"注: 所有模型过同一 BESS(1MW/4MWh/η0.95/κ27/SOC0.4-3.6)+HardTopK(K=4), oracle=LP开天眼。")
    print(f"    DA 价 std={true_prices.std():.1f}, 尖峰={true_prices.max():.0f}")


if __name__ == "__main__":
    main()
