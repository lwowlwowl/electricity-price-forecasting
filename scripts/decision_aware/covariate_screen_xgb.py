#!/usr/bin/env python
"""covariate_screen_xgb.py — 用 XGBoost 做协变量消融筛选（w10 §6.3 并行路径预实验）.

目的：在昂贵的正式版训练前，用 XGBoost（最擅长吃 exog 的模型）快速筛掉
没用的协变量，避免正式版白跑。

数据窗口：2025-01 → 2026-06（1.5 年，统一表与 model_ready 唯一重叠段）。
指标：MAE + regret / PCR（复用现有 BESSSimulator + LP oracle）。

用法：
  .venv_forecast/bin/python scripts/decision_aware/covariate_screen_xgb.py
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "decision_aware"))
os.chdir(_ROOT)

import xgboost as xgb
from scipy.optimize import linprog

# ── BESS 参数（w10 §7 规范值，与先行版 v3 一致）──────────────────────────
P_MAX = 1.0       # MW
E_MAX = 4.0       # MWh
ETA = 0.95        # 充放电效率
SOC_MIN = 0.4     # MWh
SOC_MAX = 3.6     # MWh
SOC_INIT = 0.5 * E_MAX  # 初始 SOC
KAPPA = 27.0      # 退化成本 USD/MWh
DT = 1.0          # 小时
HORIZON = 24      # 预测/结算窗口
CONTEXT = 168     # 7 天历史上下文（lag 特征）


def load_merged_data():
    """合并统一表 + model_ready 协变量，返回 2025-2026 重叠段 DataFrame。"""
    # 统一表
    df = pd.read_csv("data/raw/ERCOT/processed/ercot_unified_hourly_2020_2026.csv")
    df = df[df["节点/地区"] == "LZ_LCRA"].copy()
    df["ts"] = pd.to_datetime(df["UTC时间"], utc=True)
    df = df.set_index("ts").sort_index()
    out = pd.DataFrame({
        "price_da": df["日前价格 (USD/MWh)"].astype(float),
        "price_rt": df["实时价格 (USD/MWh)"].astype(float),
        "load": df["实际负荷 (MW)"].astype(float),
        "wind": df["风电实际 (MW)"].astype(float),
        "solar": df["光伏实际 (MW)"].astype(float),
    })
    idx = out.index
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    out["is_holiday"] = df["是否节假日"].astype(float).values

    # model_ready 协变量
    mr = pd.read_csv("data/covariates/model_ready/ercot_features_forecast_hourly.csv")
    mr["ts"] = pd.to_datetime(mr["timestamp_utc"], utc=True)
    mr = mr.set_index("ts").sort_index()
    # 只取统一表没有的列（避免重复）
    extra_cols = [c for c in mr.columns if c not in out.columns and c != "timestamp_utc"]
    out = out.join(mr[extra_cols], how="left")

    # 裁剪到重叠段
    out = out["2025-01-01":"2026-06-02"]
    out = out.dropna(subset=["price_da", "price_rt"])  # 目标必须有
    print(f"  合并后数据: {out.index.min()} → {out.index.max()}  {len(out)} 行")
    print(f"  列: {list(out.columns)}")
    return out


def build_samples(df: pd.DataFrame, feature_cols: list, target_col: str = "price_da"):
    """构建滑窗样本。每个样本 = 168h 历史 lag 特征 → 预测 24h。

    返回 X_train, y_train, X_val, y_val, X_test, y_test, 以及 test 起报点索引。

    特征：context 内每小时的目标价格 lag + 协变量当前值（统计聚合）+ 日历。
    简化：用 context 窗口的统计量（mean/std/min/max/lag1/lag24/lag168）+
          协变量在预测窗口的日前预报值（如果有）。
    """
    # 时间划分（连续，无重叠，符合 Lago 2021）
    train_end = pd.Timestamp("2025-12-31 23:00", tz="UTC")
    val_end = pd.Timestamp("2026-03-31 23:00", tz="UTC")
    # test: 2026-04-01 → 2026-06-01

    n = len(df)
    horizon = HORIZON
    context = CONTEXT

    # 起报点：每隔 24h 一个（eval_stride=24），取所有合法起点
    valid_starts = []
    for i in range(0, n - context - horizon + 1, 24):
        valid_starts.append(i)

    # 分类
    train_idx, val_idx, test_idx = [], [], []
    for i in valid_starts:
        tgt_start = df.index[i + context]
        if tgt_start <= train_end:
            train_idx.append(i)
        elif tgt_start <= val_end:
            val_idx.append(i)
        else:
            test_idx.append(i)

    def make_X(starts):
        """从起报点列表构建特征矩阵 + 目标。"""
        X_list, y_list, ts_list = [], [], []
        for i in starts:
            ctx_lo, ctx_hi = i, i + context
            tgt_lo, tgt_hi = ctx_hi, ctx_hi + horizon

            # 上下文窗口的目标价格历史
            price_hist = df[target_col].iloc[ctx_lo:ctx_hi].values

            # 目标窗口的协变量（日前预报值，起报时刻已知）
            tgt_window = df.iloc[tgt_lo:tgt_hi]

            features = {}
            # 价格 lag 统计
            features["price_lag1"] = price_hist[-1]
            features["price_lag24"] = price_hist[-24] if len(price_hist) >= 24 else price_hist[-1]
            features["price_lag168"] = price_hist[-1]  # 7天前同小时
            features["price_mean"] = price_hist.mean()
            features["price_std"] = price_hist.std()
            features["price_min"] = price_hist.min()
            features["price_max"] = price_hist.max()
            features["price_last6_mean"] = price_hist[-6:].mean()
            features["price_last24_mean"] = price_hist[-24:].mean()

            # 协变量：上下文统计 + 目标窗口预报值（如果列在 feature_cols 里）
            for col in feature_cols:
                if col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"]:
                    # 日历变量：用目标窗口第一小时的值
                    features[f"f_{col}"] = tgt_window[col].iloc[0]
                elif col in df.columns:
                    # 数值协变量：上下文统计 + 目标窗口均值（预报版可以直接用）
                    ctx_vals = df[col].iloc[ctx_lo:ctx_hi].values
                    tgt_vals = tgt_window[col].values
                    # 始终添加所有 key（用 nanmean 兜底，NaN 全部时用 0）
                    features[f"f_{col}_ctx_mean"] = float(np.nanmean(ctx_vals)) if not np.all(np.isnan(ctx_vals)) else 0.0
                    features[f"f_{col}_ctx_std"] = float(np.nanstd(ctx_vals)) if not np.all(np.isnan(ctx_vals)) else 0.0
                    features[f"f_{col}_tgt_mean"] = float(np.nanmean(tgt_vals)) if not np.all(np.isnan(tgt_vals)) else float(np.nanmean(ctx_vals)) if not np.all(np.isnan(ctx_vals)) else 0.0
                # 否则跳过（列不存在）

            X_list.append(features)
            y_list.append(df[target_col].iloc[tgt_lo:tgt_hi].values)
            ts_list.append(df.index[tgt_lo])

        return pd.DataFrame(X_list), np.array(y_list), ts_list

    X_tr, y_tr, _ = make_X(train_idx)
    X_va, y_va, _ = make_X(val_idx)
    X_te, y_te, ts_te = make_X(test_idx)

    # 确保列对齐（val/test 按 train 列顺序对齐，缺失列填 0）
    train_cols = X_tr.columns
    X_va = X_va.reindex(columns=train_cols, fill_value=0.0)
    X_te = X_te.reindex(columns=train_cols, fill_value=0.0)

    return X_tr, y_tr, X_va, y_va, X_te, y_te, ts_te


def train_xgb(X_tr, y_tr, X_va, y_va):
    """训练 24 个独立 XGBoost 回归器（逐时刻），带 early stopping。"""
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }

    models = []
    for h in range(y_tr.shape[1]):
        m = xgb.XGBRegressor(**params, early_stopping_rounds=20)
        m.fit(X_tr, y_tr[:, h], eval_set=[(X_va, y_va[:, h])], verbose=False)
        models.append(m)
    return models


def bess_revenue_lp_oracle(price: np.ndarray) -> float:
    """LP Oracle：用真实价格解线性规划求最优充放电 → R*（真上界）。

    复用 src/decision_aware/policy.py 的 lp_oracle_revenue 逻辑。
    price: [H] 真实 DA 价。
    """
    P, E, eta = P_MAX, E_MAX, ETA
    s0 = SOC_INIT
    s_min, s_max = SOC_MIN, SOC_MAX
    kappa = KAPPA
    dt = DT
    H = len(price)
    p = np.asarray(price, dtype=np.float64)

    # 决策变量 x = [dis_0..dis_{H-1}, chg_0..chg_{H-1}]
    c = np.concatenate([-p + kappa, p + kappa])

    A_ub, b_ub = [], []
    for t in range(H):
        row_dis = np.zeros(H)
        row_chg = np.zeros(H)
        row_dis[:t+1] = -1.0 / eta
        row_chg[:t+1] = eta
        row = np.concatenate([row_dis, row_chg])
        A_ub.append(row)
        b_ub.append(s_max - s0)
        A_ub.append(-row)
        b_ub.append(s0 - s_min)

    bounds = [(0, P * dt)] * (2 * H)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=bounds, method="highs")
    if res.success:
        x = res.x
        dis = x[:H]
        chg = x[H:]
        return float(np.sum((dis - chg) * p) - kappa * np.sum(dis + chg))
    else:
        # 退回 greedy
        thr = p.mean()
        u = np.sign(p - thr)
        # 简化 greedy 收益
        soc = s0
        rev = 0.0
        for t in range(H):
            ut = u[t]
            dis = max(ut, 0) * P * dt
            chg = max(-ut, 0) * P * dt
            d_act = min(dis, max(0, (soc - s_min) * eta))
            c_act = min(chg, max(0, (s_max - soc) / eta))
            rev += (d_act - c_act) * p[t] - kappa * (d_act + c_act)
            soc = soc - d_act / eta + c_act * eta
        return rev


def bess_revenue_model(pred: np.ndarray, actual: np.ndarray) -> float:
    """用预测价做 greedy 策略 → 按真实价结算 → R_model。

    pred: [H] 预测 DA 价；actual: [H] 真实 DA 价。
    策略：sign(pred - mean(pred)) greedy，SOC clip。
    """
    P, E, eta = P_MAX, E_MAX, ETA
    s0 = SOC_INIT
    s_min, s_max = SOC_MIN, SOC_MAX
    kappa = KAPPA
    dt = DT
    H = len(pred)
    p_pred = np.asarray(pred, dtype=np.float64)
    p_act = np.asarray(actual, dtype=np.float64)

    thr = p_pred.mean()
    u = np.sign(p_pred - thr)

    soc = s0
    rev = 0.0
    for t in range(H):
        ut = u[t]
        dis = max(ut, 0) * P * dt
        chg = max(-ut, 0) * P * dt
        d_act = min(dis, max(0, (soc - s_min) * eta))
        c_act = min(chg, max(0, (s_max - soc) / eta))
        rev += (d_act - c_act) * p_act[t] - kappa * (d_act + c_act)
        soc = soc - d_act / eta + c_act * eta
    return rev


def evaluate_arm(models, X_te, y_te):
    """评估一个臂：MAE + R_model + R_star(LP oracle) + regret + PCR。"""
    preds = np.column_stack([m.predict(X_te) for m in models])  # [n_samples, 24]

    n = len(y_te)
    all_mae = []
    all_r_model = []
    all_r_star = []

    for i in range(n):
        pred = preds[i]
        actual = y_te[i]
        # MAE
        all_mae.append(np.mean(np.abs(pred - actual)))
        # R_model（greedy 策略按预测，真实价结算）
        r_model = bess_revenue_model(pred, actual)
        all_r_model.append(r_model)
        # R_star（LP oracle，真实价）
        r_star = bess_revenue_lp_oracle(actual)
        all_r_star.append(r_star)

    mae = np.mean(all_mae)
    r_model_sum = np.sum(all_r_model)
    r_star_sum = np.sum(all_r_star)
    regret = r_star_sum - r_model_sum
    pcr = r_model_sum / r_star_sum * 100 if r_star_sum > 0 else float("nan")

    return {
        "MAE": float(mae),
        "R_model_sum": float(r_model_sum),
        "R_star_sum": float(r_star_sum),
        "regret": float(regret),
        "PCR": float(pcr),
        "n_samples": n,
    }


# ── 消融臂定义 ──────────────────────────────────────────────────────────────
ARMS = {
    "A0_baseline": {
        "desc": "price lag + calendar",
        "features": [],  # 日历自动加入
    },
    "A1_load": {
        "desc": "+load",
        "features": ["load"],
    },
    "A2_wind_solar": {
        "desc": "+wind +solar (统一表现有流)",
        "features": ["load", "wind", "solar"],
    },
    "A3_temp": {
        "desc": "+temperature",
        "features": ["load", "wind", "solar", "temperature"],
    },
    "A4_econ": {
        "desc": "+henry_hub +wti +natgas_storage",
        "features": ["load", "wind", "solar",
                      "henry_hub_usd_per_mmbtu", "wti_usd_per_barrel", "natgas_storage_bcf"],
    },
    "A5_storm": {
        "desc": "+storm_event_count",
        "features": ["load", "wind", "solar", "storm_event_count"],
    },
    "A6_genmix": {
        "desc": "+gas_share +renewable_share",
        "features": ["load", "wind", "solar", "gas_share", "renewable_share"],
    },
    "A7_all": {
        "desc": "全部协变量",
        "features": ["load", "wind", "solar", "temperature",
                      "henry_hub_usd_per_mmbtu", "wti_usd_per_barrel", "natgas_storage_bcf",
                      "storm_event_count", "gas_share", "renewable_share",
                      "renewable_shock", "gas_share_diff"],
    },
}

# 日历变量（所有臂都加）
CALENDAR_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"]


def main():
    t_total = time.time()
    print("=" * 70)
    print("协变量消融筛选 — XGBoost 路径（w10 §6.3 预实验）")
    print("=" * 70)

    # 加载合并数据
    df = load_merged_data()

    # 检查各协变量缺失情况
    print("\n协变量缺失统计:")
    for col in ["load", "wind", "solar", "temperature",
                "henry_hub_usd_per_mmbtu", "wti_usd_per_barrel", "natgas_storage_bcf",
                "storm_event_count", "gas_share", "renewable_share", "renewable_shock",
                "gas_share_diff"]:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            print(f"  {col}: {n_nan} NaN ({n_nan/len(df)*100:.1f}%)")
        else:
            print(f"  {col}: 列不存在")

    results = {}
    for arm_name, arm_cfg in ARMS.items():
        t_arm = time.time()
        print(f"\n{'─'*60}")
        print(f"训练臂: {arm_name} — {arm_cfg['desc']}")
        print(f"{'─'*60}")

        feature_cols = arm_cfg["features"] + CALENDAR_COLS

        # 构建样本
        X_tr, y_tr, X_va, y_va, X_te, y_te, ts_te = build_samples(df, feature_cols)
        print(f"  train={len(X_tr)} val={len(X_va)} test={len(X_te)}  特征数={X_tr.shape[1]}")

        # 训练
        t_train = time.time()
        model = train_xgb(X_tr, y_tr, X_va, y_va)
        print(f"  训练耗时: {time.time()-t_train:.1f}s")

        # 评估
        t_eval = time.time()
        metrics = evaluate_arm(model, X_te, y_te)
        print(f"  评估耗时: {time.time()-t_eval:.1f}s")
        print(f"  MAE={metrics['MAE']:.2f}  R_model={metrics['R_model_sum']:.1f}  "
              f"R*={metrics['R_star_sum']:.1f}  regret={metrics['regret']:.1f}  "
              f"PCR={metrics['PCR']:.1f}%")

        results[arm_name] = {**arm_cfg, **metrics}
        print(f"  臂总耗时: {time.time()-t_arm:.1f}s")

    # 汇总表
    print("\n" + "=" * 70)
    print("汇总：协变量消融筛选结果")
    print("=" * 70)
    print(f"{'臂':<20} {'描述':<40} {'MAE':>8} {'R_model':>10} {'R*':>10} {'regret':>10} {'PCR':>8}")
    print("-" * 106)
    for arm_name, r in results.items():
        print(f"{arm_name:<20} {r['desc']:<40} {r['MAE']:>8.2f} {r['R_model_sum']:>10.1f} "
              f"{r['R_star_sum']:>10.1f} {r['regret']:>10.1f} {r['PCR']:>7.1f}%")

    # Δ 表（相对 A0 基线）
    base = results["A0_baseline"]
    print(f"\n{'臂':<20} {'ΔMAE':>8} {'ΔR_model':>10} {'Δregret':>10} {'ΔPCR':>8} {'建议':>10}")
    print("-" * 76)
    for arm_name, r in results.items():
        if arm_name == "A0_baseline":
            continue
        d_mae = r["MAE"] - base["MAE"]
        d_r = r["R_model_sum"] - base["R_model_sum"]
        d_reg = r["regret"] - base["regret"]
        d_pcr = r["PCR"] - base["PCR"]
        # 决策建议
        if d_mae < -0.5 and d_reg < -10:
            advice = "✅ 推荐"
        elif d_reg < -10:
            advice = "✅ 推荐"
        elif d_mae < -0.5:
            advice = "🟡 可选"
        else:
            advice = "❌ 不放"
        print(f"{arm_name:<20} {d_mae:>+8.2f} {d_r:>+10.1f} {d_reg:>+10.1f} {d_pcr:>+7.1f}% {advice:>10}")

    # 保存结果
    out_path = "data/results/covariate_screen_xgb.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {out_path}")
    print(f"总耗时: {time.time()-t_total:.1f}s ({(time.time()-t_total)/60:.1f}min)")


if __name__ == "__main__":
    main()
