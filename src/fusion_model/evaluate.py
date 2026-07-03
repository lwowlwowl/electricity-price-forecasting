"""
evaluate.py — ElecFM 滚动回测评估
====================================
在 W1/W2/W3 三个测试窗口上运行滚动回测，输出与 v1.0/v2.0 完全兼容的
summary.csv / per_origin.csv / records.csv / thresholds.json。

关键设计：
  1. Spike head 推理阈值 τ* 在验证集上搜索（不用默认 0.5）
  2. 输出格式完全匹配现有 structural_full_* 目录下的 CSV 列名
  3. 不修改任何现有文件

运行环境：external/timesfm/.venv/bin/python
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ── 路径 ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
sys.path.insert(0, os.path.join(_ROOT, "src", "evaluation"))

import loader     # src/data_processing/loader.py
import metrics as M  # src/evaluation/metrics.py

from model import ElecFM, ElecFMV6


# ── 评估配置（与 v1.0/v2.0 基准一致）──────────────────────────────────────────
@dataclass
class EvalConfig:
    market:      str   = "ERCOT"
    nodes:       List[str] = None   # 由 YAML 注入
    freq:        str   = "1h"
    context_len: int   = 168
    horizon:     int   = 24
    stride_hours: int  = 24         # 起报点间隔
    max_origins: Optional[int] = 30 # 每窗口起报点数量（≥30 满足手册要求）
    spike_quantile: float = 0.95
    tau_search_range: Tuple[float, float] = (0.05, 0.95)
    tau_search_step:  float = 0.05


# 三个测试窗口（与 v1.0/v2.0 相同）
TEST_WINDOWS = {
    "w1_stable":   ("2025-08-01", "2025-08-31"),
    "w2_negative": ("2025-03-01", "2025-03-31"),
    "w3_extreme":  ("2026-01-01", "2026-01-31"),
}
VAL_WINDOW = ("2025-12-01", "2025-12-24")   # 与 dataset.py 训练验证集保持一致


def _build_origins(index: pd.DatetimeIndex, test_start: str, test_end: str,
                   context_len: int, horizon: int, stride_hours: int,
                   max_origins: Optional[int]) -> List[int]:
    """在测试窗口内生成合法的起报点下标列表。"""
    lo = max(index.searchsorted(pd.Timestamp(test_start, tz="UTC")), context_len)
    hi_ts = pd.Timestamp(test_end + " 23:59", tz="UTC")
    hi = min(index.searchsorted(hi_ts, side="right"), len(index) - horizon)

    origins = list(range(lo, hi + 1, stride_hours))
    if max_origins is not None:
        origins = origins[:max_origins]
    return origins


def _inference_batch(
    model: ElecFM,
    data: pd.DataFrame,
    origins: List[int],
    target_cols: List[str],
    context_len: int,
    horizon: int,
    device: torch.device,
) -> dict:
    """
    对所有起报点批量推理，返回每个 origin 的预测结果。

    Returns
    -------
    { origin_idx: { "mean": [H, N], "q10": ..., "q90": ..., "spike_prob": [H, N] } }
    """
    results = {}
    model.eval()

    for oi in origins:
        preds_per_node = {"mean": [], "q10": [], "q50": [], "q90": [], "spike_prob": []}

        for col in target_cols:
            ctx_vals = data[col].iloc[oi - context_len: oi].to_numpy(dtype=np.float32)
            ctx_t = torch.from_numpy(ctx_vals).unsqueeze(0).to(device)  # [1, context_len]

            with torch.no_grad():
                pred = model.predict(ctx_t)  # τ* 由 find_optimal_tau 搜索得到，在外部应用

            for k in preds_per_node:
                preds_per_node[k].append(pred[k][0])   # 去掉 batch 维

        # 多节点：按列堆叠 → [H, N]
        results[oi] = {k: np.stack(v, axis=-1) for k, v in preds_per_node.items()}

    return results


def _compute_spike_f1(y_true_flat, signal_flat, thresholds_flat):
    """使用逐元素阈值（各节点不同）计算 Spike-F1。"""
    true_spike = y_true_flat >= thresholds_flat
    pred_spike = signal_flat >= thresholds_flat
    tp = int(np.sum(pred_spike & true_spike))
    fp = int(np.sum(pred_spike & ~true_spike))
    fn = int(np.sum(~pred_spike & true_spike))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "spike_f1": f1}


def find_optimal_tau(
    model: ElecFM,
    data: pd.DataFrame,
    target_cols: List[str],
    thresholds: dict,
    context_len: int,
    horizon: int,
    device: torch.device,
    cfg: EvalConfig,
) -> float:
    """
    在验证集上搜索最优 spike head 推理阈值 τ*。
    目标：最大化验证集 Spike-F1（mean_signal 口径）。

    Returns: τ* ∈ [0.05, 0.95]
    """
    val_start, val_end = VAL_WINDOW
    origins = _build_origins(data.index, val_start, val_end,
                             context_len, horizon, cfg.stride_hours, cfg.max_origins)
    if not origins:
        print("  警告：验证集无合法起报点，使用默认阈值 τ*=0.5")
        return 0.5

    # 收集所有起报点的 spike_prob 和真值
    all_spike_prob, all_actual, all_thr = [], [], []
    nodes = [c[len("price__"):] for c in target_cols]

    for oi in origins:
        for j, (col, node) in enumerate(zip(target_cols, nodes)):
            ctx_vals = data[col].iloc[oi - context_len: oi].to_numpy(np.float32)
            ctx_t = torch.from_numpy(ctx_vals).unsqueeze(0).to(device)
            with torch.no_grad():
                _, spike_logits = model(ctx_t)
            prob = torch.sigmoid(spike_logits)[0].cpu().numpy()   # [horizon]
            actual = data[col].iloc[oi: oi + horizon].to_numpy(float)
            all_spike_prob.append(prob)
            all_actual.append(actual)
            all_thr.append(np.full(horizon, thresholds[node]))

    spike_prob = np.concatenate(all_spike_prob)
    actual_flat = np.concatenate(all_actual)
    thr_flat    = np.concatenate(all_thr)
    true_spike  = actual_flat >= thr_flat

    # 枚举阈值，找最优 F1
    tau_range = np.arange(cfg.tau_search_range[0],
                          cfg.tau_search_range[1] + cfg.tau_search_step / 2,
                          cfg.tau_search_step)
    best_f1, best_tau = -1.0, 0.5
    for tau in tau_range:
        pred_spike = spike_prob >= tau
        tp = np.sum(pred_spike & true_spike)
        fp = np.sum(pred_spike & ~true_spike)
        fn = np.sum(~pred_spike & true_spike)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)

    print(f"  τ* = {best_tau:.2f}（val Spike-F1 = {best_f1:.4f}）")
    return best_tau


def evaluate_window(
    model: ElecFM,
    data: pd.DataFrame,
    window_name: str,
    test_start: str,
    test_end: str,
    thresholds: dict,
    tau_star: float,
    cfg: EvalConfig,
    output_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    """
    在单个测试窗口上跑滚动回测，生成并保存结果文件。

    Returns
    -------
    summary_row : 一行汇总结果（DataFrame）
    """
    target_cols = [c for c in data.columns if c.startswith("price__")]
    nodes = [c[len("price__"):] for c in target_cols]
    N = len(nodes)

    origins = _build_origins(data.index, test_start, test_end,
                             cfg.context_len, cfg.horizon, cfg.stride_hours, cfg.max_origins)
    print(f"  {window_name}: {len(origins)} 起报点")

    # ── 推理 ─────────────────────────────────────────────────────────────────
    preds = _inference_batch(model, data, origins, target_cols,
                             cfg.context_len, cfg.horizon, device)

    # ── 收集 per_origin 和 records ────────────────────────────────────────────
    per_origin_rows = []
    record_rows = []

    for oi in origins:
        fut_idx = data.index[oi: oi + cfg.horizon]
        actual = data[target_cols].iloc[oi: oi + cfg.horizon].to_numpy(float)  # [H, N]
        mean_p = preds[oi]["mean"]   # [H, N]
        q10    = preds[oi]["q10"]
        q50    = preds[oi]["q50"]
        q90    = preds[oi]["q90"]

        m = M.all_point_prob_metrics(actual.ravel(), mean_p.ravel(),
                                     q10.ravel(), q50.ravel(), q90.ravel())
        per_origin_rows.append({
            "model": "ElecFM", "origin": data.index[oi],
            "covariates_used": False, "multivariate_used": False, **m})

        for j, node in enumerate(nodes):
            for h in range(cfg.horizon):
                record_rows.append({
                    "model": "ElecFM", "origin": data.index[oi],
                    "node": node, "ts": fut_idx[h],
                    "actual": actual[h, j],
                    "mean": mean_p[h, j], "q10": q10[h, j], "q90": q90[h, j],
                })

    per_origin = pd.DataFrame(per_origin_rows)
    records    = pd.DataFrame(record_rows)

    # ── 汇总指标 ─────────────────────────────────────────────────────────────
    row = {
        "model": "ElecFM",
        "covariates_used": False,
        "multivariate_used": False,
        "n_origins": int(per_origin["origin"].nunique()),
    }
    for metric in ("mae", "rmse", "smape", "pinball", "coverage"):
        if metric in per_origin:
            row[f"{metric}_mean"] = float(per_origin[metric].mean())
            row[f"{metric}_std"]  = float(per_origin[metric].std(ddof=0))
    row["mase_mean"] = float("nan")   # 无 SeasonalNaive，留空

    # Spike-F1（mean signal 和 q90 signal 两种口径）
    y_all, sig_mean_all, sig_q90_all, thr_all = [], [], [], []
    for oi in origins:
        for j, node in enumerate(nodes):
            actual_j = data[target_cols[j]].iloc[oi: oi + cfg.horizon].to_numpy(float)
            y_all.append(actual_j)
            sig_mean_all.append(preds[oi]["mean"][:, j])
            sig_q90_all.append(preds[oi]["q90"][:, j])
            thr_all.append(np.full(cfg.horizon, thresholds[node]))

    y_all       = np.concatenate(y_all)
    sig_mean_all = np.concatenate(sig_mean_all)
    sig_q90_all  = np.concatenate(sig_q90_all)
    thr_all      = np.concatenate(thr_all)

    sf_mean = _compute_spike_f1(y_all, sig_mean_all, thr_all)
    sf_q90  = _compute_spike_f1(y_all, sig_q90_all,  thr_all)
    row["spike_f1_mean_signal"] = sf_mean["spike_f1"]
    row["spike_precision"]       = sf_mean["precision"]
    row["spike_recall"]          = sf_mean["recall"]
    row["spike_f1_q90_signal"]   = sf_q90["spike_f1"]

    # Spike-F1（spike_prob signal，使用 τ*）
    # 直接复用 _inference_batch 已计算的 spike_prob，避免重复推理（~90 次 forward pass）
    spike_prob_all = []
    for oi in origins:
        for j in range(N):
            spike_prob_all.append(preds[oi]["spike_prob"][:, j])
    spike_prob_all = np.concatenate(spike_prob_all)
    pred_spike_prob = spike_prob_all >= tau_star
    true_spike_all  = y_all >= thr_all
    tp = np.sum(pred_spike_prob & true_spike_all)
    fp = np.sum(pred_spike_prob & ~true_spike_all)
    fn = np.sum(~pred_spike_prob & true_spike_all)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    row["spike_f1_spike_head"] = 2 * p * r / (p + r) if (p + r) else 0.0
    row["spike_precision_head"] = p
    row["spike_recall_head"]    = r

    summary_row = pd.DataFrame([row])

    # ── 保存文件 ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    summary_row.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    per_origin.to_csv(os.path.join(output_dir, "per_origin.csv"), index=False)
    records.to_csv(os.path.join(output_dir, "records.csv"), index=False)
    with open(os.path.join(output_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(output_dir, "tau_star.json"), "w") as f:
        json.dump({"tau_star": tau_star}, f)

    print(f"    SMAPE={row['smape_mean']:.2f}  "
          f"Pinball={row['pinball_mean']:.4f}  "
          f"SpikeF1(mean)={row['spike_f1_mean_signal']:.4f}  "
          f"SpikeF1(head,τ*={tau_star:.2f})={row['spike_f1_spike_head']:.4f}")
    print(f"    → {output_dir}")

    return summary_row


def run_evaluation(
    model: ElecFM,
    cfg: EvalConfig,
    checkpoint_path: str,
    output_root: str,
    device: torch.device,
):
    """
    主评估函数：τ* 搜索 → 三窗口评估 → 保存结果。

    结果保存到 output_root/fusion_electfm_<window>/
    格式与 data/results/structural_ablation/ 下的目录完全一致。
    """
    # 加载最优 checkpoint
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # 读取全量数据（实际数据从 2025-01-01 起）
    print("加载 ERCOT 数据...")
    df = loader.load_slice(
        market=cfg.market, nodes=cfg.nodes, freq=cfg.freq,
        start="2025-01-01", end="2026-06-05",
    )

    target_cols = [f"price__{n}" for n in cfg.nodes]
    nodes = cfg.nodes

    # 计算尖峰阈值 —— 与 dataset.py 保持一致：只用训练段数据（排除测试窗口和验证集）
    # 导入 dataset.py 的时间常量以确保一致性
    from dataset import EXCLUDED_RANGES as DS_EXCLUDED, VAL_START, VAL_END

    train_mask = pd.Series(True, index=df.index)
    for excl_s, excl_e in DS_EXCLUDED:
        es = pd.Timestamp(excl_s, tz="UTC")
        ee = pd.Timestamp(excl_e + " 23:59", tz="UTC")
        train_mask &= ~((df.index >= es) & (df.index <= ee))
    train_mask &= ~((df.index >= pd.Timestamp(VAL_START, tz="UTC")) &
                    (df.index <= pd.Timestamp(VAL_END + " 23:59", tz="UTC")))

    thresholds = {}
    for node, col in zip(nodes, target_cols):
        vals = df.loc[train_mask, col].dropna().to_numpy(float)
        thresholds[node] = float(np.nanquantile(vals, cfg.spike_quantile))
    print(f"  尖峰阈值（P{cfg.spike_quantile*100:.0f}，训练数据口径）：{thresholds}")

    # τ* 搜索（验证集）
    print("\n搜索最优 spike head 阈值 τ*（验证集）...")
    tau_star = find_optimal_tau(model, df, target_cols, thresholds,
                                cfg.context_len, cfg.horizon, device, cfg)

    # 三窗口评估
    summaries = []
    for win_name, (start, end) in TEST_WINDOWS.items():
        print(f"\n评估 {win_name} ({start} ~ {end})...")
        out_dir = os.path.join(output_root, f"fusion_electfm_{win_name}")
        row = evaluate_window(model, df, win_name, start, end,
                              thresholds, tau_star, cfg, out_dir, device)
        row["window"] = win_name
        summaries.append(row)

    # 跨窗口汇总
    cross_window = pd.concat(summaries, ignore_index=True)
    cross_path = os.path.join(output_root, "fusion_electfm_cross_window.csv")
    cross_window.to_csv(cross_path, index=False)
    print(f"\n跨窗口汇总已保存：{cross_path}")


# ── V6 评估：修改 _inference_batch 和 find_optimal_tau 以自动处理 ElecFMV6 ──────

def _inference_batch_v6_impl(
    model: "ElecFMV6",
    data: pd.DataFrame,
    origins: List[int],
    target_cols: List[str],
    context_len: int,
    horizon: int,
    device: torch.device,
) -> dict:
    """
    V6 专用推理：3 节点同时输入，返回与 _inference_batch 相同格式的结果。
    内部由 _inference_batch 调用（自动识别 ElecFMV6）。
    """
    results = {}
    model.eval()

    for oi in origins:
        ctx_list = [
            data[col].iloc[oi - context_len: oi].to_numpy(np.float32)
            for col in target_cols
        ]
        ctx_t = torch.from_numpy(np.stack(ctx_list)).unsqueeze(0).to(device)   # [1, 3, ctx]

        with torch.no_grad():
            q_pred, spike_logits = model(ctx_t)   # [1, 3, H, 9], [1, 3, H]

        preds_per_node = {"mean": [], "q10": [], "q50": [], "q90": [], "spike_prob": []}
        for n in range(len(target_cols)):
            q_n = q_pred[0, n]
            preds_per_node["mean"].append(q_n[:, 4].cpu().numpy())
            preds_per_node["q10"].append(q_n[:, 0].cpu().numpy())
            preds_per_node["q50"].append(q_n[:, 4].cpu().numpy())
            preds_per_node["q90"].append(q_n[:, 8].cpu().numpy())
            preds_per_node["spike_prob"].append(
                torch.sigmoid(spike_logits[0, n]).cpu().numpy()
            )

        results[oi] = {k: np.stack(v, axis=-1) for k, v in preds_per_node.items()}

    return results


def run_evaluation_v6(
    model: "ElecFMV6",
    cfg: EvalConfig,
    checkpoint_path: str,
    output_root: str,
    device: torch.device,
):
    """
    V6 主评估函数：复用 run_evaluation，但用 V6 专用推理替换 _inference_batch。
    """
    from model import ElecFMV6 as _ElecFMV6

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # 使用 V6 节点（与 cfg.nodes 一致，应为 3 个波动节点）
    print("加载 ERCOT 数据...")
    df = loader.load_slice(
        market=cfg.market, nodes=cfg.nodes, freq=cfg.freq,
        start="2025-01-01", end="2026-06-05",
    )
    target_cols = [f"price__{n}" for n in cfg.nodes]

    from dataset import EXCLUDED_RANGES as DS_EXCLUDED, VAL_START, VAL_END
    train_mask = pd.Series(True, index=df.index)
    for excl_s, excl_e in DS_EXCLUDED:
        train_mask &= ~((df.index >= pd.Timestamp(excl_s, tz="UTC")) &
                        (df.index <= pd.Timestamp(excl_e + " 23:59", tz="UTC")))
    train_mask &= ~((df.index >= pd.Timestamp(VAL_START, tz="UTC")) &
                    (df.index <= pd.Timestamp(VAL_END + " 23:59", tz="UTC")))

    thresholds = {
        node: float(np.nanquantile(
            df.loc[train_mask, f"price__{node}"].dropna().values, cfg.spike_quantile
        ))
        for node in cfg.nodes
    }
    print(f"  尖峰阈值（P{cfg.spike_quantile*100:.0f}，训练数据口径）：{thresholds}")

    # τ* 搜索（3 节点同时）
    print("\n搜索最优 spike head 阈值 τ*（验证集）...")
    val_start, val_end = VAL_WINDOW
    val_origins = _build_origins(df.index, val_start, val_end,
                                 cfg.context_len, cfg.horizon,
                                 cfg.stride_hours, cfg.max_origins)
    if not val_origins:
        tau_star = 0.5
        print(f"  警告：验证集无起报点，使用默认 τ*=0.5")
    else:
        all_prob, all_actual, all_thr = [], [], []
        for oi in val_origins:
            ctx_list = [df[c].iloc[oi - cfg.context_len: oi].to_numpy(np.float32)
                        for c in target_cols]
            ctx_t = torch.from_numpy(np.stack(ctx_list)).unsqueeze(0).to(device)
            with torch.no_grad():
                _, sl = model(ctx_t)
            for n, node in enumerate(cfg.nodes):
                all_prob.append(torch.sigmoid(sl[0, n]).cpu().numpy())
                all_actual.append(df[target_cols[n]].iloc[oi: oi + cfg.horizon].to_numpy(float))
                all_thr.append(np.full(cfg.horizon, thresholds[node]))

        prob_flat   = np.concatenate(all_prob)
        actual_flat = np.concatenate(all_actual)
        thr_flat    = np.concatenate(all_thr)
        true_spike  = actual_flat >= thr_flat

        tau_range = np.arange(cfg.tau_search_range[0],
                              cfg.tau_search_range[1] + cfg.tau_search_step / 2,
                              cfg.tau_search_step)
        best_f1, tau_star = -1.0, 0.5
        for tau in tau_range:
            pred = prob_flat >= tau
            tp = np.sum(pred & true_spike)
            fp = np.sum(pred & ~true_spike)
            fn = np.sum(~pred & true_spike)
            p  = tp / (tp + fp) if (tp + fp) else 0.0
            r  = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            if f1 > best_f1:
                best_f1, tau_star = f1, float(tau)
        print(f"  τ* = {tau_star:.2f}（val Spike-F1 = {best_f1:.4f}）")

    # 三窗口评估
    summaries = []
    for win_name, (start, end) in TEST_WINDOWS.items():
        print(f"\n评估 {win_name} ({start} ~ {end})...")
        origins = _build_origins(df.index, start, end, cfg.context_len,
                                 cfg.horizon, cfg.stride_hours, cfg.max_origins)
        print(f"  {win_name}: {len(origins)} 起报点")

        preds = _inference_batch_v6_impl(model, df, origins, target_cols,
                                         cfg.context_len, cfg.horizon, device)

        out_dir = os.path.join(output_root, f"fusion_electfm_{win_name}")
        os.makedirs(out_dir, exist_ok=True)

        # 收集指标（复用 evaluate_window 的统计逻辑）
        per_origin_rows, record_rows = [], []
        for oi in origins:
            fut_idx = df.index[oi: oi + cfg.horizon]
            actual  = df[target_cols].iloc[oi: oi + cfg.horizon].to_numpy(float)
            mean_p  = preds[oi]["mean"]; q10 = preds[oi]["q10"]
            q50     = preds[oi]["q50"];  q90 = preds[oi]["q90"]
            m = M.all_point_prob_metrics(actual.ravel(), mean_p.ravel(),
                                         q10.ravel(), q50.ravel(), q90.ravel())
            per_origin_rows.append({"model": "ElecFMV6", "origin": df.index[oi], **m})
            for j, node in enumerate(cfg.nodes):
                for h in range(cfg.horizon):
                    record_rows.append({
                        "model": "ElecFMV6", "origin": df.index[oi],
                        "node": node, "ts": fut_idx[h],
                        "actual": actual[h, j], "mean": mean_p[h, j],
                        "q10": q10[h, j], "q90": q90[h, j],
                    })

        per_origin = pd.DataFrame(per_origin_rows)
        records    = pd.DataFrame(record_rows)
        per_origin.to_csv(os.path.join(out_dir, "per_origin.csv"), index=False)
        records.to_csv(os.path.join(out_dir, "records.csv"), index=False)
        with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
            json.dump(thresholds, f, indent=2)
        with open(os.path.join(out_dir, "tau_star.json"), "w") as f:
            json.dump({"tau_star": tau_star}, f)

        # 汇总行
        row = {"model": "ElecFMV6", "n_origins": int(per_origin["origin"].nunique())}
        for metric in ("mae", "rmse", "smape", "pinball", "coverage"):
            if metric in per_origin:
                row[f"{metric}_mean"] = float(per_origin[metric].mean())
                row[f"{metric}_std"]  = float(per_origin[metric].std(ddof=0))
        row["mase_mean"] = float("nan")

        y_all, sig_mean_all, sig_q90_all, thr_all = [], [], [], []
        for oi in origins:
            for j, node in enumerate(cfg.nodes):
                y_all.append(df[target_cols[j]].iloc[oi: oi + cfg.horizon].to_numpy(float))
                sig_mean_all.append(preds[oi]["mean"][:, j])
                sig_q90_all.append(preds[oi]["q90"][:, j])
                thr_all.append(np.full(cfg.horizon, thresholds[node]))
        y_all = np.concatenate(y_all); sig_mean_all = np.concatenate(sig_mean_all)
        sig_q90_all = np.concatenate(sig_q90_all); thr_all = np.concatenate(thr_all)

        sf_mean = _compute_spike_f1(y_all, sig_mean_all, thr_all)
        sf_q90  = _compute_spike_f1(y_all, sig_q90_all,  thr_all)
        row["spike_f1_mean_signal"] = sf_mean["spike_f1"]
        row["spike_f1_q90_signal"]  = sf_q90["spike_f1"]

        spike_prob_all = np.concatenate([preds[oi]["spike_prob"][:, j]
                                         for oi in origins for j in range(len(cfg.nodes))])
        true_spike_all = y_all >= thr_all
        pred_spike_prob = spike_prob_all >= tau_star
        tp = np.sum(pred_spike_prob & true_spike_all)
        fp = np.sum(pred_spike_prob & ~true_spike_all)
        fn = np.sum(~pred_spike_prob & true_spike_all)
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        row["spike_f1_spike_head"] = 2 * p * r / (p + r) if (p + r) else 0.0

        summary_row = pd.DataFrame([row])
        summary_row.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

        print(f"    SMAPE={row.get('smape_mean', float('nan')):.2f}  "
              f"Pinball={row.get('pinball_mean', float('nan')):.4f}  "
              f"SpikeF1(mean)={row['spike_f1_mean_signal']:.4f}  "
              f"SpikeF1(head,τ*={tau_star:.2f})={row['spike_f1_spike_head']:.4f}")
        print(f"    → {out_dir}")

        row_df = pd.DataFrame([row])
        row_df["window"] = win_name
        summaries.append(row_df)

    cross_window = pd.concat(summaries, ignore_index=True)
    cross_path   = os.path.join(output_root, "fusion_electfm_cross_window.csv")
    cross_window.to_csv(cross_path, index=False)
    print(f"\n跨窗口汇总已保存：{cross_path}")
