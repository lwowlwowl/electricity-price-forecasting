"""
evaluate.py — ElecFM 滚动回测评估
====================================
支持两种评估模式（对应 Lago 2021 checklist #1 #12）：

  [主结果] 连续 12 个月测试期（2025-07-01 ~ 2026-06-01）
    ─ run_evaluation() / run_evaluation_v6() 默认走这条路径
    ─ 对应 TEST_PERIOD = (TEST_START, TEST_END)
    ─ 输出主指标：rMAE + spike-F1(head)；辅助：MAE/SMAPE/pinball

  [子分析] W1/W2/W3 三窗口（保留，用于论文"不同市场状态"对比表）
    ─ run_evaluation(subwindows=True) 额外生成每窗口结果
    ─ 对应 TEST_WINDOWS 字典（2025-08、2025-03、2026-01 各一个月）

关键设计：
  1. 尖峰阈值 τ 仅用训练段（2025-01 ~ 2025-05-31）的数据计算
  2. τ* 在验证集（2025-06）上搜索
  3. 输出格式与现有 structural_full_* 目录兼容

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


# ── 时间划分常量（与 dataset.py 保持完全一致）────────────────────────────────
# 导入 dataset.py 的常量以确保单一数据源（不重复硬编码日期）
from dataset import (
    TRAIN_START, TRAIN_END,
    VAL_START,   VAL_END,
    TEST_START,  TEST_END,
)

# 主评估窗口：连续 12 个月（Lago checklist #1 #12）
TEST_PERIOD = (TEST_START, TEST_END)   # ("2025-07-01 00:00", "2026-06-01 23:00")

# 验证集窗口（τ* 搜索用）
VAL_WINDOW = (VAL_START, VAL_END)     # ("2025-06-01 00:00", "2025-06-30 23:00")

# 子分析窗口（保留 W1/W2/W3，用于"不同市场状态"子表）
TEST_WINDOWS = {
    "w1_stable":   ("2025-08-01", "2025-08-31"),  # 平稳夏季
    "w2_negative": ("2025-03-01", "2025-03-31"),  # 负价格频繁春季
    "w3_extreme":  ("2026-01-01", "2026-01-31"),  # 极端冬季
}


# ── 评估配置（与 v1.0/v2.0 基准一致）──────────────────────────────────────────
@dataclass
class EvalConfig:
    market:       str   = "ERCOT"
    nodes:        List[str] = None   # 由 YAML 注入
    freq:         str   = "1h"
    context_len:  int   = 168
    horizon:      int   = 24
    stride_hours: int   = 24          # 起报点间隔（24h = 逐日滚动）
    max_origins:  Optional[int] = None  # None = 跑满连续测试期（~335 个起报点）
    spike_quantile: float = 0.95
    tau_search_range: Tuple[float, float] = (0.05, 0.95)
    tau_search_step:  float = 0.05
    run_subwindows: bool = True       # 是否额外运行 W1/W2/W3 子窗口分析


def _compute_naive7_mae(data: pd.DataFrame, oi: int, horizon: int,
                        target_cols: list, floor: float = 1.0) -> float:
    """
    naive7（period=168）在起报点 oi 的 horizon 步预测 MAE，对所有节点取均值。
    naive7_pred[h] = p_{oi+h-168}（168 步前同时刻实际价格）。
    分母 floor 防止 CAISO 负价格区间 naive7_mae → 0 导致 rMAE 爆炸。
    返回 nan 若数据不足。
    """
    n = len(data)
    errors = []
    for h in range(horizon):
        actual_idx = oi + h
        naive_idx  = oi + h - 168
        if naive_idx < 0 or actual_idx >= n:
            continue
        for col in target_cols:
            v_a = float(data.iloc[actual_idx][col])
            v_n = float(data.iloc[naive_idx][col])
            if not (np.isnan(v_a) or np.isnan(v_n)):
                errors.append(abs(v_a - v_n))
    if not errors:
        return float("nan")
    raw = float(np.mean(errors))
    return max(raw, floor)          # floor 防爆


def _build_origins(index: pd.DatetimeIndex, test_start: str, test_end: str,
                   context_len: int, horizon: int, stride_hours: int,
                   max_origins: Optional[int]) -> List[int]:
    """在测试窗口内生成合法的起报点下标列表。

    test_start / test_end：日期字符串，可带或不带时间部分
    （兼容 "2025-07-01" 和 "2025-07-01 00:00" 两种格式）。
    """
    lo_ts = pd.Timestamp(test_start, tz="UTC")
    # test_end 取当天最后一刻：去掉可能已有的时间后缀，再接 23:00
    end_date = str(test_end).split()[0]   # 取日期部分（"2025-06-01"）
    hi_ts = pd.Timestamp(end_date + " 23:00", tz="UTC")

    lo = max(index.searchsorted(lo_ts), context_len)
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

    # ── 推理（含计时，Lago checklist #3）────────────────────────────────────
    import time as _time
    _t0 = _time.time()
    preds = _inference_batch(model, data, origins, target_cols,
                             cfg.context_len, cfg.horizon, device)
    _elapsed = _time.time() - _t0
    _per_day_s = _elapsed / max(len(origins), 1)
    print(f"  推理完成：{_elapsed:.1f}s 总计  "
          f"{_per_day_s:.2f}s/起报点（{len(origins)} 起报点）", flush=True)

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

        # naive7 MAE（rMAE 分母，Lago checklist #6）
        n7_mae = _compute_naive7_mae(data, oi, cfg.horizon, target_cols)

        m = M.all_point_prob_metrics(actual.ravel(), mean_p.ravel(),
                                     q10.ravel(), q50.ravel(), q90.ravel(),
                                     naive7_mae=n7_mae)
        per_origin_rows.append({
            "model": "ElecFM", "origin": data.index[oi],
            "naive7_mae": n7_mae,
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
        if metric in per_origin.columns:
            row[f"{metric}_mean"] = float(per_origin[metric].mean())
            row[f"{metric}_std"]  = float(per_origin[metric].std(ddof=0))
    row["mase_mean"] = float("nan")   # 无 SeasonalNaive，留空

    # rMAE（Lago checklist #6 主指标）— per-origin 均值，已逐起报点独立计算
    if "rmae" in per_origin.columns:
        valid = per_origin["rmae"].dropna()
        row["rmae_mean"] = float(valid.mean()) if len(valid) else float("nan")
        row["rmae_std"]  = float(valid.std(ddof=0)) if len(valid) > 1 else float("nan")

    # 计算成本（Lago checklist #3）
    row["inference_time_s"]         = round(_elapsed, 2)
    row["inference_time_per_day_s"] = round(_per_day_s, 3)

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

    rmae_str = f"rMAE={row['rmae_mean']:.4f}  " if "rmae_mean" in row else ""
    print(f"    {rmae_str}"
          f"MAE={row['mae_mean']:.2f}  SMAPE={row['smape_mean']:.2f}  "
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

    # 计算尖峰阈值：只用训练段（2025-01-01 ~ TRAIN_END）数据，与 dataset.py 一致
    train_mask = df.index <= pd.Timestamp(TRAIN_END, tz="UTC")
    thresholds = {}
    for node, col in zip(nodes, target_cols):
        vals = df.loc[train_mask, col].dropna().to_numpy(float)
        thresholds[node] = float(np.nanquantile(vals, cfg.spike_quantile))
    print(f"  尖峰阈值（P{cfg.spike_quantile*100:.0f}，训练数据口径）：{thresholds}")

    # τ* 搜索（验证集）
    print("\n搜索最优 spike head 阈值 τ*（验证集）...")
    tau_star = find_optimal_tau(model, df, target_cols, thresholds,
                                cfg.context_len, cfg.horizon, device, cfg)

    # ── 主评估：连续 12 个月测试期（Lago checklist #1 #12）──────────────────
    print(f"\n[主] 连续 12 个月评估 ({TEST_PERIOD[0]} ~ {TEST_PERIOD[1]})...")
    main_dir = os.path.join(output_root, "fusion_electfm_continuous")
    main_row = evaluate_window(model, df, "continuous",
                               TEST_PERIOD[0].split()[0],  # strip time portion
                               TEST_PERIOD[1].split()[0],
                               thresholds, tau_star, cfg, main_dir, device)
    main_row["window"] = "continuous"
    main_row.to_csv(os.path.join(main_dir, "summary.csv"), index=False)

    # ── 子分析：W1/W2/W3（论文"不同市场状态"子表，run_subwindows=True 时运行）──
    summaries = [main_row]
    if cfg.run_subwindows:
        for win_name, (start, end) in TEST_WINDOWS.items():
            print(f"\n[子] {win_name} ({start} ~ {end})...")
            out_dir = os.path.join(output_root, f"fusion_electfm_{win_name}")
            # 子窗口限制最多 30 个起报点（控速）
            sub_cfg = EvalConfig(**{**cfg.__dict__, "max_origins": 30})
            row = evaluate_window(model, df, win_name, start, end,
                                  thresholds, tau_star, sub_cfg, out_dir, device)
            row["window"] = win_name
            summaries.append(row)

    # 汇总（主窗口 + 子窗口）
    cross_window = pd.concat(summaries, ignore_index=True)
    cross_path = os.path.join(output_root, "fusion_electfm_all_windows.csv")
    cross_window.to_csv(cross_path, index=False)
    print(f"\n汇总（连续 + 子窗口）已保存：{cross_path}")


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

    # 计算尖峰阈值：只用训练段（~ TRAIN_END），与 dataset.py 一致
    train_mask = df.index <= pd.Timestamp(TRAIN_END, tz="UTC")
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

    # 评估：连续 12 个月（主）+ W1/W2/W3（子分析，可选）
    windows_to_eval = {"continuous": TEST_PERIOD}
    if cfg.run_subwindows:
        windows_to_eval.update(TEST_WINDOWS)

    summaries = []
    for win_name, (start, end) in windows_to_eval.items():
        _max_ori = None if win_name == "continuous" else 30
        print(f"\n评估 {win_name} ({start} ~ {end})...")
        origins = _build_origins(df.index, start, end, cfg.context_len,
                                 cfg.horizon, cfg.stride_hours, _max_ori)
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
    cross_path   = os.path.join(output_root, "fusion_electfm_all_windows.csv")
    cross_window.to_csv(cross_path, index=False)
    print(f"\n汇总（连续 + 子窗口）已保存：{cross_path}")
