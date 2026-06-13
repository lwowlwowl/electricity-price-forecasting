"""
评估指标模块 metrics.py
========================
实现手册 §5 的三类指标：点误差、概率误差、尖峰预警(Spike-F1)。

所有函数都接受 numpy 数组，按"逐时刻"展平后计算，便于在滚动回测里
把多个起报点 × 多个预测时刻的记录汇总成一个总指标。

★ 重点：Spike-F1
  把电价预测转成"是否尖峰"的二分类问题，评估模型的尖峰预警能力。
  关键防泄露约束（手册 §5.3）：尖峰阈值 P95 只能用起报点之前的历史算，
  绝不能用包含测试期/未来的全量数据算。本模块只负责"给定阈值算 F1"，
  阈值的计算时机由 backtest.py 控制，从流程上杜绝泄露。
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np


# ── 1. 点预测指标 ─────────────────────────────────────────────────────────────
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差。"""
    return float(np.nanmean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差，对大误差（尖峰失误）更敏感。"""
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    对称平均绝对百分比误差（%）。
    注意：电价可能为负或接近零，分母用 |真值|+|预测| 并加 eps 防爆。
    """
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.nanmean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def mase(y_true: np.ndarray, y_pred: np.ndarray,
         naive_mae: float, eps: float = 1e-9) -> float:
    """
    平均绝对标度误差 = MAE / 季节性Naive的MAE。
    跨节点/跨市场可比（消除量纲）。naive_mae 由回测引擎传入
    （同一配置下季节性朴素基线的 MAE）。<1 表示比朴素好。
    """
    return float(mae(y_true, y_pred) / (naive_mae + eps))


# ── 2. 概率预测指标 ───────────────────────────────────────────────────────────
def pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, q: float) -> float:
    """
    单个分位数 q 的 pinball（分位数）损失。
    q=0.5 时退化为 0.5*MAE。越小越好。
    """
    diff = y_true - y_quantile
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.nanmean(loss))


def avg_pinball(y_true, q10, q50, q90) -> float:
    """q10/q50/q90 三个分位数的平均 pinball loss。"""
    parts = [
        pinball_loss(y_true, q10, 0.10),
        pinball_loss(y_true, q50, 0.50),
        pinball_loss(y_true, q90, 0.90),
    ]
    return float(np.mean(parts))


def coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    """
    区间覆盖率：真值落在 [q_low, q_high] 的比例。
    对 [q10, q90] 理想值约 0.80。返回 0~1。
    """
    inside = (y_true >= q_low) & (y_true <= q_high)
    return float(np.nanmean(inside.astype(float)))


# ── 3. ★ Spike-F1 尖峰预警 ────────────────────────────────────────────────────
def compute_spike_threshold(history_values: np.ndarray, quantile: float = 0.95) -> float:
    """
    用【历史】电价计算尖峰阈值（默认 P95）。
    调用方必须保证 history_values 只包含起报点之前的数据，避免泄露。
    """
    return float(np.nanquantile(history_values, quantile))


def spike_f1(
    y_true: np.ndarray,
    y_signal: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    给定尖峰阈值，计算尖峰二分类的 Precision/Recall/F1。

    参数
    ----
    y_true   : 实际电价（展平后的一维数组）
    y_signal : 用于判定"预测是否尖峰"的信号——可传 pred_mean 或 pred_q90
    threshold: 尖峰阈值（由 compute_spike_threshold 用历史算出）

    返回
    ----
    dict: precision / recall / spike_f1 / tp / fp / fn / n_spike(真实尖峰数)
    """
    yt = np.asarray(y_true, dtype=float)
    ys = np.asarray(y_signal, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(ys))
    yt, ys = yt[mask], ys[mask]

    true_spike = yt >= threshold
    pred_spike = ys >= threshold

    tp = int(np.sum(pred_spike & true_spike))
    fp = int(np.sum(pred_spike & ~true_spike))
    fn = int(np.sum(~pred_spike & true_spike))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "spike_f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_spike": int(np.sum(true_spike)),
    }


# ── 4. 汇总：一把算齐所有点/概率指标 ──────────────────────────────────────────
def all_point_prob_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    q10: Optional[np.ndarray] = None,
    q50: Optional[np.ndarray] = None,
    q90: Optional[np.ndarray] = None,
    naive_mae: Optional[float] = None,
) -> Dict[str, float]:
    """
    一次性算齐点误差 + 概率误差（不含 Spike-F1，后者需阈值另算）。
    分位数缺失时自动跳过概率项。
    """
    out = {
        "mae": mae(y_true, mean),
        "rmse": rmse(y_true, mean),
        "smape": smape(y_true, mean),
    }
    if naive_mae is not None:
        out["mase"] = mase(y_true, mean, naive_mae)
    if q10 is not None and q90 is not None:
        q50_eff = q50 if q50 is not None else mean
        out["pinball"] = avg_pinball(y_true, q10, q50_eff, q90)
        out["coverage"] = coverage(y_true, q10, q90)
    return out


# ── 自测 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.normal(30, 10, 500)
    y[::50] += 200                          # 人为塞一些尖峰
    pred = y + rng.normal(0, 5, 500)        # 预测=真值+噪声

    thr = compute_spike_threshold(y[:400], 0.95)   # 仅用前 400 个算阈值
    print("=" * 60)
    print("metrics 自测")
    print("=" * 60)
    m = all_point_prob_metrics(y, pred, pred - 8, pred, pred + 8, naive_mae=9.0)
    for k, v in m.items():
        print(f"  {k:10s}= {v:.4f}")
    sf = spike_f1(y, pred, thr)
    print(f"  spike阈值  = {thr:.2f}")
    print(f"  Spike-F1   = P={sf['precision']:.3f} R={sf['recall']:.3f} "
          f"F1={sf['spike_f1']:.3f} (TP={sf['tp']} FP={sf['fp']} FN={sf['fn']})")
    print("\n✅ metrics 工作正常")
