"""
滚动回测引擎 backtest.py
=========================
实现手册 §4 的多起报点 walk-forward 评估。这是整个框架的心脏：

  给定测试期 [test_start, test_end]，按 STRIDE 设一串起报点 t：
    context = 数据[t - CONTEXT_LEN : t]   ← 只给历史，从结构上杜绝泄露
    actual  = 数据[t : t + HORIZON]       ← 真值，仅用于评估
    对每个模型：pred = model.predict(context, future_cov, horizon)
    记录 (起报点, 模型, 节点, 时刻, pred_mean, pred_q90, actual)
  汇总所有起报点 → 每个(模型)给出指标的【均值±标准差】，而非单个数字。

三条公平性原则在这里落地：
  1. 同起跑线：context 严格截断在起报点之前。
  2. 多点统计：遍历多个起报点，报告分布。
  3. 能力诚实：模型不支持协变量则自动降级，并标记 covariates_used=False。

★ Spike-F1 防泄露：尖峰阈值用"测试期开始之前的全部历史"算（global 口径），
   或每个起报点用其之前历史滚动算（rolling 口径）。两种都不碰未来。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# 让本文件无论从哪运行都能 import 同级/兄弟模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                                   # evaluation/
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "models"))    # models/
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "data_processing"))  # loader

import metrics as M                       # noqa: E402
from base import Forecaster, Forecast     # noqa: E402

# 基础模型适配器（带批量接口）。没装大模型依赖时也要能跑基线，故容错导入。
try:
    from foundation import FoundationForecaster  # noqa: E402
except Exception:                                # pragma: no cover
    FoundationForecaster = ()                     # isinstance(x, ()) 恒为 False


# ── 回测配置 ──────────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    context_len: int = 168          # 每个起报点回看多少步（零样本/统计/基础模型用）
    horizon: int = 24               # 预测多少步
    stride: int = 24                # 起报点间隔（步数）
    test_start: Optional[str] = None  # 测试期起点（含）；None=数据中段自动定
    test_end: Optional[str] = None    # 测试期终点（含）
    spike_quantile: float = 0.95
    spike_mode: str = "global"      # global / rolling
    max_origins: Optional[int] = None  # 限制起报点数量（调试用）
    # 需训练模型（needs_training=True，如树模型/神经网）专用的训练窗口长度。
    # 168 步（7 天）不足以训练树模型/神经网，给它们额外回看更长历史（仍严格
    # 截断在起报点之前 → 无泄漏）。None=不启用，所有模型都用 context_len。
    train_context_len: Optional[int] = None
    # 多变量旋钮（手册 §6 消融 C）：
    #   True  = 多节点【联合建模】，把一个节点组的多列序列一起喂给模型，
    #           让模型利用节点间空间相关性（Toto/Chronos2 的强项）。
    #   False = 单变量，每个节点【独立】预测（逐列分别建模）。
    # 不支持多变量的模型（如 TimesFM 本期）即便置 True 也自动降级为逐列，
    # 并在结果里写 multivariate_used=False（能力诚实原则，手册 §3.3）。
    multivariate: bool = False


# ── 单条预测记录 ──────────────────────────────────────────────────────────────
@dataclass
class _Records:
    """收集所有 (起报点×节点×时刻) 的预测与真值，最后统一算指标。"""
    rows: list = field(default_factory=list)

    def add(self, model, origin, node, ts, actual, mean, q10, q90):
        self.rows.append({
            "model": model, "origin": origin, "node": node, "ts": ts,
            "actual": actual, "mean": mean, "q10": q10, "q90": q90,
        })

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ── 起报点生成 ────────────────────────────────────────────────────────────────
def _make_origins(index: pd.DatetimeIndex, cfg: BacktestConfig) -> List[int]:
    """返回所有合法起报点在 index 中的位置（整数下标）。"""
    n = len(index)
    # 起报点前必须留够历史：取零样本上下文与训练窗口里更大的那个
    min_hist = max(cfg.context_len, cfg.train_context_len or 0)
    # 测试期下标范围
    if cfg.test_start is not None:
        lo = index.searchsorted(pd.to_datetime(cfg.test_start, utc=True))
    else:
        lo = min_hist                             # 至少留够上下文
    if cfg.test_end is not None:
        hi = index.searchsorted(pd.to_datetime(cfg.test_end, utc=True))
    else:
        hi = n - cfg.horizon

    lo = max(lo, min_hist)                        # 起报点前必须有完整历史
    hi = min(hi, n - cfg.horizon)                 # 起报点后必须有完整 horizon

    origins = list(range(lo, hi + 1, cfg.stride))
    if cfg.max_origins is not None:
        origins = origins[: cfg.max_origins]
    return origins


# ── 统一预测入口：一次性拿到某模型在【所有起报点】上的预测 ────────────────────
def _predict_all_origins(
    fc: Forecaster,
    data: pd.DataFrame,
    origins: List[int],
    cfg: BacktestConfig,
    target_cols: List[str],
    cov_cols: List[str],
    use_cov: bool,
    future_known: bool,
) -> Dict[int, Forecast]:
    """
    返回 {origin_index -> Forecast}。

    - 基础模型(FoundationForecaster)：把所有起报点打包，**一次子进程**批量预测，
      模型只加载一次（避免逐 origin 起子进程）。结果还会被适配器内部缓存，
      因此 run_backtest 与 _spike_f1 阶段重复请求时直接命中缓存。
    - 基线模型：逐起报点调用 predict()（纯 numpy，开销可忽略）。
    """
    # 需训练模型用更长的训练窗口（若配置了 train_context_len），其余用 context_len。
    # 两者都只取起报点 oi 之前的数据 → 无泄漏。
    if getattr(fc, "needs_training", False) and cfg.train_context_len:
        ctx_len = cfg.train_context_len
    else:
        ctx_len = cfg.context_len

    # 先准备每个起报点的 context / future_cov（两条路径共用）
    valid_oi: List[int] = []
    context_dfs: List[pd.DataFrame] = []
    future_covs: List[Optional[pd.DataFrame]] = []
    for oi in origins:
        ctx = data.iloc[oi - ctx_len: oi]
        fut = data.iloc[oi: oi + cfg.horizon]
        if len(fut) < cfg.horizon:
            continue
        ctx_in = ctx[target_cols + (cov_cols if use_cov else [])]
        future_cov = fut[cov_cols] if (use_cov and future_known) else None
        valid_oi.append(oi)
        context_dfs.append(ctx_in)
        future_covs.append(future_cov)

    out: Dict[int, Forecast] = {}
    if isinstance(fc, FoundationForecaster):
        # 多变量旋钮：仅当本次实验要求多变量、且模型原生支持时才联合建模；
        # 否则降级为逐列单变量（worker 内部据此切换）。这一降级会反映在
        # fc.multivariate_used 上，由回测主循环写进结果（能力诚实）。
        forecasts = fc.predict_batch(
            context_dfs, future_covs, cfg.horizon,
            multivariate=cfg.multivariate)
        for oi, f in zip(valid_oi, forecasts):
            out[oi] = f
    else:
        # 基线（朴素/统计/树模型）均为单变量模型：逐起报点、逐列独立预测，
        # 与多变量旋钮无关。
        for oi, ctx_in, future_cov in zip(valid_oi, context_dfs, future_covs):
            out[oi] = fc.predict(ctx_in, future_covariates=future_cov,
                                 horizon=cfg.horizon)
    return out


def _as_2d(forecast: Forecast):
    """把 Forecast 的 mean/q10/q90 统一成 (horizon, n_series)。"""
    if forecast.mean.ndim == 1:
        mean = forecast.mean[:, None]
        q90 = forecast.q90[:, None] if forecast.q90 is not None else mean
        q10 = forecast.q10[:, None] if forecast.q10 is not None else mean
    else:
        mean = forecast.mean
        q90 = forecast.q90 if forecast.q90 is not None else mean
        q10 = forecast.q10 if forecast.q10 is not None else mean
    return mean, q10, q90


# ── 主回测函数 ────────────────────────────────────────────────────────────────
def run_backtest(
    data: pd.DataFrame,
    forecasters: List[Forecaster],
    cfg: BacktestConfig,
    covariates: Optional[List[str]] = None,
    future_covariates_known: bool = True,
) -> Dict[str, object]:
    """
    在一份对齐好的数据上，对多个模型做滚动回测。

    参数
    ----
    data : load_slice() 的输出。含 price__<node> 目标列 + 可选协变量列，
           索引为时间戳。
    forecasters : Forecaster 实例列表。
    cfg : BacktestConfig。
    covariates : 本次实验想用的协变量列名（必须是 data 里已有的列）。
                 模型不支持协变量时自动降级（手册 §3.3）。
    future_covariates_known : 预测窗口内是否提供未来协变量真值
                 （手册 §2.3 的"未来真值上界分析"，默认 True）。

    返回
    ----
    {
      "summary":  DataFrame[每模型的指标均值±标准差],
      "per_origin": DataFrame[每起报点每模型的指标],
      "records":  DataFrame[逐时刻原始预测],
      "thresholds": {node: 尖峰阈值},
    }
    """
    covariates = covariates or []
    target_cols = [c for c in data.columns if c.startswith("price__")]
    nodes = [c[len("price__"):] for c in target_cols]
    cov_cols = [c for c in covariates if c in data.columns]

    origins = _make_origins(data.index, cfg)
    if not origins:
        raise ValueError("没有合法起报点：检查 context_len/horizon/测试期是否超出数据范围")

    # ── 尖峰阈值（防泄露）──────────────────────────────────────────────────
    # global 口径：用第一个起报点之前的全部历史，按节点算 P95，全程固定。
    thresholds = {}
    first_origin_ts = data.index[origins[0]]
    hist_before_test = data.loc[data.index < first_origin_ts]
    for node, col in zip(nodes, target_cols):
        vals = hist_before_test[col].to_numpy(dtype=float)
        if vals.size == 0:
            vals = data[col].to_numpy(dtype=float)   # 兜底
        thresholds[node] = M.compute_spike_threshold(vals, cfg.spike_quantile)

    # ── 先跑一遍季节性朴素，得到 MASE 的分母（每模型同口径）──────────────
    # 这里直接在汇总阶段用记录里的 naive，简单起见用全局 naive_mae 占位。
    rec = _Records()
    per_origin_rows = []

    import time as _time
    for _mi, fc in enumerate(forecasters, 1):
        use_cov = fc.supports_covariates and len(cov_cols) > 0
        print(f"  [{_mi}/{len(forecasters)}] {fc.name} 预测中 "
              f"({'train' if fc.needs_training else 'zeroshot'}) …", flush=True)
        _t0 = _time.time()
        # ★ 一次性拿到该模型在所有起报点上的预测：
        #   基础模型走批量子进程（只加载一次），基线走逐点循环。
        preds = _predict_all_origins(
            fc, data, origins, cfg, target_cols, cov_cols,
            use_cov, future_covariates_known)
        print(f"      ↳ {fc.name} 完成，用时 {_time.time() - _t0:.1f}s", flush=True)

        for oi in origins:
            if oi not in preds:
                continue
            origin_ts = data.index[oi]
            fut = data.iloc[oi: oi + cfg.horizon]
            mean, q10, q90 = _as_2d(preds[oi])   # (horizon, n_series)
            actual = fut[target_cols].to_numpy(dtype=float)   # (horizon, n_series)

            for j, node in enumerate(nodes):
                for h in range(cfg.horizon):
                    rec.add(fc.name, origin_ts, node, fut.index[h],
                            actual[h, j], mean[h, j], q10[h, j], q90[h, j])

            # 记录这个 (模型, 起报点) 的指标（按所有节点汇总）
            m = M.all_point_prob_metrics(
                actual.ravel(), mean.ravel(),
                q10.ravel(), mean.ravel(), q90.ravel())
            # multivariate_used：实验要求多变量 且 模型原生支持，才算真正用上。
            mv_used = bool(cfg.multivariate
                           and getattr(fc, "supports_multivariate", False)
                           and len(nodes) > 1)
            per_origin_rows.append({
                "model": fc.name, "origin": origin_ts,
                "covariates_used": use_cov,
                "multivariate_used": mv_used, **m})

    records = rec.to_df()           # 逐时刻记录，已带 model 列
    per_origin = pd.DataFrame(per_origin_rows)

    summary = _summarize(per_origin, records, forecasters, nodes,
                         target_cols, thresholds, cfg, data, origins,
                         cov_cols, future_covariates_known)

    return {
        "summary": summary,
        "per_origin": per_origin,
        "records": records,
        "thresholds": thresholds,
    }


def _summarize(per_origin, records, forecasters, nodes, target_cols,
               thresholds, cfg, data, origins, cov_cols, future_known):
    """
    汇总每个模型：点/概率指标的均值±标准差 + 全局 Spike-F1。
    Spike-F1 需要逐时刻信号，因此这里按模型重跑一次轻量收集（带模型名）。
    """
    rows = []
    # 先算季节性朴素的 MAE 作为 MASE 分母
    naive_mae = None
    for fc in forecasters:
        if fc.name == "SeasonalNaive":
            sub = per_origin[per_origin["model"] == "SeasonalNaive"]
            if len(sub):
                naive_mae = float(sub["mae"].mean())

    for fc in forecasters:
        sub = per_origin[per_origin["model"] == fc.name]
        if sub.empty:
            continue
        row = {"model": fc.name,
               "covariates_used": bool(sub["covariates_used"].iloc[0]),
               "multivariate_used": bool(sub["multivariate_used"].iloc[0])
               if "multivariate_used" in sub else False,
               "n_origins": int(sub["origin"].nunique())}
        for metric in ("mae", "rmse", "smape", "pinball", "coverage"):
            if metric in sub:
                row[f"{metric}_mean"] = float(sub[metric].mean())
                row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        if naive_mae:
            row["mase_mean"] = float(sub["mae"].mean() / naive_mae)

        # Spike-F1：对该模型逐时刻收集 mean/q90 信号
        sf_mean, sf_q90 = _spike_f1_for_model(
            fc, data, origins, nodes, target_cols, thresholds, cfg,
            cov_cols, future_known)
        row["spike_f1_mean_signal"] = sf_mean["spike_f1"]
        row["spike_precision"] = sf_mean["precision"]
        row["spike_recall"] = sf_mean["recall"]
        row["spike_f1_q90_signal"] = sf_q90["spike_f1"]
        rows.append(row)

    return pd.DataFrame(rows)


def _spike_f1_for_model(fc, data, origins, nodes, target_cols, thresholds,
                        cfg, cov_cols, future_known):
    """对单个模型在所有起报点上汇总 Spike-F1（mean 信号 和 q90 信号各一套）。"""
    use_cov = fc.supports_covariates and len(cov_cols) > 0
    y_true, sig_mean, sig_q90, thr_arr = [], [], [], []

    # 复用统一预测入口：基础模型会命中内部缓存（run_backtest 已算过），不再起子进程。
    preds = _predict_all_origins(
        fc, data, origins, cfg, target_cols, cov_cols, use_cov, future_known)

    for oi in origins:
        if oi not in preds:
            continue
        fut = data.iloc[oi: oi + cfg.horizon]
        mean, _q10, q90 = _as_2d(preds[oi])
        actual = fut[target_cols].to_numpy(dtype=float)
        for j, node in enumerate(nodes):
            y_true.append(actual[:, j])
            sig_mean.append(mean[:, j])
            sig_q90.append(q90[:, j])
            thr_arr.append(np.full(cfg.horizon, thresholds[node]))

    if not y_true:
        empty = {"precision": 0.0, "recall": 0.0, "spike_f1": 0.0}
        return empty, empty

    y_true = np.concatenate(y_true)
    sig_mean = np.concatenate(sig_mean)
    sig_q90 = np.concatenate(sig_q90)
    thr_arr = np.concatenate(thr_arr)

    # 阈值按节点不同，逐元素比较：true/pred = value >= 各自阈值
    def _f1(signal):
        true_spike = y_true >= thr_arr
        pred_spike = signal >= thr_arr
        tp = int(np.sum(pred_spike & true_spike))
        fp = int(np.sum(pred_spike & ~true_spike))
        fn = int(np.sum(~pred_spike & true_spike))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": p, "recall": r, "spike_f1": f1}

    return _f1(sig_mean), _f1(sig_q90)
