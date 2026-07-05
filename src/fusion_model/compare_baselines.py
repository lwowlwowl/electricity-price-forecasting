"""
compare_baselines.py — ElecFM vs LEAR / SeasonalNaive 对比评估
==============================================================
对应 Lago 2021 checklist #2（对比开源 SOTA 基线）。

功能：
  1. 用 backtest 引擎在连续 12 个月测试期跑 LEAR / SeasonalNaive / Naive
  2. 加载预先计算好的 ElecFM continuous 记录（records.csv）
  3. 计算 rMAE / MAE / Spike-F1 三类指标对比表
  4. 在 ElecFM vs LEAR 之间做 GW test（日度 L1 差，HAC 稳健）
  5. 输出 comparison_summary.csv + gw_tests.csv

用法
----
  # 使用默认配置（ERCOT volatility 节点，连续测试期）
  python compare_baselines.py

  # 指定不同市场或节点
  python compare_baselines.py --market PJM --nodes_group ablation

  # 跳过 LEAR（只跑 naive baselines，速度快）
  python compare_baselines.py --skip_lear

  # 指定 ElecFM 结果目录
  python compare_baselines.py --elecfm_dir data/results/fusion/v7_full/fusion_electfm_continuous

注意
----
  - LEAR 每天重新校准（1-10s/天），~335 起报点需 1-3h
  - ElecFM records 不存在时自动跳过合并（仍输出基线对比）
  - GW test 适用于 ElecFM vs LEAR（两者都有 recalibration 机制）；
    vs SeasonalNaive 用 DM test（SeasonalNaive 固定参数）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _SCRIPT_DIR)                                  # src/fusion_model (dataset.py)
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
sys.path.insert(0, os.path.join(_ROOT, "src", "models"))
sys.path.insert(0, os.path.join(_ROOT, "src", "evaluation"))

import loader                                       # noqa: E402
from backtest import run_backtest, BacktestConfig   # noqa: E402
from forecasters import build_forecaster            # noqa: E402
from dataset import TEST_START, TEST_END, TRAIN_END  # noqa: E402

_NODES_YAML = os.path.join(_ROOT, "configs", "nodes.yaml")
_RESULTS_ROOT = os.path.join(_ROOT, "data", "results")


# ── 配置 ──────────────────────────────────────────────────────────────────────
DEFAULT_MARKET     = "ERCOT"
DEFAULT_NODES_GRP  = "volatility"  # nodes.yaml 中的节点组
DEFAULT_CONTEXT    = 168           # 基础模型回看窗口
DEFAULT_TRAIN_CTX  = 84 * 24      # LEAR 滚动校准窗口 (84 天 = 12 周)
DEFAULT_HORIZON    = 24
SPIKE_QUANTILE     = 0.95
RMAE_FLOOR         = 1.0           # rMAE 分母下限（防 CAISO 负价格爆炸）


def _load_nodes(market: str, group: str) -> List[str]:
    import yaml
    with open(_NODES_YAML) as f:
        cfg = yaml.safe_load(f)
    entry = cfg[market]["ablation"][group]
    # ablation 组的值可以是节点列表或嵌套 dict（volatility/spikes/stable）
    if isinstance(entry, list):
        return entry
    # 如果是嵌套 dict，取第一个列表（volatility 组）
    return next(iter(entry.values())) if isinstance(entry, dict) else [entry]


def _run_baselines(
    market: str,
    nodes: List[str],
    models: List[str],
    train_context_len: int = DEFAULT_TRAIN_CTX,
) -> dict:
    """
    在连续 12 个月测试期跑 baselines，返回 run_backtest 的结果字典。
    """
    print(f"\n加载数据：{market}  节点={nodes}")
    data = loader.load_slice(
        market=market, nodes=nodes, freq="1h",
        start="2025-01-01", end=TEST_END,
    )
    print(f"  数据维度：{data.shape}  {data.index[0]} → {data.index[-1]}")

    forecasters = [build_forecaster(m) for m in models]
    print(f"\n模型：{[f.name for f in forecasters]}")

    cfg = BacktestConfig(
        context_len=DEFAULT_CONTEXT,
        horizon=DEFAULT_HORIZON,
        stride=DEFAULT_HORIZON,          # 每天一个起报点
        test_start=TEST_START.split()[0],
        test_end=TEST_END.split()[0],
        spike_quantile=SPIKE_QUANTILE,
        spike_mode="global",
        train_context_len=train_context_len,
    )

    print(f"\n开始滚动回测（{TEST_START.split()[0]} ~ {TEST_END.split()[0]}）…")
    return run_backtest(data, forecasters, cfg)


def _load_elecfm_records(elecfm_dir: Optional[str]) -> Optional[pd.DataFrame]:
    """
    从 ElecFM continuous 评估结果加载 records.csv。
    返回 None 若文件不存在（跳过 ElecFM 对比）。
    """
    if elecfm_dir is None:
        # 自动搜索最新 fusion 结果
        fusion_root = os.path.join(_RESULTS_ROOT, "fusion")
        if not os.path.isdir(fusion_root):
            return None
        # 找所有包含 fusion_electfm_continuous/records.csv 的目录
        candidates = []
        for d in sorted(os.listdir(fusion_root)):
            p = os.path.join(fusion_root, d, "fusion_electfm_continuous", "records.csv")
            if os.path.isfile(p):
                candidates.append((os.path.getmtime(p), p))
        if not candidates:
            return None
        _, path = max(candidates)
    else:
        path = os.path.join(elecfm_dir, "records.csv")
        if not os.path.isfile(path):
            print(f"  ⚠️  ElecFM records 不存在：{path}，跳过 ElecFM 对比")
            return None

    print(f"  加载 ElecFM records：{path}")
    df = pd.read_csv(path)
    df["model"] = "ElecFM"   # 统一模型名
    return df


def _compute_rmae_from_records(
    records: pd.DataFrame,
    data: pd.DataFrame,
    target_cols: List[str],
    origins: List[pd.Timestamp],
    horizon: int = 24,
) -> pd.DataFrame:
    """
    从 records 和原始 data 计算每个模型的 rMAE。

    records 已有 actual 列，但没有 naive7 预测。
    通过 data 回查 168 步前实际价格得 naive7 MAE，再算 rMAE。

    返回 DataFrame，列：model, rmae, naive7_mae_mean
    """
    # 构建 origin → naive7_mae 映射
    origin_n7 = {}
    idx_map = {ts: i for i, ts in enumerate(data.index)}
    for origin_ts in origins:
        oi = idx_map.get(pd.Timestamp(origin_ts))
        if oi is None:
            continue
        errors = []
        for h in range(horizon):
            actual_i = oi + h
            naive_i  = oi + h - 168
            if naive_i < 0 or actual_i >= len(data):
                continue
            for col in target_cols:
                va = float(data.iloc[actual_i][col])
                vn = float(data.iloc[naive_i][col])
                if not (np.isnan(va) or np.isnan(vn)):
                    errors.append(abs(va - vn))
        origin_n7[pd.Timestamp(origin_ts)] = (
            max(float(np.mean(errors)), RMAE_FLOOR) if errors else float("nan"))

    rows = []
    for model, grp in records.groupby("model"):
        origins_m = grp["origin"].unique()
        rmae_vals = []
        for org in origins_m:
            n7 = origin_n7.get(pd.Timestamp(org))
            if n7 is None or np.isnan(n7):
                continue
            sub = grp[grp["origin"] == org]
            model_mae = float(np.abs(sub["actual"] - sub["mean"]).mean())
            rmae_vals.append(model_mae / n7)
        if rmae_vals:
            rows.append({"model": model,
                         "rmae": float(np.mean(rmae_vals)),
                         "naive7_mae_mean": float(np.mean(list(origin_n7.values())))})
    return pd.DataFrame(rows)


def _gw_test_elecfm_vs_baseline(
    all_records: pd.DataFrame,
    elecfm_name: str = "ElecFM",
    baseline_name: str = "LEAR",
) -> Optional[dict]:
    """
    ElecFM vs baseline 的 GW test（日度 L1 损失差，HAC 稳健）。
    适用于两者均有 recalibration 机制的场景（see TODO §1.2 说明）。
    """
    try:
        from stat_tests import gw_test, dm_test
    except ImportError:
        print("  ⚠️  stat_tests 不可用，跳过 GW/DM test")
        return None

    # 检查两个模型都存在
    models_in_records = set(all_records["model"].unique())
    if elecfm_name not in models_in_records or baseline_name not in models_in_records:
        print(f"  ⚠️  {elecfm_name} 或 {baseline_name} 不在 records 中，跳过")
        return None

    # ElecFM vs LEAR → GW test（recalibration 双方都有）
    gw = gw_test(all_records, elecfm_name, baseline_name)
    gw["test_type"] = "GW"

    # ElecFM vs SeasonalNaive → DM test（SeasonalNaive 固定参数）
    dm_results = []
    for sn in ("SeasonalNaive", "Naive"):
        if sn in models_in_records:
            dm = dm_test(all_records, elecfm_name, sn)
            dm_results.append({
                "test_type": "DM",
                "model_a": elecfm_name,
                "model_b": sn,
                "dm_stat": float(dm["dm_stat"].mean()),
                "p_value": float(dm["p_value"].mean()),
                "sig_hours": int((dm["p_value"] < 0.05).sum()),
                "n_days": int(dm["n_days"].iloc[0]) if "n_days" in dm.columns else None,
            })

    return {"gw": gw, "dm_vs_naive": dm_results}


def compare(
    market: str = DEFAULT_MARKET,
    nodes_group: str = DEFAULT_NODES_GRP,
    nodes: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    skip_lear: bool = False,
    elecfm_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> dict:
    """
    主对比函数。

    Returns
    -------
    dict with keys: baseline_result, comparison_df, gw_results
    """
    # 节点
    if nodes is None:
        nodes = _load_nodes(market, nodes_group)
    print(f"节点：{nodes}")

    # 模型列表
    if models is None:
        models = ["SeasonalNaive", "Naive"]
        if not skip_lear:
            models.insert(0, "LEAR")
    target_cols = [f"price__{n}" for n in nodes]

    # ── 1. 跑 baselines ────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("第一步：跑 baseline 模型（连续 12 个月测试期）")
    print("=" * 64)
    baseline_result = _run_baselines(market, nodes, models)
    baseline_summary = baseline_result["summary"]
    baseline_records = baseline_result["records"]

    print("\n── 基线汇总（按 rMAE/MAE 排序）──")
    sort_col = "rmae_mean" if "rmae_mean" in baseline_summary.columns else "mae_mean"
    bs_sorted = baseline_summary.sort_values(sort_col)
    show = [c for c in ["model", "rmae_mean", "mae_mean", "smape_mean",
                        "spike_f1_mean_signal", "n_origins"]
            if c in bs_sorted.columns]
    print(bs_sorted[show].round(4).to_string(index=False))

    # ── 2. 加载 ElecFM records ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("第二步：加载 ElecFM 结果（若存在）")
    print("=" * 64)
    elecfm_records = _load_elecfm_records(elecfm_dir)

    # ── 3. 合并 records，计算跨模型指标 ────────────────────────────────────
    all_records = baseline_records.copy()
    if elecfm_records is not None:
        # 对齐列（ElecFM records 可能有额外列）
        common_cols = list(set(all_records.columns) & set(elecfm_records.columns))
        all_records = pd.concat(
            [all_records[common_cols], elecfm_records[common_cols]],
            ignore_index=True)
        print(f"  合并后共 {len(all_records)} 条记录，"
              f"模型：{sorted(all_records['model'].unique())}")

    # ── 4. rMAE（用 baseline 数据的原始 data 回查 naive7）──────────────────
    print("\n" + "=" * 64)
    print("第三步：计算 rMAE（naive7 period=168）")
    print("=" * 64)

    # 重新加载数据以获取 naive7 参考值
    data = loader.load_slice(
        market=market, nodes=nodes, freq="1h",
        start="2025-01-01", end=TEST_END,
    )
    # 从 baseline records 收集测试期内的所有 origins
    origins_in_records = all_records["origin"].unique()

    rmae_df = _compute_rmae_from_records(
        all_records, data, target_cols, origins_in_records)
    print(rmae_df.round(4).to_string(index=False))

    # ── 5. 合并汇总表 ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("第四步：构建汇总对比表")
    print("=" * 64)

    # 从 baseline_summary 取基础指标
    comparison_rows = []
    models_in_all = all_records["model"].unique()
    for model in sorted(models_in_all):
        row = {"model": model}
        # 点指标：从 all_records 按模型算
        sub = all_records[all_records["model"] == model]
        row["mae"]   = float(np.abs(sub["actual"] - sub["mean"]).mean())
        row["smape"] = float(
            (2 * np.abs(sub["actual"] - sub["mean"]) /
             (np.abs(sub["actual"]) + np.abs(sub["mean"]) + 1e-6)).mean() * 100)
        row["n_origins"] = int(sub["origin"].nunique())
        # Spike-F1：需 threshold（使用训练集 P95）
        # 直接用 backtest 结果中的 spike_f1（只对 baseline models 有）
        bs_row = baseline_summary[baseline_summary["model"] == model]
        if not bs_row.empty and "spike_f1_mean_signal" in bs_row.columns:
            row["spike_f1"] = float(bs_row["spike_f1_mean_signal"].iloc[0])
        # rMAE
        rmae_row = rmae_df[rmae_df["model"] == model]
        if not rmae_row.empty:
            row["rmae"] = float(rmae_row["rmae"].iloc[0])
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    sort_c = "rmae" if "rmae" in comparison_df.columns else "mae"
    comparison_df = comparison_df.sort_values(sort_c)
    print(comparison_df.round(4).to_string(index=False))

    # ── 6. GW / DM test ────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("第五步：统计检验（GW test: ElecFM vs LEAR；DM test: ElecFM vs Naive）")
    print("=" * 64)
    gw_results = _gw_test_elecfm_vs_baseline(all_records)

    if gw_results:
        gw = gw_results["gw"]
        print(f"\nGW test (ElecFM vs LEAR):")
        print(f"  dm_stat={gw['dm_stat']:.3f}  p={gw['p_value']:.4f}  "
              f"n_days={gw['n_days']}")
        if gw["p_value"] < 0.05:
            winner = "LEAR" if gw["mean_loss_diff"] > 0 else "ElecFM"
            print(f"  ✅ 显著（α=0.05），{winner} 更优")
        else:
            print(f"  ⚪ 不显著（α=0.05），两者无统计差异")
        for dm in gw_results.get("dm_vs_naive", []):
            print(f"\nDM test (ElecFM vs {dm['model_b']}):")
            print(f"  dm_stat_mean={dm['dm_stat']:.3f}  "
                  f"p_mean={dm['p_value']:.4f}  "
                  f"sig_hours={dm['sig_hours']}/24")

    # ── 7. 落盘 ────────────────────────────────────────────────────────────
    if out_dir is None:
        out_dir = os.path.join(_RESULTS_ROOT, "fusion",
                               f"baseline_comparison_{market.lower()}")
    os.makedirs(out_dir, exist_ok=True)

    comparison_df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    baseline_result["summary"].to_csv(
        os.path.join(out_dir, "baseline_summary.csv"), index=False)

    if gw_results:
        gw_rows = [gw_results["gw"]] + gw_results.get("dm_vs_naive", [])
        pd.DataFrame(gw_rows).to_csv(
            os.path.join(out_dir, "stat_tests.csv"), index=False)

    print(f"\n✅ 对比结果已写入：{out_dir}/")
    print(f"   comparison_summary.csv / baseline_summary.csv / stat_tests.csv")

    return {
        "baseline_result": baseline_result,
        "comparison_df": comparison_df,
        "gw_results": gw_results,
        "out_dir": out_dir,
    }


# ── CLI 入口 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ElecFM vs LEAR / baseline 对比评估（Lago checklist #2）")
    parser.add_argument("--market",      default=DEFAULT_MARKET)
    parser.add_argument("--nodes_group", default=DEFAULT_NODES_GRP,
                        help="nodes.yaml 中的节点组（ablation=3代表节点 / all=全节点）")
    parser.add_argument("--skip_lear",   action="store_true",
                        help="跳过 LEAR（节省 1-3h）")
    parser.add_argument("--elecfm_dir",  default=None,
                        help="ElecFM continuous 结果目录（含 records.csv）")
    parser.add_argument("--out_dir",     default=None,
                        help="输出目录（默认 data/results/fusion/baseline_comparison_<market>）")
    args = parser.parse_args()

    compare(
        market=args.market,
        nodes_group=args.nodes_group,
        skip_lear=args.skip_lear,
        elecfm_dir=args.elecfm_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
