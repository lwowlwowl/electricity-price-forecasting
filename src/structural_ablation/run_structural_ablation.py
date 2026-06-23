"""
结构消融实验入口 run_structural_ablation.py
============================================
手册 v2.1 §6 "结构层消融"的执行器。与现有参数消融 run_ablation.py 并列，
不修改任何原有代码。

工作流程：
  1) 读取结构消融配置 YAML
  2) 加载数据（复用 loader.load_slice）
  3) 为每个 (model, ablation_type) 组合创建一个 AblationForecaster
  4) 通过现有回测引擎 run_backtest 跑多起报点滚动回测
  5) 汇总跨消融对比总表 + 折线图

配置格式示例（configs/structural_ablation/toto2_attention.yaml）：

    name: structural_toto2_attention
    market: ERCOT
    nodes_group: volatility
    freq: 1h
    context_len: 168
    horizon: 24
    data_start: "2025-06-01"
    data_end: "2025-09-01"
    backtest:
      test_start: "2025-08-01"
      test_end: "2025-08-31"
      stride_hours: 24
      max_origins: 30
    spike:
      quantile: 0.95
      mode: global
    multivariate: false
    structural_ablation:
      models: [toto2, chronos2]
      ablations: [skip_attention, halve_heads, skip_ffn]
      include_baseline: true       # 同时跑无消融基线
      baseline_models: [Naive, SeasonalNaive]  # 可选：一起跑的基线模型

用法：
    python run_structural_ablation.py configs/structural_ablation/toto2_attention.yaml
"""

from __future__ import annotations

import os
import sys
import json
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# ── 路径设置 ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "data_processing"))
sys.path.insert(0, os.path.join(SRC_DIR, "models"))
sys.path.insert(0, os.path.join(SRC_DIR, "evaluation"))
sys.path.insert(0, os.path.join(SRC_DIR, "parameter_ablation"))

from loader import load_slice                        # noqa: E402
from base import Forecaster, Forecast                # noqa: E402
from backtest import run_backtest, BacktestConfig     # noqa: E402
from foundation_ablation import (                    # noqa: E402
    AblationForecaster,
    build_ablation_forecaster,
    ABLATION_WORKERS_DIR,
)

# 尝试导入绘图（不影响核心流程）
try:
    from plotting import plot_ablation               # noqa: E402
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# 尝试导入基线 forecasters builder
try:
    from forecasters import build_forecaster         # noqa: E402
    HAS_BASELINE = True
except ImportError:
    HAS_BASELINE = False

RESULTS_DIR = os.path.join(ROOT, "data", "results")
CONFIG_DIR = os.path.join(ROOT, "configs")
NODES_YAML = os.path.join(CONFIG_DIR, "nodes.yaml")


# ── YAML 加载（与 run_experiment.py 相同逻辑）────────────────────────────────
def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return _tiny_yaml(path)


def _tiny_yaml(path: str) -> dict:
    root: dict = {}
    stack = [(-1, root)]
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#")[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            key, _, val = line.strip().partition(":")
            key, val = key.strip(), val.strip()
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1]
            if val == "":
                parent[key] = {}
                stack.append((indent, parent[key]))
            else:
                parent[key] = _parse_scalar(val)
    return root


def _parse_scalar(v: str):
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [x.strip() for x in inner.split(",")] if inner else []
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    # 去除引号
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        return v[1:-1]
    return v


def _read_nodes_group(market: str, group: str):
    cfg = _load_yaml(NODES_YAML)
    return cfg[market][group]


# ── 让 run_backtest 识别 AblationForecaster 的批量接口 ─────────────────────
# backtest.py 的 _predict_all_origins 用 isinstance(fc, FoundationForecaster)
# 来决定是否走 predict_batch 路径。为了不修改 backtest.py 文件，我们在运行时
# 给 backtest 模块注入识别逻辑。
import backtest as _bt  # noqa: E402

_original_predict_all_origins = _bt._predict_all_origins


def _patched_predict_all_origins(fc, data, origins, cfg, target_cols, cov_cols,
                                 use_cov, future_known):
    """
    包装原 _predict_all_origins，对 AblationForecaster 走 predict_batch 批量路径。
    """
    if isinstance(fc, AblationForecaster):
        # 复制原函数的数据准备逻辑
        if getattr(fc, "needs_training", False) and cfg.train_context_len:
            ctx_len = cfg.train_context_len
        else:
            ctx_len = cfg.context_len

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
        forecasts = fc.predict_batch(
            context_dfs, future_covs, cfg.horizon,
            multivariate=cfg.multivariate)
        for oi, f in zip(valid_oi, forecasts):
            out[oi] = f
        return out
    else:
        return _original_predict_all_origins(
            fc, data, origins, cfg, target_cols, cov_cols, use_cov, future_known)


# 运行时 monkey-patch（仅影响本脚本的 runtime，不修改文件）
_bt._predict_all_origins = _patched_predict_all_origins


# ── 消融适用性表（镜像自 ablations.py，避免在主进程中 import torch）──────────
# 注意：如果 ablations.py 中增加了新的消融类型，这里也需要同步更新。
ABLATION_APPLICABILITY = {
    "skip_attention":        ["toto2", "chronos2", "timesfm"],
    "halve_heads":           ["toto2", "chronos2", "timesfm"],
    "skip_variate":          ["toto2", "chronos2"],
    "skip_time":             ["toto2", "chronos2"],
    "remove_rope":           ["toto2", "chronos2", "timesfm"],
    "disable_xpos":          ["toto2"],
    "simplify_patch_emb":    ["toto2", "chronos2", "timesfm"],
    "simplify_output_head":  ["chronos2", "timesfm"],  # toto2 不适用
    "point_only":            ["timesfm"],
    "skip_ffn":              ["toto2", "chronos2", "timesfm"],
    "skip_layernorm":        ["toto2", "chronos2", "timesfm"],
    "truncate_front_half":   ["toto2", "chronos2", "timesfm"],
    "truncate_front_quarter": ["toto2", "chronos2", "timesfm"],
    "truncate_back_half":    ["toto2", "chronos2", "timesfm"],
    "skip_layer":            ["toto2", "chronos2", "timesfm"],
}


def _filter_applicable(models: List[str], ablations: List[str]) -> List[tuple]:
    """返回 [(model_type, ablation_type), ...] 所有合法的组合。
    支持 skip_layer_N 格式：自动解析为 skip_layer 查找适用性。"""
    pairs = []
    for model in models:
        for abl in ablations:
            # skip_layer_N 格式：解析 base type 为 skip_layer
            base_abl = "skip_layer" if abl.startswith("skip_layer_") else abl
            applicable = ABLATION_APPLICABILITY.get(base_abl, [])
            if model in applicable:
                pairs.append((model, abl))
            else:
                print(f"  ⚠️  跳过不适用组合：{model} × {abl}")
    return pairs


# ── 主函数 ────────────────────────────────────────────────────────────────
def run_structural_ablation(cfg: dict) -> dict:
    """
    执行结构消融实验。

    返回：{
        "merged":  pd.DataFrame  跨消融对比总表,
        "summary_path": str,
        "png_path": str | None,
    }
    """
    name = cfg["name"]
    market = cfg["market"]
    group = cfg.get("nodes_group")
    nodes = cfg.get("nodes") or _read_nodes_group(market, group)
    freq = cfg.get("freq", "1h")
    covariates = cfg.get("covariates", []) or []

    sa_cfg = cfg["structural_ablation"]
    models = sa_cfg["models"]                       # e.g. ["toto2", "chronos2"]
    ablations = sa_cfg["ablations"]                 # e.g. ["skip_attention", "halve_heads"]
    include_baseline = sa_cfg.get("include_baseline", True)
    baseline_models = sa_cfg.get("baseline_models", ["Naive", "SeasonalNaive"])

    print("=" * 70)
    print(f"结构消融实验：{name}")
    print("=" * 70)
    print(f"市场={market}  节点={nodes}  频率={freq}")
    print(f"上下文={cfg['context_len']}  步长={cfg['horizon']}  协变量={covariates or '无'}")
    print(f"消融模型：{models}")
    print(f"消融类型：{ablations}")
    if include_baseline:
        print(f"基线模型：{baseline_models}")

    # 1) 取数
    data = load_slice(
        market=market, nodes=nodes, freq=freq,
        covariates=covariates,
        start=cfg.get("data_start"), end=cfg.get("data_end"),
    )
    print(f"\n取数完成：{data.shape}  {data.index[0]} → {data.index[-1]}")

    # 2) 回测配置
    bt = cfg.get("backtest", {})
    bcfg = BacktestConfig(
        context_len=cfg["context_len"],
        horizon=cfg["horizon"],
        stride=bt.get("stride_hours", 24),
        test_start=bt.get("test_start"),
        test_end=bt.get("test_end"),
        spike_quantile=cfg.get("spike", {}).get("quantile", 0.95),
        spike_mode=cfg.get("spike", {}).get("mode", "global"),
        max_origins=bt.get("max_origins"),
        train_context_len=cfg.get("train_context_len") or bt.get("train_context_len"),
        multivariate=bool(cfg.get("multivariate", False)),
    )

    # 3) 构造 forecasters
    pairs = _filter_applicable(models, ablations)
    forecasters: List[Forecaster] = []

    # 无消融基线（每个消融模型也跑一个 baseline = 无消融的原始推理）
    if include_baseline:
        for model in models:
            fc_baseline = build_ablation_forecaster(model, "none")
            # "none" 消融 = 不做任何修改，直接推理（作为对照组）
            fc_baseline.name = f"{model.capitalize()}[baseline]"
            forecasters.append(fc_baseline)

    # 消融版
    for model, abl in pairs:
        fc = build_ablation_forecaster(model, abl)
        forecasters.append(fc)

    # 额外基线（Naive 等纯统计模型）
    if include_baseline and HAS_BASELINE:
        for m in baseline_models:
            try:
                forecasters.append(build_forecaster(m))
            except Exception as e:
                print(f"  ⚠️  构建基线模型 {m} 失败：{e}")

    print(f"\n模型列表（共 {len(forecasters)} 个）：")
    for fc in forecasters:
        print(f"  - {fc.name}")

    # 4) 滚动回测
    print("\n开始滚动回测 …")
    result = run_backtest(data, forecasters, bcfg, covariates=covariates)

    summary = result["summary"]
    print("\n── 结果汇总（按 MAE 升序）──")
    show_cols = [c for c in ["model", "n_origins",
                             "mae_mean", "mae_std", "rmse_mean", "mase_mean",
                             "coverage_mean", "spike_f1_mean_signal"]
                 if c in summary.columns]
    summary_sorted = summary.sort_values("mae_mean")
    print(summary_sorted[show_cols].round(4).to_string(index=False))

    # 5) 落盘
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    summary_sorted.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    result["per_origin"].to_csv(os.path.join(out_dir, "per_origin.csv"), index=False)
    result["records"].to_csv(os.path.join(out_dir, "records.csv"), index=False)
    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump(result["thresholds"], f, indent=2)

    # 跨消融对比总表
    merged_rows = []
    for model in models:
        # 基线行
        baseline_name = f"{model.capitalize()}[baseline]"
        row = summary_sorted[summary_sorted["model"] == baseline_name].copy()
        if not row.empty:
            row = row.copy()
            row["ablation_type"] = "baseline"
            row["model_type"] = model
            merged_rows.append(row)
        # 消融行
        for abl in ablations:
            abl_name = f"{model.capitalize()}[{abl}]"
            # 适配大小写：forecaster 可能用 Toto2 / Chronos2 / TimesFM
            for actual_name in [abl_name, f"{model}[{abl}]",
                                f"Toto2[{abl}]", f"Chronos2[{abl}]", f"TimesFM[{abl}]"]:
                row = summary_sorted[summary_sorted["model"] == actual_name].copy()
                if not row.empty:
                    row = row.copy()
                    row["ablation_type"] = abl
                    row["model_type"] = model
                    merged_rows.append(row)
                    break

    merged = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
    merged_path = os.path.join(out_dir, "structural_ablation_summary.csv")
    if not merged.empty:
        merged.to_csv(merged_path, index=False)
        print(f"\n✅ 结构消融对比总表：{merged_path}")

        # 打印 delta 对比
        _print_delta(merged)

    # 6) 折线图
    png_path = None
    if HAS_PLOTTING and not merged.empty:
        try:
            per_level = []
            for abl in ["baseline"] + ablations:
                sub = merged[merged["ablation_type"] == abl]
                if not sub.empty:
                    per_level.append((abl, sub))
            png_path = os.path.join(out_dir, "structural_ablation_compare.png")
            plot_ablation(
                per_level,
                knob_label="消融类型",
                out_png=png_path,
                title=f"结构消融：{name}",
            )
            print(f"📊 消融对比图：{png_path}")
        except Exception as e:
            print(f"⚠️  出图失败（不影响结果）：{e}")

    print(f"\n{'=' * 70}")
    print(f"✅ 结构消融实验完成：{out_dir}/")
    print("=" * 70)

    return {
        "merged": merged,
        "summary": summary_sorted,
        "summary_path": merged_path,
        "png_path": png_path,
        "out_dir": out_dir,
    }


def _print_delta(merged: pd.DataFrame):
    """打印消融相对 baseline 的指标变化百分比。"""
    print("\n── 消融 vs. Baseline（MAE 变化%）──")
    for model in merged["model_type"].unique():
        sub = merged[merged["model_type"] == model]
        baseline = sub[sub["ablation_type"] == "baseline"]
        if baseline.empty or "mae_mean" not in baseline.columns:
            continue
        base_mae = baseline["mae_mean"].values[0]
        if base_mae == 0:
            continue
        for _, row in sub.iterrows():
            if row["ablation_type"] == "baseline":
                continue
            abl_mae = row.get("mae_mean", np.nan)
            delta = (abl_mae - base_mae) / base_mae * 100
            arrow = "↑" if delta > 0 else "↓"
            print(f"  {model:10s} | {row['ablation_type']:25s} | "
                  f"MAE: {abl_mae:.4f} ({arrow}{abs(delta):.1f}%)")


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("用法：python run_structural_ablation.py <结构消融配置.yaml>")
        print("\n示例配置见：configs/structural_ablation/")
        sys.exit(1)

    cfg_path = sys.argv[1]
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(ROOT, cfg_path)

    if not os.path.exists(cfg_path):
        print(f"❌ 配置文件不存在：{cfg_path}")
        sys.exit(1)

    cfg = _load_yaml(cfg_path)
    print(f"加载结构消融配置：{cfg_path}")
    run_structural_ablation(cfg)


if __name__ == "__main__":
    main()
