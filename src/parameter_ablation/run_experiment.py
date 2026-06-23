"""
配置驱动的实验入口 run_experiment.py
=====================================
把所有"旋钮"收进一份 YAML 配置，一条命令跑完整条评估路径：

    load_slice(取数) → 选节点(nodes.yaml) → 构造模型 → 滚动回测 → 落盘结果

杜绝硬编码：市场、节点组、频率、上下文、步长、起报点、协变量、模型清单
全部来自配置文件。这正是手册 §6 "消融用配置驱动"的落地。

用法：
    python run_experiment.py configs/parameter_ablation/baseline.yaml
    python run_experiment.py                # 不传参则用内置基准配置(快速冒烟)
"""

from __future__ import annotations

import os
import sys
import json

import pandas as pd

# ── 路径与跨模块 import ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "data_processing"))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))
sys.path.insert(0, os.path.join(ROOT, "src", "evaluation"))

from loader import load_slice                     # noqa: E402
from forecasters import build_forecaster          # noqa: E402
from backtest import run_backtest, BacktestConfig  # noqa: E402

CONFIG_DIR = os.path.join(ROOT, "configs")
NODES_YAML = os.path.join(CONFIG_DIR, "nodes.yaml")
RESULTS_DIR = os.path.join(ROOT, "data", "results")


# ── 极简 YAML 读取（避免强依赖 pyyaml）────────────────────────────────────────
def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return _tiny_yaml(path)


def _tiny_yaml(path: str) -> dict:
    """只支持本项目用到的简单 key: value / 嵌套 / 行内列表 子集。"""
    root = {}
    stack = [(-1, root)]          # (缩进层级, 该层对应的 dict)
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
    return v


def _read_nodes_group(market: str, group: str):
    """从 nodes.yaml 取某市场某组的节点清单。"""
    cfg = _load_yaml(NODES_YAML)
    return cfg[market][group]


# ── 内置基准配置（不传 YAML 时用，用于快速冒烟）──────────────────────────────
DEFAULT_CONFIG = {
    "name": "baseline_smoke",
    "market": "ERCOT",
    "nodes_group": "volatility",
    "freq": "1h",
    "context_len": 168,
    "horizon": 24,
    "data_start": "2025-06-01",
    "data_end": "2025-09-01",
    "backtest": {"test_start": "2025-08-01", "test_end": "2025-08-31",
                 "stride_hours": 24, "max_origins": 30},
    "models": ["Naive", "SeasonalNaive", "ETS", "Theta"],
    "covariates": [],
    "spike": {"quantile": 0.95, "mode": "global"},
}


def run(cfg: dict) -> dict:
    name = cfg["name"]
    market = cfg["market"]
    group = cfg.get("nodes_group")
    nodes = cfg.get("nodes") or _read_nodes_group(market, group)
    freq = cfg.get("freq", "1h")
    covariates = cfg.get("covariates", []) or []

    print("=" * 70)
    print(f"实验：{name}")
    print("=" * 70)
    print(f"市场={market}  节点={nodes}  频率={freq}")
    print(f"上下文={cfg['context_len']}  步长={cfg['horizon']}  协变量={covariates or '无'}")

    # 1) 取数
    data = load_slice(
        market=market, nodes=nodes, freq=freq,
        covariates=covariates,
        start=cfg.get("data_start"), end=cfg.get("data_end"),
    )
    print(f"\n取数完成：{data.shape}  {data.index[0]} → {data.index[-1]}")

    # 2) 构造模型
    forecasters = [build_forecaster(m) for m in cfg["models"]]
    print(f"模型：{[f.name for f in forecasters]}")

    # 3) 回测配置
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
        # 需训练模型（树模型/神经网）专用训练窗口；零样本/基础模型仍用 context_len
        train_context_len=cfg.get("train_context_len") or bt.get("train_context_len"),
        # 多变量旋钮（手册 §6 消融 C）：true=多节点联合建模，false=逐列单变量。
        multivariate=bool(cfg.get("multivariate", False)),
    )

    # 4) 滚动回测
    print("\n开始滚动回测 …")
    result = run_backtest(data, forecasters, bcfg, covariates=covariates)

    summary = result["summary"]
    print("\n── 结果汇总（按 MAE 升序）──")
    show_cols = [c for c in ["model", "covariates_used", "n_origins",
                             "mae_mean", "mae_std", "rmse_mean", "mase_mean",
                             "coverage_mean", "spike_f1_mean_signal",
                             "spike_recall", "spike_f1_q90_signal"]
                 if c in summary.columns]
    summary_sorted = summary.sort_values("mae_mean")
    print(summary_sorted[show_cols].round(4).to_string(index=False))

    # 5) 落盘
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    summary_sorted.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    result["per_origin"].to_csv(os.path.join(out_dir, "per_origin.csv"), index=False)
    # 落盘逐时刻预测记录（用于时序图）
    records_csv = os.path.join(out_dir, "records.csv")
    result["records"].to_csv(records_csv, index=False)
    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump(result["thresholds"], f, indent=2)
    print(f"\n✅ 结果已写入：{out_dir}/")

    # 6) 自动出图
    try:
        from plotting import plot_summary, plot_timeseries
        png1 = plot_summary(summary_sorted,
                            os.path.join(out_dir, "summary_compare.png"),
                            title=f"{name} 模型对比")
        print(f"📊 对比柱状图：{png1}")
        png2 = plot_timeseries(result["records"],
                               os.path.join(out_dir, "timeseries_compare.png"),
                               thresholds=result["thresholds"],
                               title=f"{name} 预测时序对比")
        print(f"📈 时序预测图：{png2}")
    except Exception as e:
        print(f"⚠️  出图失败（不影响结果）：{e}")

    return result


def main():
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(ROOT, cfg_path)
        cfg = _load_yaml(cfg_path)
        print(f"加载配置：{cfg_path}")
    else:
        cfg = DEFAULT_CONFIG
        print("未指定配置文件，使用内置基准配置（冒烟）")
    run(cfg)


if __name__ == "__main__":
    main()
