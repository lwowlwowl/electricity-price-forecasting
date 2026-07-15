"""
Chronos2 单协变量扫描 run_chronos2_covariate_scan.py
====================================================
轻量脚本：只跑 Chronos2，固定 [] 基线 + 逐个单独协变量，看每个协变量
对 MAE / Spike-F1 的边际影响。

协变量来源 = 预报版 model_ready（{market}_features_forecast_hourly.csv）：
  - load / temperature / wind / solar  → 日前预报值
  - 经济类（气价/库存/WTI/风暴）       → 保持 shift(24) 滞后（"原来是 lag 就也是 lag"）
即"四个协变量为预报的，其他原来不是就也不是"。可用 --no-forecast 切回纯滞后版对照。

为什么只测 Chronos2：全模型清单里只有它 supports_covariates=True（见 forecasters.py），
其余模型喂不进协变量、结果与 [] 完全相同，逐个跑无意义。

用法：
    python3 src/parameter_ablation/run_chronos2_covariate_scan.py
    python3 src/parameter_ablation/run_chronos2_covariate_scan.py --market PJM
    python3 src/parameter_ablation/run_chronos2_covariate_scan.py \
        --covariates load,temperature,wind,solar
    python3 src/parameter_ablation/run_chronos2_covariate_scan.py --no-forecast   # 纯滞后对照

精简代表集（默认，按市场自动过滤不存在的列）：
  ERCOT: load, temperature, wind, solar, henry_hub, wti, storm_event_count,
         gas_share, renewable_shock          （9 个）
  PJM  : load, temperature, henry_hub, wti, hrc_futures(钢价), storm_event_count,
         gas_share, renewable_shock          （8 个，含钢价）
两个市场分别跑：
  python3 src/parameter_ablation/run_chronos2_covariate_scan.py --market ERCOT
  python3 src/parameter_ablation/run_chronos2_covariate_scan.py --market PJM
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "data_processing"))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))
sys.path.insert(0, os.path.join(ROOT, "src", "evaluation"))

from loader import load_slice_model_ready                 # noqa: E402
from forecasters import build_forecaster                  # noqa: E402
from backtest import run_backtest, BacktestConfig         # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "data", "results", "parameter_ablation")
MODEL_READY_DIR = os.path.join(ROOT, "data", "covariates", "model_ready")

# 精简代表集：每家族选最干净、最不冗余的 1-3 个代表，逐个单独测边际影响。
# 实际扫描时按市场过滤掉不存在的列（如 PJM 无 wind/solar、仅 PJM 有钢价）。
#
#   预报版(日前预报) : load, temperature, wind, solar
#   经济/燃料(滞后) : henry_hub(气价, 边际燃料), wti(油价), hrc_futures(钢价, 仅PJM)
#   风暴(滞后)      : storm_event_count（其余6列同源描述同一批事件，取最干净代理）
#   发电结构(滞后)  : gas_share(天然气=边际机组→定价), renewable_shock(可再生骤变→尖峰)
#                     （gas_share_diff/renewable_share/各 *_gen_mwh 与之共线，不重复测）
CURATED_COVARIATES = [
    "load", "temperature", "wind", "solar",
    "henry_hub_usd_per_mmbtu", "wti_usd_per_barrel", "hrc_futures_usd_per_ton",
    "storm_event_count",
    "gas_share", "renewable_shock",
]


def _available_covariates(market: str, forecast: bool) -> set:
    """读 model_ready 文件表头，返回该市场实际可用的协变量列名集合。"""
    fname = (f"{market.lower()}_features_forecast_hourly.csv"
             if forecast else f"{market.lower()}_features_hourly.csv")
    path = os.path.join(MODEL_READY_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}（先跑 scripts/covariates/15_build_forecast_model_ready.py）")
    with open(path) as f:
        header = f.readline().strip().split(",")
    return set(header) - {"timestamp_utc"}


def run_one(forecaster, covariates, *, market, nodes_group, freq, forecast,
            data_start, data_end, context_len, horizon, test_start, test_end,
            stride, spike_quantile, spike_mode, max_origins):
    """跑单个协变量组合，返回一行汇总 dict（Chronos2 only）。"""
    data = load_slice_model_ready(
        market=market,
        nodes=_resolve_nodes(market, nodes_group),
        freq=freq, covariates=covariates, start=data_start, end=data_end,
        forecast=forecast,
    )
    print(f"\n取数：{data.shape}  协变量={covariates or '无'}  "
          f"来源={'预报版' if forecast else '滞后版'}")

    bcfg = BacktestConfig(
        context_len=context_len,
        horizon=horizon,
        stride=stride,
        test_start=test_start,
        test_end=test_end,
        spike_quantile=spike_quantile,
        spike_mode=spike_mode,
        max_origins=max_origins,
        multivariate=False,
    )
    result = run_backtest(data, [forecaster], bcfg, covariates=covariates)
    s = result["summary"].iloc[0]
    return {
        "covariates": ",".join(covariates) if covariates else "[]",
        "n_origins": int(s.get("n_origins", 0)),
        "rmae_mean": float(s.get("rmae_mean", float("nan"))),
        "mae_mean": float(s.get("mae_mean", float("nan"))),
        "rmse_mean": float(s.get("rmse_mean", float("nan"))),
        "smape_mean": float(s.get("smape_mean", float("nan"))),
        "spike_f1_mean": float(s.get("spike_f1_mean_signal", float("nan"))),
        "spike_recall": float(s.get("spike_recall", float("nan"))),
        "spike_f1_q90": float(s.get("spike_f1_q90_signal", float("nan"))),
        "coverage_mean": float(s.get("coverage_mean", float("nan"))),
    }


def _resolve_nodes(market, nodes_group):
    """复用 run_experiment 的 nodes.yaml 读取逻辑（轻量内联，避免循环 import）。"""
    import yaml
    nodes_yaml = os.path.join(ROOT, "configs", "nodes.yaml")
    with open(nodes_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    market_cfg = cfg.get(market, {})
    if nodes_group == "all":
        return list(market_cfg.get("all", []))
    abl = market_cfg.get("ablation", {})
    if nodes_group in abl:
        e = abl[nodes_group]
        return e if isinstance(e, list) else [e]
    if nodes_group in market_cfg:
        e = market_cfg[nodes_group]
        return e if isinstance(e, list) else [e]
    raise KeyError(f"节点组 '{nodes_group}' 不在 nodes.yaml[{market}]")


def main():
    p = argparse.ArgumentParser(
        description="Chronos2 单协变量扫描（MAE / Spike-F1 边际影响）",
        add_help=False,
    )
    p.add_argument("--market", default="ERCOT")
    p.add_argument("--nodes_group", default="ablation",
                   help="nodes.yaml 里的节点组（ablation=3 代表节点 / all=全节点）")
    p.add_argument("--freq", default="1h")
    p.add_argument("--context_len", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--data_start", default="2025-01-01")
    p.add_argument("--data_end", default="2026-06-05")
    p.add_argument("--test_start", default="2025-07-01")
    p.add_argument("--test_end", default="2026-06-01")
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--spike_quantile", type=float, default=0.95)
    p.add_argument("--spike_mode", default="global")
    p.add_argument("--max_origins", type=int, default=None,
                   help="限制起报点数（调试用）")
    p.add_argument("--covariates", default=None,
                   help="逗号分隔；逐个单独测试 + [] 基线。默认用内置 8 个")
    p.add_argument("--no-forecast", action="store_true",
                   help="用纯滞后版 model_ready 对照（默认预报版）")
    p.add_argument("--out_name", default="chronos2_covariate_scan")
    p.add_argument("-h", "--help", action="store_true")
    args, _ = p.parse_known_args()

    if args.help:
        p.print_help()
        return

    if args.covariates:
        cov_list = [c.strip() for c in args.covariates.split(",") if c.strip()]
    else:
        cov_list = list(CURATED_COVARIATES)

    forecast = not args.no_forecast

    # 按市场过滤掉不存在的列（PJM 无 wind/solar；仅 PJM 有 hrc_futures）
    available = _available_covariates(args.market, forecast)
    dropped = [c for c in cov_list if c not in available]
    cov_list = [c for c in cov_list if c in available]
    if dropped:
        print(f"  ℹ️  {args.market} 无以下列，已跳过：{dropped}")

    print("=" * 70)
    print(f"Chronos2 单协变量扫描　市场={args.market}　节点组={args.nodes_group}")
    print(f"协变量来源：{'预报版（4 预报 + 经济/风暴/发电滞后）' if forecast else '纯滞后版'}")
    print(f"扫描：[] + 逐个 {len(cov_list)} 个 → {cov_list}")
    print("=" * 70)

    # Chronos2 零样本，构造一次复用（worker 只加载一次模型，cache 按任务区分）
    fc = build_forecaster("Chronos2", period=24)

    rows = []
    # 1) 基线 []
    rows.append(run_one(
        fc, [], market=args.market, nodes_group=args.nodes_group, freq=args.freq,
        forecast=forecast, data_start=args.data_start, data_end=args.data_end,
        context_len=args.context_len, horizon=args.horizon,
        test_start=args.test_start, test_end=args.test_end, stride=args.stride,
        spike_quantile=args.spike_quantile, spike_mode=args.spike_mode,
        max_origins=args.max_origins,
    ))
    # 2) 逐个单独协变量
    for cov in cov_list:
        rows.append(run_one(
            fc, [cov], market=args.market, nodes_group=args.nodes_group,
            freq=args.freq, forecast=forecast, data_start=args.data_start,
            data_end=args.data_end, context_len=args.context_len,
            horizon=args.horizon, test_start=args.test_start,
            test_end=args.test_end, stride=args.stride,
            spike_quantile=args.spike_quantile, spike_mode=args.spike_mode,
            max_origins=args.max_origins,
        ))

    df = pd.DataFrame(rows)

    # 相对基线的边际变化（Δ）
    base = df.iloc[0]
    df["Δmae_vs_base"] = (df["mae_mean"] - base["mae_mean"]).round(4)
    df["Δspike_f1_vs_base"] = (df["spike_f1_mean"] - base["spike_f1_mean"]).round(4)

    print("\n" + "=" * 70)
    print("── Chronos2 单协变量扫描结果（按 MAE 升序）──")
    print("=" * 70)
    show = ["covariates", "n_origins", "rmae_mean", "mae_mean", "Δmae_vs_base",
            "spike_f1_mean", "Δspike_f1_vs_base", "spike_recall", "spike_f1_q90"]
    print(df[show].round(4).to_string(index=False))

    out_dir = os.path.join(RESULTS_DIR, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    tag = "forecast" if forecast else "lag"
    out_csv = os.path.join(out_dir, f"scan_{args.market.lower()}_{tag}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ 结果已写入：{out_csv}")


if __name__ == "__main__":
    main()
