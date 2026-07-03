"""
compare_sweep.py — sweep 结果汇总对比
========================================
支持两种目录结构：
  新格式（sweep）: data/results/fusion_electfm_lsp01/fusion_electfm_w1_stable/summary.csv
  旧格式（第一轮）: data/results/fusion/fusion_electfm_w1_stable/summary.csv

用法：
    python src/fusion_model/compare_sweep.py
"""
import os, json
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_ROOT = os.path.join(ROOT, "data", "results")

WINDOWS = {
    "fusion_electfm_w1_stable":   "W1",
    "fusion_electfm_w2_negative":  "W2",
    "fusion_electfm_w3_extreme":   "W3",
}
KEY_METRICS = [
    "smape_mean", "mae_mean", "pinball_mean", "coverage_mean",
    "spike_f1_mean_signal", "spike_f1_q90_signal", "spike_f1_spike_head",
]
BASELINE_W1 = {
    "label":       "TimesFM-2.5（零样本）",
    "smape_mean":  27.67, "mae_mean": 12.04, "pinball_mean": 3.999,
    "coverage_mean": 0.775, "spike_f1_mean_signal": 0.3796,
    "spike_f1_q90_signal": 0.4257, "spike_f1_spike_head": None,
}


def _read_summary(top_dir: str, win_key: str):
    """从 top_dir 下读取某窗口的 summary.csv，返回 dict 或 None。"""
    csv = os.path.join(top_dir, win_key, "summary.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    row = df.iloc[0].to_dict()

    tau_f = os.path.join(top_dir, win_key, "tau_star.json")
    if os.path.exists(tau_f):
        with open(tau_f) as f:
            row["tau_star"] = json.load(f).get("tau_star")
    return row


def discover_runs() -> list[dict]:
    """
    发现所有实验目录，返回 [{label, lambda_spike, top_dir}]
    """
    runs = []
    for d in sorted(os.listdir(RESULTS_ROOT)):
        path = os.path.join(RESULTS_ROOT, d)
        if not os.path.isdir(path):
            continue
        if d == "fusion":              # 旧格式第一轮
            runs.append({"label": "第一轮（LR=1e-4，quant_head 可训）",
                          "lambda_spike": 0.2, "top_dir": path})
        elif d.startswith("fusion_electfm_lsp"):
            lsp = d.replace("fusion_electfm_lsp", "")
            try:
                ls = float(f"0.{lsp.lstrip('0') or '0'}")
            except Exception:
                ls = None
            runs.append({"label": d, "lambda_spike": ls, "top_dir": path})
    return runs


def build_table(window_key: str) -> pd.DataFrame:
    """构建某窗口的所有 run 对比 DataFrame。"""
    runs = discover_runs()
    rows = []
    for run in runs:
        r = _read_summary(run["top_dir"], window_key)
        if r is None:
            continue
        row = {"模型/配置": run["label"], "λ_spike": run["lambda_spike"]}
        for m in KEY_METRICS:
            row[m] = round(float(r[m]), 4) if m in r and r[m] is not None and not (isinstance(r[m], float) and np.isnan(r[m])) else None
        row["τ*"] = r.get("tau_star")
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison():
    wins = list(WINDOWS.items())

    for win_key, win_name in wins:
        df = build_table(win_key)

        # W1 加基准行
        if win_key == "fusion_electfm_w1_stable":
            base_row = {"模型/配置": BASELINE_W1["label"], "λ_spike": None}
            for m in KEY_METRICS:
                base_row[m] = BASELINE_W1.get(m)
            base_row["τ*"] = None
            df = pd.concat([pd.DataFrame([base_row]), df], ignore_index=True)

        if df.empty:
            print(f"⚠️  {win_name}：暂无结果（sweep 还在跑中？）")
            continue

        print(f"\n{'='*85}")
        print(f"{win_name} 窗口")
        print(f"{'='*85}")
        print(df.to_string(index=False, float_format="{:.4f}".format, na_rep="—"))

        # W1 改善量
        if win_key == "fusion_electfm_w1_stable":
            print()
            print("  vs TimesFM-2.5 零样本基准：")
            base_smape = BASELINE_W1["smape_mean"]
            base_cov   = BASELINE_W1["coverage_mean"]
            base_f1    = BASELINE_W1["spike_f1_mean_signal"]
            for _, row in df.iterrows():
                if row["模型/配置"] == BASELINE_W1["label"]:
                    continue
                ls    = row["λ_spike"]
                smape = row.get("smape_mean")
                cov   = row.get("coverage_mean")
                f1h   = row.get("spike_f1_spike_head")
                if smape is not None:
                    d_smape = smape - base_smape
                    d_cov   = (cov - base_cov) if cov else None
                    d_f1    = (f1h - base_f1)  if f1h else None
                    cov_str = f"Coverage {d_cov:+.3f}" if d_cov is not None else ""
                    f1_str  = f"SpikeF1/head {d_f1:+.4f}" if d_f1 is not None else ""
                    print(f"    λ={ls}: SMAPE {d_smape:+.2f}  {cov_str}  {f1_str}")


if __name__ == "__main__":
    print_comparison()
