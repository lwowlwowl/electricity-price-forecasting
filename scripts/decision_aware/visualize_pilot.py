#!/usr/bin/env python
"""visualize_pilot.py — 先行版可视化。

出三张图到 data/results/da_tsfm_pilot/：
  1. train_curve.png   — 训练曲线（pred/regret/val MAE + α-β 退火），从日志解析
  2. forecast_vs_actual.png — β=1 模型在若干 test 窗口的预测 vs 真值
  3. bess_schedule.png — 一个 test 窗口的电价 + 模型充放电动作 + SOC（decision-aware 的"钱图"）

用法: external/chronos-forecasting/.venv/bin/python scripts/decision_aware/visualize_pilot.py
       [--log logs/da_pilot_fixed20.log] [--ckpt last] [--n-windows 4]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
os.chdir(_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# macOS 中文字体（避免中文标签显示成方框）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch

from decision_aware.config import PilotConfig, STREAM_COLS
from decision_aware.dataset import build_datasets
from decision_aware.model import DecisionAwareTSFM
from decision_aware.policy import BESSSimulator, STEPolicy, oracle_revenue
from decision_aware.train import get_device

OUT_DIR = "data/results/da_tsfm_pilot"

# 配色（与 drawio 一致：蓝=price/pred，绿=BESS/收益，橙=head，紫=oracle）
C_PRICE, C_PRED, C_BESS, C_SOC, C_ORACLE = "#0066CC", "#FF9900", "#689F38", "#7E57C2", "#B0BEC5"


def parse_log(path: str):
    """解析训练日志，返回 dict of lists。"""
    pat = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+tr loss=([-\d.]+) pred=([-\d.]+) bus=([-\d.]+) "
        r"regret=([-\d.]+) mae=([-\d.]+)\s+\|\s+val loss=([-\d.]+) pred=([-\d.]+) "
        r"bus=([-\d.]+) regret=([-\d.]+) mae=([-\d.]+)\s+\|\s+α=([-\d.]+) β=([-\d.]+)")
    d = {k: [] for k in
         ("epoch", "tr_pred", "tr_reg", "tr_mae", "val_pred", "val_reg", "val_mae", "alpha", "beta")}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            g = m.groups()
            d["epoch"].append(int(g[0]))
            for key, idx in (("tr_pred", 2), ("tr_reg", 4), ("tr_mae", 5),
                             ("val_pred", 7), ("val_reg", 9), ("val_mae", 10),
                             ("alpha", 11), ("beta", 12)):
                v = float(g[idx])
                d[key].append(v)
    return d


def plot_train_curve(log_path: str, out: str):
    d = parse_log(log_path)
    if not d["epoch"]:
        print(f"  ⚠ 日志 {log_path} 解析为空，跳过训练曲线"); return
    ep = np.array(d["epoch"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot(ep, d["tr_pred"], ".-", color=C_PRICE, label="train pred(Huber)")
    ax.plot(ep, d["val_pred"], ".-", color=C_PRED, label="val pred")
    ax.set_title("L_pred (Huber, $尺度)"); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
    ax = axes[1]
    ax.plot(ep, d["tr_reg"], ".-", color=C_PRICE, label="train regret")
    ax.plot(ep, d["val_reg"], ".-", color=C_PRED, label="val regret")
    ax.axvline(np.argmax([b >= 1.0 for b in d["beta"]]) if any(b >= 1 for b in d["beta"]) else 0,
               color="gray", ls="--", alpha=.5, label="β=1 起")
    ax.set_title("Regret (业务损失, $)"); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
    ax = axes[2]
    ax.plot(ep, d["val_mae"], ".-", color=C_PRED, label="val MAE")
    ax.set_title("val MAE ($/MWh)"); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
    ax2 = ax.twinx()
    ax2.plot(ep, d["beta"], ":", color=C_BESS, alpha=.7, label="β")
    ax2.plot(ep, d["alpha"], ":", color=C_SOC, alpha=.7, label="α")
    ax2.set_ylabel("α / β"); ax2.legend(loc="center right")
    fig.suptitle(f"先行版训练曲线（{os.path.basename(log_path)}）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


@torch.no_grad()
def collect_predictions(cfg, test_ds, ckpt_tag, device, n_windows=4):
    model = DecisionAwareTSFM(cfg).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint_path(ckpt_tag),
                                     map_location=device, weights_only=True))
    model.eval()
    sim = BESSSimulator(cfg.bess_power_mw, cfg.bess_energy_mwh, cfg.bess_eta, cfg.bess_init_soc_frac)
    pol = STEPolicy(cfg.ste_k)
    # 均匀挑 n_windows 个 test 起报点
    idxs = np.linspace(0, len(test_ds) - 1, n_windows).astype(int)
    samples = []
    for i in idxs:
        s = test_ds[i]
        batch = {k: v.unsqueeze(0).to(device) for k, v in s.items()}
        out = model(batch)
        p_da = out["p_da"][0].cpu().numpy()
        actual = s["price_tgt"].numpy()
        u = pol(out["p_da"])[0].cpu().numpy()
        R = sim(out["p_da"], s["price_tgt"].to(device).unsqueeze(0)).cpu().numpy()[0]
        Rs = oracle_revenue(s["price_tgt"].to(device).unsqueeze(0), sim).cpu().numpy()[0]
        ts = test_ds.index[i + cfg.context_len: i + cfg.context_len + cfg.horizon_da]
        samples.append(dict(ts=ts, p_da=p_da, actual=actual, u=u, R=float(R), Rs=float(Rs)))
    return samples


def plot_forecast_vs_actual(samples, out):
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, samples):
        h = np.arange(len(s["actual"]))
        ax.plot(h, s["actual"], ".-", color=C_PRICE, lw=2, label="真值(actual)")
        ax.plot(h, s["p_da"], ".-", color=C_PRED, lw=1.6, label="预测(p̂_DA, β=1)")
        ax.set_title(f"R={s['R']:.0f} / R*={s['Rs']:.0f}  ({s['R']/max(s['Rs'],1e-9)*100:.0f}%)")
        ax.set_xlabel("hour ahead"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("price ($/MWh)")
    fig.suptitle("先行版 β=1 模型：预测 vs 真值（4 个 test 窗口）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_bess_schedule(s, out, cfg):
    """单窗口：电价(线) + 充放电动作(柱) + SOC(线) + 充/放阴影。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    h = np.arange(len(s["actual"]))
    # 动作柱：充电负(绿向下)、放电正(橙向上)
    dis = np.clip(s["u"], 0, 1)
    chg = np.clip(-s["u"], 0, 1)
    ax.bar(h, s["actual"], width=0.7, alpha=0.18, color=C_PRICE, label="电价(actual)")
    ax2 = ax.twinx()
    ax2.bar(h, -chg, width=0.5, color=C_BESS, alpha=0.6, label="充电(charge)")
    ax2.bar(h, dis, width=0.5, color=C_PRED, alpha=0.8, label="放电(discharge)")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("动作 u (充↑负 / 放↓正)", color=C_BESS)
    ax2.set_ylim(-1.3, 1.3)
    ax.set_ylabel("price ($/MWh)")
    ax.set_xlabel("hour ahead"); ax.set_title(
        f"BESS 决策（β=1 模型）  R={s['R']:.0f}  Oracle R*={s['Rs']:.0f}  "
        f"占比 {s['R']/max(s['Rs'],1e-9)*100:.0f}%")
    # 标注谷/峰
    pmin, pmax = int(np.argmin(s["actual"])), int(np.argmax(s["actual"]))
    ax.annotate("谷", (pmin, s["actual"][pmin]), color=C_BESS, fontsize=12, ha="center", va="bottom")
    ax.annotate("峰", (pmax, s["actual"][pmax]), color=C_PRED, fontsize=12, ha="center", va="bottom")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decision_aware/pilot_ercot.yaml")
    ap.add_argument("--log", default="logs/da_pilot_fixed20.log")
    ap.add_argument("--ckpt", default="last", help="best 或 last（last=β=1）")
    ap.add_argument("--n-windows", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = PilotConfig.from_yaml(args.config)

    print("=" * 60); print("先行版可视化"); print("=" * 60)
    plot_train_curve(args.log, os.path.join(OUT_DIR, "train_curve.png"))

    device = get_device()
    _, _, test_ds, _ = build_datasets(cfg)
    print(f"  加载 ckpt: {cfg.checkpoint_path(args.ckpt)} (tag={args.ckpt})")
    samples = collect_predictions(cfg, test_ds, args.ckpt, device, args.n_windows)
    plot_forecast_vs_actual(samples, os.path.join(OUT_DIR, "forecast_vs_actual.png"))
    plot_bess_schedule(samples[len(samples) // 2], os.path.join(OUT_DIR, "bess_schedule.png"), cfg)

    # 汇总结论
    R = np.mean([s["R"] for s in samples])
    Rs = np.mean([s["Rs"] for s in samples])
    print(f"\n  [{args.ckpt}] 可视化样本均值: R={R:.1f}  R*={Rs:.1f}  占比={R/max(Rs,1e-9)*100:.0f}%")
    print(f"\n✅ 图已存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
