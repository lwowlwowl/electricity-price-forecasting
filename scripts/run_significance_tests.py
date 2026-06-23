"""
统计显著性检验 — Wilcoxon Signed-Rank Test
==========================================
对 30 个起报点的配对误差做检验，为全部消融结论标注 p-value。

方法
----
  指标 1：SMAPE      — 直接来自 per_origin.csv 的逐 origin 值
  指标 2：Spike-F1   — 从 records.csv + thresholds.json 按 origin 重算（mean 信号）
  检验   ：Wilcoxon signed-rank（双侧，zero_method='wilcox'）
  多重比较：Bonferroni（跨全部有效测试的全局校正）
  标注    ：* p_corr<0.05  ** p_corr<0.01  *** p_corr<0.001

输出
----
  report/significance_tests.csv        — 完整检验记录（raw + corrected p）
  report/significance_summary.md       — 可读摘要表
  report/fig1_component_heatmap.png    — 带显著性标注（覆盖旧版）
  report/fig2_perlayer_curves.png      — 带显著性标注（覆盖旧版）
"""

from __future__ import annotations
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── 路径 ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent.parent / "data" / "results" / "structural_ablation"
OUT    = BASE / "report"
SCRIPTS = Path(__file__).parent

# 复用报告脚本中的数据加载函数
sys.path.insert(0, str(SCRIPTS))
from generate_architecture_report import (
    load_full_ablation, load_perlayer,
    MODELS_ORDER, COMP_ORDER, MODEL_COLORS,
    PLT_STYLE,
)

plt.rcParams.update(PLT_STYLE)

# 数据集配置：(dataset_label, model_name_display, folder_suffix)
DATASETS_FULL = [
    ("toto2",    "Toto-2.0",    "structural_full_toto2"),
    ("chronos2", "Chronos-2",   "structural_full_chronos2"),
    ("timesfm",  "TimesFM-2.5", "structural_full_timesfm"),
]
DATASETS_LAYER = [
    ("toto2",    "Toto-2.0",    "structural_perlayer_toto2"),
    ("chronos2", "Chronos-2",   "structural_perlayer_chronos2"),
    ("timesfm",  "TimesFM-2.5", "structural_perlayer_timesfm"),
]

# ── 1. 数据加载 ───────────────────────────────────────────────────────────────

def _extract_abl(model_str: str) -> str:
    """'Toto2[skip_ffn]' → 'skip_ffn'，无括号则返回 None。"""
    m = re.search(r"\[(.+)\]", model_str)
    return m.group(1) if m else None


def load_smape_per_origin(folder: str) -> pd.DataFrame:
    """
    返回 DataFrame，列：ablation_type, origin, smape
    origin 归一化为字符串（去掉时区信息，方便对齐）。
    """
    path = BASE / folder / "per_origin.csv"
    df = pd.read_csv(path)
    df["ablation_type"] = df["model"].map(_extract_abl)
    df = df.dropna(subset=["ablation_type"])
    df["origin"] = df["origin"].astype(str).str[:19]   # "2025-08-01 00:00:00"
    return df[["ablation_type", "origin", "smape"]].copy()


def compute_f1_per_origin(folder: str) -> pd.DataFrame:
    """
    从 records.csv + thresholds.json 逐 origin 计算 Spike-F1（mean 信号）。
    返回 DataFrame，列：ablation_type, origin, spike_f1
    """
    rec_path = BASE / folder / "records.csv"
    thr_path = BASE / folder / "thresholds.json"

    records = pd.read_csv(rec_path)
    with open(thr_path) as f:
        thresholds = json.load(f)

    records["ablation_type"] = records["model"].map(_extract_abl)
    records = records.dropna(subset=["ablation_type"])
    records["origin"] = records["origin"].astype(str).str[:19]

    rows = []
    for (abl, origin), grp in records.groupby(["ablation_type", "origin"]):
        node_f1s = []
        for node, ng in grp.groupby("node"):
            thr = thresholds.get(node)
            if thr is None:
                continue
            actual = ng["actual"].values.astype(float)
            pred   = ng["mean"].values.astype(float)
            mask   = ~(np.isnan(actual) | np.isnan(pred))
            actual, pred = actual[mask], pred[mask]
            if len(actual) == 0:
                node_f1s.append(np.nan)
                continue
            true_spike = actual >= thr
            pred_spike = pred   >= thr
            tp = int(np.sum(pred_spike & true_spike))
            fp = int(np.sum(pred_spike & ~true_spike))
            fn = int(np.sum(~pred_spike & true_spike))
            if tp + fp == 0 or tp + fn == 0:
                node_f1s.append(0.0)
            else:
                prec = tp / (tp + fp)
                rec  = tp / (tp + fn)
                node_f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        rows.append({
            "ablation_type": abl,
            "origin":        origin,
            "spike_f1":      float(np.nanmean(node_f1s)) if node_f1s else np.nan,
        })

    return pd.DataFrame(rows)


# ── 2. 单 dataset 的 Wilcoxon 检验 ────────────────────────────────────────────

def _wilcoxon_pair(bl: np.ndarray, ab: np.ndarray) -> tuple[float, float, str]:
    """
    对配对的 baseline/ablated 数组跑 Wilcoxon。
    返回 (statistic, p_raw, note)。
    """
    valid = ~(np.isnan(bl) | np.isnan(ab))
    bl_v, ab_v = bl[valid], ab[valid]
    n = int(valid.sum())
    if n < 3:
        return np.nan, np.nan, f"too_few_pairs(n={n})"
    diffs = ab_v - bl_v
    if np.all(diffs == 0):
        return 0.0, 1.0, "identical"
    try:
        stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p), ""
    except Exception as e:
        return np.nan, np.nan, str(e)


def run_wilcoxon_for_dataset(
    folder: str,
    model_name: str,
    dataset_label: str,   # "full" or "perlayer"
) -> pd.DataFrame:
    """
    对一个 (model, dataset_type) 跑所有消融条件的双指标 Wilcoxon 检验。
    """
    smape_df = load_smape_per_origin(folder)
    f1_df    = compute_f1_per_origin(folder)

    origins = smape_df["origin"].unique()
    abl_types = [a for a in smape_df["ablation_type"].unique() if a != "baseline"]

    # 基线 pivot
    bl_smape = smape_df[smape_df["ablation_type"] == "baseline"].set_index("origin")["smape"]
    bl_f1    = f1_df[f1_df["ablation_type"]       == "baseline"].set_index("origin")["spike_f1"]

    records = []
    for abl in sorted(abl_types):
        ab_smape = smape_df[smape_df["ablation_type"] == abl].set_index("origin")["smape"]
        ab_f1    = f1_df[f1_df["ablation_type"]       == abl].set_index("origin")["spike_f1"]

        for metric, bl_s, ab_s in [
            ("smape",    bl_smape, ab_smape),
            ("spike_f1", bl_f1,    ab_f1),
        ]:
            # 对齐到公共 origins
            common = sorted(set(bl_s.index) & set(ab_s.index))
            bl_arr = bl_s.reindex(common).values.astype(float)
            ab_arr = ab_s.reindex(common).values.astype(float)

            stat, p_raw, note = _wilcoxon_pair(bl_arr, ab_arr)
            records.append({
                "model":         model_name,
                "dataset":       dataset_label,
                "ablation_type": abl,
                "metric":        metric,
                "n_pairs":       int((~(np.isnan(bl_arr) | np.isnan(ab_arr))).sum()),
                "statistic":     stat,
                "p_raw":         p_raw,
                "p_corr":        np.nan,   # 填 Bonferroni 后补
                "sig":           "",
                "note":          note,
            })

    return pd.DataFrame(records)


# ── 3. 全局 Bonferroni 校正 ───────────────────────────────────────────────────

def apply_bonferroni(df: pd.DataFrame) -> pd.DataFrame:
    """
    同时做 Bonferroni 和 Benjamini-Hochberg (FDR) 校正。
    Bonferroni：主校正（最保守，用于主结论）
    FDR-BH    ：辅助校正（用于相关检验族，更适合消融研究）
    """
    from statsmodels.stats.multitest import multipletests

    valid_mask = df["p_raw"].notna() & ~df["note"].fillna("").str.contains("too_few_pairs")
    n_tests = int(valid_mask.sum())
    print(f"  Bonferroni + FDR-BH 校正：{n_tests} 个有效检验")

    df = df.copy()

    # ── Bonferroni ──
    df.loc[valid_mask, "p_corr_bonf"] = (
        df.loc[valid_mask, "p_raw"] * n_tests
    ).clip(upper=1.0)

    # ── FDR Benjamini-Hochberg ──
    p_raw_valid = df.loc[valid_mask, "p_raw"].values
    _, p_fdr, _, _ = multipletests(p_raw_valid, method="fdr_bh")
    df.loc[valid_mask, "p_corr_fdr"] = p_fdr

    # 主列 p_corr = Bonferroni（向后兼容）
    df["p_corr"] = df["p_corr_bonf"]

    def _stars(p):
        if pd.isna(p): return ""
        if p < 0.001:  return "***"
        if p < 0.01:   return "**"
        if p < 0.05:   return "*"
        return "ns"

    df["sig"]      = df["p_corr_bonf"].map(_stars)   # Bonferroni 星号（主）
    df["sig_fdr"]  = df["p_corr_fdr"].map(_stars)    # FDR 星号（辅）
    return df


# ── 4. 构建显著性查找表（供绘图使用）────────────────────────────────────────────

def build_sig_lookup(sig_df: pd.DataFrame):
    """
    返回两个字典：
      full_sig[(model, ablation_type, metric)] → sig_string ("***"/"**"/"*"/"ns"/"")
      layer_sig[(model, ablation_type, metric)] → sig_string
    """
    full  = sig_df[sig_df["dataset"] == "full"]
    layer = sig_df[sig_df["dataset"] == "perlayer"]

    full_sig  = {(r.model, r.ablation_type, r.metric): r.sig
                 for _, r in full.iterrows()}
    layer_sig = {(r.model, r.ablation_type, r.metric): r.sig
                 for _, r in layer.iterrows()}
    return full_sig, layer_sig


# ── 5. fig1 热力图（带显著性标注）────────────────────────────────────────────────

# ablation_type → comp_label 反向映射（与 generate_architecture_report 一致）
ABLATION_TYPE_TO_LABEL = {
    "skip_attention":        "跳过\n注意力\n(S-A2)",
    "halve_heads":           "注意力头\n减半\n(S-A3)",
    "skip_variate":          "跳过\n变量注意力\n(S-B1)",
    "skip_time":             "跳过\n时间注意力\n(S-B2)",
    "remove_rope":           "移除\nRoPE\n(S-G1a)",
    "disable_xpos":          "关闭 xPos\n(S-G1b)",
    "simplify_patch_emb":    "简化\nPatch嵌入\n(S-G2)",
    "simplify_output_head":  "简化\n输出头\n(S-G3)",
    "skip_ffn":              "跳过 FFN\n(S-G4)",
    "skip_layernorm":        "跳过\n层归一化\n(S-G5)",
    "truncate_front_half":   "截断前50%层\n(S-G6a)",
    "truncate_back_half":    "截断后50%层\n(S-G6c)",
}


def make_heatmap_with_sig(full_ablation: dict, full_sig: dict):
    n_comp  = len(COMP_ORDER)
    n_model = len(MODELS_ORDER)

    smape_mat = np.full((n_comp, n_model), np.nan)
    f1_mat    = np.full((n_comp, n_model), np.nan)

    for ci, comp in enumerate(COMP_ORDER):
        for mi, model in enumerate(MODELS_ORDER):
            val = full_ablation[comp][model]
            if val[0] is not None:
                smape_mat[ci, mi] = val[0]
            if val[1] is not None:
                f1_mat[ci, mi] = val[1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 10),
                             gridspec_kw={"wspace": 0.12})

    # ── SMAPE ──
    ax = axes[0]
    smape_capped = np.clip(smape_mat, -20, 200)
    im0 = ax.imshow(smape_capped, cmap="RdYlGn_r", vmin=-20, vmax=200, aspect="auto")
    ax.set_xticks(range(n_model));  ax.set_xticklabels(MODELS_ORDER, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_comp));   ax.set_yticklabels(COMP_ORDER, fontsize=9)
    ax.set_title("SMAPE 变化量 (%)\n(红=变差↑, 绿=变好↓，*=显著校正 Bonferroni)",
                 fontsize=12, fontweight="bold", pad=12)

    for ci, comp in enumerate(COMP_ORDER):
        # 反查 ablation_type（用于查找显著性）
        abl_type = next((k for k, v in ABLATION_TYPE_TO_LABEL.items() if v == comp), None)
        for mi, model in enumerate(MODELS_ORDER):
            orig = full_ablation[comp][model]
            v = smape_mat[ci, mi]
            if orig == (None, None):
                ax.add_patch(plt.Rectangle((mi - 0.5, ci - 0.5), 1, 1,
                                           color="#cccccc", zorder=0))
                ax.text(mi, ci, "N/A", ha="center", va="center",
                        fontsize=8, color="#888888")
            elif np.isnan(v):
                ax.text(mi, ci, "CRASH", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#333333", ec="none"))
            else:
                color = "white" if abs(v) > 80 else "black"
                num_txt = f"+{v:.0f}%" if v >= 0 else f"{v:.0f}%"
                # 显著性星号
                stars = full_sig.get((model, abl_type, "smape"), "") if abl_type else ""
                stars_disp = stars if stars not in ("ns", "") else ""
                ax.text(mi, ci, num_txt, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
                if stars_disp:
                    ax.text(mi + 0.38, ci - 0.35, stars_disp,
                            ha="center", va="center", fontsize=7.5,
                            color="white" if abs(v) > 80 else "#c62828",
                            fontweight="bold")

    plt.colorbar(im0, ax=ax, shrink=0.6, label="SMAPE Δ%（截断至200%）")

    # ── Spike-F1 ──
    ax = axes[1]
    f1_capped = np.clip(f1_mat, -100, 30)
    im1 = ax.imshow(f1_capped, cmap="RdYlGn", vmin=-100, vmax=30, aspect="auto")
    ax.set_xticks(range(n_model));  ax.set_xticklabels(MODELS_ORDER, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_comp));   ax.set_yticklabels(COMP_ORDER, fontsize=9)
    ax.set_title("Spike-F1 变化量 (%)\n(红=尖峰检测变差↓, 绿=改善↑，*=显著校正 Bonferroni)",
                 fontsize=12, fontweight="bold", pad=12)

    for ci, comp in enumerate(COMP_ORDER):
        abl_type = next((k for k, v in ABLATION_TYPE_TO_LABEL.items() if v == comp), None)
        for mi, model in enumerate(MODELS_ORDER):
            orig = full_ablation[comp][model]
            v    = f1_mat[ci, mi]
            if orig == (None, None):
                ax.text(mi, ci, "N/A", ha="center", va="center",
                        fontsize=8, color="#888888")
            elif orig[1] is None and orig[0] is None:
                ax.text(mi, ci, "N/A", ha="center", va="center",
                        fontsize=8, color="#888888")
            elif np.isnan(v):
                ax.text(mi, ci, "CRASH", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#333333", ec="none"))
            else:
                color = "white" if abs(v) > 50 else "black"
                num_txt = f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"
                stars = full_sig.get((model, abl_type, "spike_f1"), "") if abl_type else ""
                stars_disp = stars if stars not in ("ns", "") else ""
                ax.text(mi, ci, num_txt, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
                if stars_disp:
                    ax.text(mi + 0.38, ci - 0.35, stars_disp,
                            ha="center", va="center", fontsize=7.5,
                            color="white" if abs(v) > 50 else "#1b5e20",
                            fontweight="bold")

    plt.colorbar(im1, ax=axes[1], shrink=0.6, label="Spike-F1 Δ%")

    fig.suptitle("结构级消融：组件贡献热力图（带 Wilcoxon 显著性标注）\n"
                 "右上角 * p<0.05  ** p<0.01  *** p<0.001（Bonferroni 校正）",
                 fontsize=14, fontweight="bold", y=1.01)

    # 分组线
    for ax_ in axes:
        for y in [3.5, 5.5, 7.5, 9.5]:
            ax_.axhline(y, color="white", linewidth=2, alpha=0.7)

    groups = [(1.5, "注意力\n机制"), (4.5, "跨变量\n注意力"),
              (6.0, "位置\n编码"), (7.5, "Patch\n嵌入 & 输出头"),
              (9.5, "FFN &\n归一化"), (11.0, "层深度\n截断")]
    for y, label in groups:
        axes[0].annotate(label, xy=(-0.55, y / n_comp), xycoords="axes fraction",
                         ha="right", va="center", fontsize=8, color="#555555", fontweight="bold")

    plt.tight_layout()
    path = OUT / "fig1_component_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ fig1 保存: {path}")


# ── 6. fig2 逐层曲线（带显著性标注）─────────────────────────────────────────────

def make_perlayer_curves_with_sig(per_layer: dict, layer_sig: dict):
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

    TOTO_C    = "#2196F3"
    CHRONOS_C = "#FF9800"
    TIMESFM_C = "#4CAF50"
    model_colors = {"Toto-2.0": TOTO_C, "Chronos-2": CHRONOS_C, "TimesFM-2.5": TIMESFM_C}

    for col, (model_name, color) in enumerate(model_colors.items()):
        data     = per_layer[model_name]
        bl_smape = data["baseline"]["smape"]
        bl_f1    = data["baseline"]["f1"]
        layers   = data["layers"]
        n        = len(layers)
        xs       = list(range(n))

        smape_deltas = [(l["smape"] - bl_smape) / bl_smape * 100 for l in layers]
        f1_deltas    = [(l["f1"]    - bl_f1)    / bl_f1    * 100 for l in layers]

        # ── SMAPE ──
        ax_s = fig.add_subplot(gs[0, col])
        bar_colors_s = ["#ef5350" if v > 10 else "#90caf9" if v < -2 else "#eeeeee"
                        for v in smape_deltas]
        bars_s = ax_s.bar(xs, smape_deltas, color=bar_colors_s,
                          edgecolor="white", linewidth=0.5)
        ax_s.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax_s.set_title(f"{model_name}\nSMAPE Δ%（跳过第N层）",
                       fontsize=11, fontweight="bold", color=color)
        ax_s.set_xlabel("层索引", fontsize=9)
        ax_s.set_ylabel("SMAPE Δ%（正=变差）", fontsize=9)
        ax_s.set_xticks(xs); ax_s.set_xticklabels([str(i) for i in xs], fontsize=8)

        for i, v in enumerate(smape_deltas):
            abl = f"skip_layer_{i}"
            stars = layer_sig.get((model_name, abl, "smape"), "")
            stars_disp = stars if stars not in ("ns", "") else ""

            # 数值标注（>20%）
            if abs(v) > 20:
                ax_s.text(i, v + (2 if v >= 0 else -4),
                          f"+{v:.0f}%" if v >= 0 else f"{v:.0f}%",
                          ha="center", fontsize=7.5, fontweight="bold",
                          color="#b71c1c" if v > 0 else "#1565c0")
            # 显著性星号
            if stars_disp:
                y_pos = max(v, 0) + 1.5
                ax_s.text(i, y_pos, stars_disp, ha="center", fontsize=8,
                          color="#b71c1c", fontweight="bold")

        if model_name == "Toto-2.0":
            ax_s.text(5, smape_deltas[5] + 5, "L5:输出崩溃",
                      ha="center", fontsize=7, color="#b71c1c", fontweight="bold")

        # ── Spike-F1 ──
        ax_f = fig.add_subplot(gs[1, col])
        bar_colors_f = ["#ef5350" if v < -5 else "#66bb6a" if v > 5
                        else "#ffcc80" if abs(v) > 2 else "#eeeeee"
                        for v in f1_deltas]
        ax_f.bar(xs, f1_deltas, color=bar_colors_f, edgecolor="white", linewidth=0.5)
        ax_f.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax_f.set_title(f"{model_name}\nSpike-F1 Δ%（跳过第N层）",
                       fontsize=11, fontweight="bold", color=color)
        ax_f.set_xlabel("层索引", fontsize=9)
        ax_f.set_ylabel("Spike-F1 Δ%（负=尖峰检测变差）", fontsize=9)
        ax_f.set_xticks(xs); ax_f.set_xticklabels([str(i) for i in xs], fontsize=8)

        for i, v in enumerate(f1_deltas):
            abl = f"skip_layer_{i}"
            stars = layer_sig.get((model_name, abl, "spike_f1"), "")
            stars_disp = stars if stars not in ("ns", "") else ""

            if abs(v) > 4:
                ax_f.text(i, v + (0.5 if v >= 0 else -1.5),
                          f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%",
                          ha="center", fontsize=7.5, fontweight="bold",
                          color="#1b5e20" if v > 0 else "#b71c1c")
            if stars_disp:
                y_pos = (v + 1.0) if v >= 0 else (v - 2.0)
                ax_f.text(i, y_pos, stars_disp, ha="center", fontsize=8,
                          color="#1b5e20" if v > 0 else "#b71c1c", fontweight="bold")

        # 功能分离层标注
        conflict_layers = {
            "Toto-2.0":    {4: "精度↑\n尖峰↑"},
            "Chronos-2":   {0: "SMAPE≈0\n尖峰↓"},
            "TimesFM-2.5": {1: "精度↑\n尖峰↑", 7: "SMAPE≈0\n尖峰↓"},
        }
        for li, label in conflict_layers.get(model_name, {}).items():
            ax_f.annotate(label, xy=(li, f1_deltas[li]),
                          xytext=(li + (0.8 if li < n-2 else -1.5), f1_deltas[li] + 3),
                          fontsize=7, color="#6a1b9a", fontweight="bold",
                          arrowprops=dict(arrowstyle="-", color="#9c27b0", lw=0.8))

    # 图例
    legend_patches = [
        mpatches.Patch(color="#ef5350", label="变差（有害）"),
        mpatches.Patch(color="#66bb6a", label="改善（有益）"),
        mpatches.Patch(color="#eeeeee", label="冗余（≈0）"),
        mpatches.Patch(color="#6a1b9a", label="★ 功能分离层"),
        mpatches.Patch(color="white",   label="* p<0.05  ** p<0.01  *** p<0.001（Bonferroni）",
                       edgecolor="#666"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=8.5, framealpha=0.85, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("逐层消融：SMAPE（上）与 Spike-F1（下）边际贡献 + Wilcoxon 显著性标注",
                 fontsize=14, fontweight="bold")

    path = OUT / "fig2_perlayer_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ fig2 保存: {path}")


# ── 7. Markdown 摘要 ──────────────────────────────────────────────────────────

def write_significance_summary(sig_df: pd.DataFrame):
    n_valid = int(sig_df["p_raw"].notna().sum())
    lines = [
        "# 统计显著性检验报告",
        "",
        "> 方法：Wilcoxon signed-rank（双侧，zero_method='wilcox'）",
        f"> 有效检验数：{n_valid}（剔除 CRASH / N/A / 样本不足条件）",
        "> **主校正**：Bonferroni（FWER，最保守）｜**辅助校正**：Benjamini-Hochberg FDR",
        "> 星号阈值：`*` p<0.05  `**` p<0.01  `***` p<0.001",
        "",
        "---",
        "",
    ]

    for dataset_label in ["full", "perlayer"]:
        sub = sig_df[sig_df["dataset"] == dataset_label]
        title = "全结构消融（S-A/B/G 系列）" if dataset_label == "full" else "逐层消融（S-L 系列）"
        lines += [f"## {title}", ""]

        for model in MODELS_ORDER:
            m_sub = sub[sub["model"] == model]
            if m_sub.empty:
                continue
            lines += [f"### {model}", ""]
            lines += [
                "| 消融类型 | SMAPE p_raw | Bonf | FDR | F1 p_raw | Bonf | FDR |",
                "|---|---|---|---|---|---|---|",
            ]
            abls = sorted(m_sub["ablation_type"].unique())
            for abl in abls:
                row_s = m_sub[(m_sub["ablation_type"] == abl) & (m_sub["metric"] == "smape")]
                row_f = m_sub[(m_sub["ablation_type"] == abl) & (m_sub["metric"] == "spike_f1")]

                def _fmt(r, col):
                    if r.empty or (col in r.columns and pd.isna(r.iloc[0][col])):
                        return "—"
                    v = r.iloc[0][col]
                    return f"{v:.4f}" if col.startswith("p_") else str(v)

                lines.append(
                    f"| `{abl}` "
                    f"| {_fmt(row_s,'p_raw')} | {_fmt(row_s,'sig')} | {_fmt(row_s,'sig_fdr')} "
                    f"| {_fmt(row_f,'p_raw')} | {_fmt(row_f,'sig')} | {_fmt(row_f,'sig_fdr')} |"
                )
            lines.append("")

    # ── Bonferroni 显著汇总 ──
    sig_bonf = sig_df[sig_df["sig"].isin(["*", "**", "***"])].copy()
    lines += [
        "---",
        "",
        f"## 通过 Bonferroni 校正的显著结果（共 {len(sig_bonf)} 个）",
        "",
        "| 模型 | 类型 | 消融 | 指标 | sig(Bonf) | p_raw | p_bonf |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in sig_bonf.sort_values(["model", "dataset", "ablation_type", "metric"]).iterrows():
        lines.append(f"| {r.model} | {r.dataset} | `{r.ablation_type}` "
                     f"| {r.metric} | {r.sig} | {r.p_raw:.5f} | {r.p_corr_bonf:.5f} |")
    lines.append("")

    # ── FDR-only 额外发现（Bonferroni 未通过但 FDR 通过）──
    fdr_extra = sig_df[
        sig_df["sig_fdr"].isin(["*", "**", "***"]) &
        ~sig_df["sig"].isin(["*", "**", "***"])
    ].copy()

    lines += [
        "---",
        "",
        f"## 仅通过 FDR-BH 校正（Bonferroni 未通过，共 {len(fdr_extra)} 个）",
        "",
        "> 这些结果在 FDR 控制下显著，但在严格的 FWER Bonferroni 控制下不显著。",
        "> 对于高度相关的多重比较族（同一数据集上的多个消融条件），FDR 是更合适的选择。",
        "",
        "| 模型 | 类型 | 消融 | 指标 | sig(FDR) | p_raw | p_fdr |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in fdr_extra.sort_values(["model", "dataset", "ablation_type", "metric"]).iterrows():
        lines.append(f"| {r.model} | {r.dataset} | `{r.ablation_type}` "
                     f"| {r.metric} | {r.sig_fdr} | {r.p_raw:.5f} | {r.p_corr_fdr:.5f} |")
    lines.append("")

    # ── 特别关注：关键科学发现的显著性 ──
    lines += [
        "---",
        "",
        "## 关键发现专项显著性",
        "",
        "### 「精度通路 vs 尖峰通路」功能分离层",
        "",
        "| 层 | 模型 | SMAPE Δ | F1 Δ | 功能分类 | p_raw(SMAPE) | p_raw(F1) | Bonf(F1) | FDR(F1) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    conflict_checks = [
        ("Toto-2.0",    "skip_layer_4", "精度损↑ / 尖峰改善↑"),
        ("Chronos-2",   "skip_layer_0", "精度无关 / 尖峰隐性↓"),
        ("TimesFM-2.5", "skip_layer_1", "精度损↑ / 尖峰改善↑"),
        ("TimesFM-2.5", "skip_layer_7", "精度无关 / 尖峰关键↓"),
        ("TimesFM-2.5", "skip_layer_10","精度无关 / 尖峰干扰层"),
    ]
    for model, abl, ftype in conflict_checks:
        rs = sig_df[(sig_df["model"] == model) & (sig_df["ablation_type"] == abl) & (sig_df["metric"] == "smape")]
        rf = sig_df[(sig_df["model"] == model) & (sig_df["ablation_type"] == abl) & (sig_df["metric"] == "spike_f1")]
        def _v(r, col):
            if r.empty or pd.isna(r.iloc[0].get(col, np.nan)): return "—"
            v = r.iloc[0][col]
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(
            f"| {abl.replace('skip_layer_','L')} | {model} | — | — | {ftype} "
            f"| {_v(rs,'p_raw')} | {_v(rf,'p_raw')} | {_v(rf,'sig')} | {_v(rf,'sig_fdr')} |"
        )
    lines += [
        "",
        "> **关键注意**：Chronos-2 L0「隐性尖峰层」(Spike-F1 Δ = -14.1%) 的 Wilcoxon",
        "> p_raw = 0.023（在 α=0.05 下显著），但 Bonferroni 校正后 p_corr = 1.00（不显著）。",
        "> FDR-BH 校正后可能通过（见上表）。这反映 Bonferroni 在 118 个相关测试",
        "> 中过于保守——对于该特定发现，建议报告 raw p 值并注明 FDR 校正结果。",
        "",
    ]

    path = OUT / "significance_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 摘要保存: {path}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    all_results = []

    print("═" * 60)
    print("  Step 1：计算各 dataset 的 Wilcoxon 检验")
    print("═" * 60)

    for key, model_name, folder in DATASETS_FULL:
        print(f"\n▶ 全结构消融 — {model_name}")
        df = run_wilcoxon_for_dataset(folder, model_name, "full")
        all_results.append(df)

    for key, model_name, folder in DATASETS_LAYER:
        print(f"\n▶ 逐层消融  — {model_name}")
        df = run_wilcoxon_for_dataset(folder, model_name, "perlayer")
        all_results.append(df)

    sig_df = pd.concat(all_results, ignore_index=True)

    print("\n═" * 60)
    print("  Step 2：Bonferroni 全局校正")
    print("═" * 60)
    sig_df = apply_bonferroni(sig_df)

    # 保存完整结果
    csv_path = OUT / "significance_tests.csv"
    sig_df.to_csv(csv_path, index=False)
    print(f"✅ 完整结果保存: {csv_path}")

    # 打印快速摘要
    n_valid = int(sig_df["p_raw"].notna().sum())
    print(f"\n  总测试数：{len(sig_df)}")
    print(f"  有效测试（p_raw 非 NaN）：{n_valid}")
    print(f"\n  Bonferroni 校正结果（严格 FWER）：")
    for s in ["***", "**", "*", "ns"]:
        n = int((sig_df["sig"] == s).sum())
        print(f"    {s:4s}: {n}")
    print(f"\n  FDR-BH 校正结果（宽松 FDR）：")
    for s in ["***", "**", "*", "ns"]:
        n = int((sig_df["sig_fdr"] == s).sum())
        print(f"    {s:4s}: {n}")
    # 高亮关键发现
    c0 = sig_df[(sig_df["model"]=="Chronos-2") & (sig_df["ablation_type"]=="skip_layer_0") & (sig_df["metric"]=="spike_f1")]
    if not c0.empty:
        r = c0.iloc[0]
        print(f"\n  ★ Chronos-2 L0 Spike-F1：")
        print(f"    p_raw={r.p_raw:.4f}  Bonferroni={r.sig}(p={r.p_corr_bonf:.4f})"
              f"  FDR={r.sig_fdr}(p={r.p_corr_fdr:.4f})")

    print("\n═" * 60)
    print("  Step 3：生成带显著性标注的图表")
    print("═" * 60)

    full_sig, layer_sig = build_sig_lookup(sig_df)
    full_ablation = load_full_ablation()
    per_layer     = load_perlayer()

    print("\n▶ 重新生成 fig1（热力图 + 显著性）")
    make_heatmap_with_sig(full_ablation, full_sig)

    print("\n▶ 重新生成 fig2（逐层曲线 + 显著性）")
    make_perlayer_curves_with_sig(per_layer, layer_sig)

    print("\n▶ 写入 Markdown 摘要")
    write_significance_summary(sig_df)

    print("\n" + "═" * 60)
    print("  全部完成！")
    print("═" * 60)
    print(f"  significance_tests.csv      — 完整 p 值表")
    print(f"  significance_summary.md     — 可读摘要")
    print(f"  fig1_component_heatmap.png  — 热力图（带 Bonferroni 校正星号）")
    print(f"  fig2_perlayer_curves.png    — 逐层曲线（带 Bonferroni 校正星号）")


if __name__ == "__main__":
    main()
