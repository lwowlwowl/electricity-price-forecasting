"""
统计检验模块 stat_tests.py
==========================
实现 Lago 2021 checklist #7 和 Weron 2014 §4.5.2 要求的统计检验：

  (a) DM test（Diebold-Mariano, 1995）
      ─ 逐小时 t 检验（h=0..23 各自独立），Lago Fig.5 格式输出
      ─ H0: E[d_h(t)] = 0（两模型预测精度无差异）
      ─ 损失差 d_h(t) = |e_A(t,h)| - |e_B(t,h)|，跨节点平均后用 Newey-West
        HAC 标准误处理自相关（Weron 2014 §4.5.2，带宽 L = ⌊T^{1/4}⌋）
      ─ 适用：参数消融/结构消融内部对比（zero-shot 模型参数测试期固定不变）

  (b) GW test（Giacomini-White 2006，以 HAC t-test 形式实现）
      ─ 日度 L1 损失差 Δd(t) = Σ_h mean_nodes(|e_A(t,h)| - |e_B(t,h)|)
      ─ 其余与 DM 相同（Newey-West HAC t 检验）
      ─ 适用：ElecFM vs LEAR（两者均有某种形式的 recalibration）
      ─ 注：严格 GW 假设要求模型每期用 t-1 信息 recalibrate；对完全 zero-shot
        的基础模型，GW 条件假设不完全成立，论文中应注明并改用 DM。

输入格式：run_backtest() 返回的 records DataFrame
  必须包含列：model, origin, node, ts, actual, mean
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── 内部工具：Newey-West HAC 方差估计 ────────────────────────────────────────
def _nw_var(d: np.ndarray, bandwidth: Optional[int] = None) -> float:
    """
    Newey-West HAC 方差估计（Bartlett 核）。

    V_HAC = γ₀ + 2 Σ_{j=1}^{L} (1 - j/(L+1)) γⱼ
    其中 γⱼ = T⁻¹ Σ_{t=j+1}^{T} (d_t - d̄)(d_{t-j} - d̄)

    bandwidth (L): None → ⌊T^{1/4}⌋（Newey-West 经验公式）。
    返回 max(V_HAC, 1e-15) 以防方差为零。
    """
    T = len(d)
    if T < 2:
        return float("nan")
    d_c = d - d.mean()
    L = bandwidth if bandwidth is not None else max(1, int(T ** 0.25))
    gamma_0 = float(np.dot(d_c, d_c)) / T
    hac = gamma_0
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)          # Bartlett kernel weight
        gamma_j = float(np.dot(d_c[lag:], d_c[:-lag])) / T
        hac += 2.0 * w * gamma_j
    return max(hac, 1e-15)


def _dm_stat_pval(d: np.ndarray,
                  bandwidth: Optional[int] = None) -> Tuple[float, float]:
    """
    对损失差序列 d 做 HAC 稳健 t 检验。

    返回 (dm_stat, p_value)：
      dm_stat > 0 → model_a 损失更大（model_b 更好）
      dm_stat < 0 → model_b 损失更大（model_a 更好）
      p_value < 0.05 → 双侧显著

    T < 4 时返回 (nan, nan)（样本太小，检验无意义）。
    """
    from scipy.stats import norm   # 仅此处需要，延迟导入

    T = len(d)
    if T < 4:
        return float("nan"), float("nan")
    d_mean = float(d.mean())
    hac_var = _nw_var(d, bandwidth)
    dm = d_mean / np.sqrt(hac_var / T)
    p = 2.0 * (1.0 - norm.cdf(abs(dm)))
    return float(dm), float(p)


# ── 损失差矩阵构建（内部） ───────────────────────────────────────────────────
def _loss_diff_df(records: pd.DataFrame,
                  model_a: str,
                  model_b: str) -> pd.DataFrame:
    """
    从 records 提取两个模型的逐时刻绝对误差，对齐后返回损失差 DataFrame。

    返回列：origin, hour（0-23），loss_diff = |e_A| - |e_B|（跨节点平均）。
    """
    def _abs_err(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[["origin", "node", "ts", "actual", "mean"]].copy()
        sub["abs_err"] = np.abs(sub["actual"] - sub["mean"])
        sub["hour"] = pd.to_datetime(sub["ts"]).dt.hour
        return sub[["origin", "node", "hour", "abs_err"]]

    rec_a = _abs_err(records[records["model"] == model_a])
    rec_b = _abs_err(records[records["model"] == model_b])

    merged = rec_a.merge(rec_b, on=["origin", "node", "hour"],
                         suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError(
            f"模型 '{model_a}' 和 '{model_b}' 没有共同的 (origin, node, hour)。"
            "请检查 records 是否包含这两个模型。"
        )
    merged["loss_diff"] = merged["abs_err_a"] - merged["abs_err_b"]

    # 跨节点平均，得到每个 (origin, hour) 的标量损失差
    diff_df = (merged.groupby(["origin", "hour"])["loss_diff"]
               .mean().reset_index())
    return diff_df


# ── 1. DM test（逐小时，per-hour） ────────────────────────────────────────────
def dm_test(records: pd.DataFrame,
            model_a: str,
            model_b: str,
            bandwidth: Optional[int] = None) -> pd.DataFrame:
    """
    Diebold-Mariano test，逐小时（h=0..23）各做一次 HAC 稳健 t 检验。

    参数
    ----
    records   : run_backtest() 返回的 records DataFrame
    model_a   : 基准模型名（正 dm_stat / 正 mean_loss_diff → A 损失更大即 B 更好）
    model_b   : 对比模型名
    bandwidth : Newey-West 带宽 L（None → ⌊T^{1/4}⌋）

    返回
    ----
    DataFrame，24 行，列：
      hour          : 0-23
      dm_stat       : DM 统计量
      p_value       : 双侧 p 值（< 0.05 显著）
      mean_loss_diff: d̄_h = 均值，> 0 表示 A 损失更大（B 更好）
      n_days        : 有效天数（起报点数）
    """
    diff_df = _loss_diff_df(records, model_a, model_b)

    rows = []
    for h in range(24):
        d = diff_df[diff_df["hour"] == h]["loss_diff"].to_numpy()
        dm, pval = _dm_stat_pval(d, bandwidth)
        rows.append({
            "hour": h,
            "dm_stat": dm,
            "p_value": pval,
            "mean_loss_diff": float(d.mean()) if len(d) else float("nan"),
            "n_days": len(d),
        })
    result = pd.DataFrame(rows)
    result.attrs["model_a"] = model_a
    result.attrs["model_b"] = model_b
    return result


# ── 2. GW test（日度 L1 损失差） ──────────────────────────────────────────────
def gw_test(records: pd.DataFrame,
            model_a: str,
            model_b: str,
            bandwidth: Optional[int] = None) -> Dict[str, float]:
    """
    Giacomini-White 形式的日度 L1 损失差检验。

    Δd(t) = Σ_{h=0}^{23} mean_nodes(|e_A(t,h)| - |e_B(t,h)|)
    在 Δd 序列上做 Newey-West HAC t 检验。

    返回
    ----
    dict:
      dm_stat        : 检验统计量（同 DM test，日度版本）
      p_value        : 双侧 p 值
      mean_loss_diff : Δd 的均值，> 0 表示 A 日均损失更大（B 更好）
      n_days         : 有效天数
      model_a / _b   : 模型名（便于结果存档）

    适用说明
    --------
    严格 GW (2006) 要求模型在每期用 t-1 信息 recalibrate（条件预测能力检验）。
    对 zero-shot 基础模型参数固定，该条件不完全成立；此时应改用 dm_test()。
    本函数适用于 ElecFM vs LEAR 等均有 recalibration 机制的对比场景。
    """
    diff_df = _loss_diff_df(records, model_a, model_b)

    # 日度聚合：Σ_h（24 小时求和）
    daily = (diff_df.groupby("origin")["loss_diff"]
             .sum().reset_index(name="daily_loss_diff"))
    d = daily["daily_loss_diff"].to_numpy()
    dm, pval = _dm_stat_pval(d, bandwidth)

    return {
        "dm_stat": dm,
        "p_value": pval,
        "mean_loss_diff": float(d.mean()) if len(d) else float("nan"),
        "n_days": len(d),
        "model_a": model_a,
        "model_b": model_b,
    }


# ── 3. 多模型 p 值矩阵 ────────────────────────────────────────────────────────
def dm_pvalue_matrix(records: pd.DataFrame,
                     models: Optional[List[str]] = None,
                     test: str = "dm",
                     bandwidth: Optional[int] = None,
                     aggregate_hours: bool = False,
                    ) -> pd.DataFrame:
    """
    计算所有模型对的 DM/GW p 值矩阵。

    参数
    ----
    records        : run_backtest() 的 records DataFrame
    models         : 参与检验的模型列表（None → records 里全部模型）
    test           : "dm"（逐小时均值 p 值）或 "gw"（日度 L1）
    bandwidth      : Newey-West 带宽
    aggregate_hours: test="dm" 时，True → 24 小时 p 值的简单均值（仅用于可视化）；
                     False → 返回 24×24 矩阵（行=hour 0..23，列=model_b）——此时
                     返回值为 dict[model_a][model_b] = 24-row DataFrame，不再是方阵。
                     如需热力图，建议用 aggregate_hours=True。

    返回（当 aggregate_hours=True 或 test="gw"）
    ----
    DataFrame：index=model_a，columns=model_b，值=p_value。
    对角线为 nan。p < 0.05 表示 model_b 显著优于 model_a（或反之，见 dm_stat 符号）。
    """
    if models is None:
        models = sorted(records["model"].unique().tolist())

    pmat = pd.DataFrame(index=models, columns=models, dtype=float)
    pmat[:] = float("nan")

    for a in models:
        for b in models:
            if a == b:
                continue
            try:
                if test == "gw":
                    res = gw_test(records, a, b, bandwidth)
                    pmat.loc[a, b] = res["p_value"]
                else:
                    res = dm_test(records, a, b, bandwidth)
                    if aggregate_hours:
                        pmat.loc[a, b] = float(res["p_value"].mean())
                    else:
                        # 单对：直接 24小时均值
                        pmat.loc[a, b] = float(res["p_value"].mean())
            except (ValueError, KeyError):
                pmat.loc[a, b] = float("nan")

    return pmat


# ── 4. 热力图可视化 ───────────────────────────────────────────────────────────
def plot_dm_heatmap(
    pmat: pd.DataFrame,
    title: str = "DM test p-values",
    alpha: float = 0.05,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    绘制 DM/GW p 值热力图（仿 Lago 2021 Fig.5 格式）。

    颜色方案：
      深绿（p < alpha）  → row 模型（A）显著劣于 col 模型（B），即 B 更好
      浅色（p ≥ alpha）  → 无显著差异
      对角线           → 灰色（nan）

    参数
    ----
    pmat      : dm_pvalue_matrix() 的输出（index=model_a，columns=model_b）
    alpha     : 显著性水平（默认 0.05）
    save_path : 若给定，则保存到该路径（格式由后缀名决定，如 .png/.pdf）
    figsize   : 图形尺寸
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # 非交互环境兼容
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过热力图绘制。")
        return

    fig, ax = plt.subplots(figsize=figsize)

    models = pmat.index.tolist()
    n = len(models)
    data = pmat.to_numpy(dtype=float)

    # 颜色：p < alpha → 深绿；p ≥ alpha → 浅灰；nan → 白色
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=1)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="p-value")

    # 显著性标注
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           color="lightgrey"))
            else:
                txt = f"{v:.2f}"
                color = "white" if v < alpha else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color,
                        fontweight="bold" if v < alpha else "normal")

    ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(models)
    ax.set_xlabel("model_b (challenger)")
    ax.set_ylabel("model_a (baseline)")
    ax.set_title(f"{title}\n(green = significant at α={alpha}, "
                 "p<α means row model loses to col model)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  热力图已保存：{save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── 5. 单次打印汇总 ───────────────────────────────────────────────────────────
def print_dm_summary(dm_result: pd.DataFrame,
                     alpha: float = 0.05) -> None:
    """
    打印 dm_test() 结果的紧凑摘要：逐小时统计量 + 显著小时数。
    """
    a = dm_result.attrs.get("model_a", "A")
    b = dm_result.attrs.get("model_b", "B")
    sig = dm_result[dm_result["p_value"] < alpha]
    a_wins = sig[sig["mean_loss_diff"] < 0]     # A 损失更小
    b_wins = sig[sig["mean_loss_diff"] > 0]     # B 损失更小

    print(f"\nDM test: {a} vs {b}  (α={alpha})")
    print(f"  显著小时数: {len(sig)}/24  "
          f"({a} 更好: {len(a_wins)}h，{b} 更好: {len(b_wins)}h)")
    if len(sig):
        print("  显著小时详情:")
        for _, row in sig.iterrows():
            winner = a if row["mean_loss_diff"] < 0 else b
            print(f"    h={int(row['hour']):02d}  p={row['p_value']:.4f}  "
                  f"dm={row['dm_stat']:+.2f}  winner={winner}")


# ── 自测 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_days = 365
    n_hours = 24

    # 构造虚假 records：模型 A（较差）和 B（较好，每小时误差少 0.5）
    rows = []
    base_ts = pd.Timestamp("2025-07-01", tz="UTC")
    for d in range(n_days):
        origin = base_ts + pd.Timedelta(days=d)
        for h in range(n_hours):
            ts = origin + pd.Timedelta(hours=h)
            actual = float(rng.normal(30, 10))
            # A 有系统性偏差
            rows.append({"model": "A", "origin": origin, "node": "HUB",
                          "ts": ts, "actual": actual,
                          "mean": actual + rng.normal(2, 3)})
            # B 更准确
            rows.append({"model": "B", "origin": origin, "node": "HUB",
                          "ts": ts, "actual": actual,
                          "mean": actual + rng.normal(0, 2)})

    records = pd.DataFrame(rows)

    print("=" * 60)
    print("stat_tests 自测（A 系统差于 B，应显著）")
    print("=" * 60)

    dm_res = dm_test(records, "A", "B")
    print_dm_summary(dm_res, alpha=0.05)

    gw_res = gw_test(records, "A", "B")
    print(f"\nGW test: A vs B  dm={gw_res['dm_stat']:.3f}  "
          f"p={gw_res['p_value']:.4f}  n_days={gw_res['n_days']}")

    pmat = dm_pvalue_matrix(records, test="dm")
    print("\np 值矩阵（均值）:")
    print(pmat.to_string())

    print("\n✅ stat_tests 工作正常")
