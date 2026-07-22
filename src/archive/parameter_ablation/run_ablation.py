"""
消融执行器 run_ablation.py
==========================
手册 §6 的落地：一个消融 = 固定所有旋钮、只转动一个 `key`，遍历它的
`values` 各档，对每一档跑一次完整滚动回测，最后汇总成

  1) 一张【跨档总表】 ablation_summary.csv   （行=模型×档位，含全部指标）
  2) 一张【旋钮扫描折线图】 ablation_<key>.png（横轴=档位，每模型一条线）

这就是手册要求的"每个消融一表一图"。

配置约定（在基准配置基础上加一段 ablate）：

    # configs/parameter_ablation/ablation_B_context.yaml
    name: ablation_B_context
    ...（基准旋钮）...
    ablate:
      key: context_len          # 要转动的旋钮（顶层配置的某个 key）
      values: [168, 336, 720]   # 各档取值
      labels: ["7天", "14天", "30天"]   # 可选，折线图横轴显示用

用法：
    python run_ablation.py configs/parameter_ablation/ablation_B_context.yaml
"""

from __future__ import annotations

import copy
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "evaluation"))

import run_experiment as RE          # noqa: E402  复用取数/建模/回测/落盘
from plotting import plot_ablation   # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "data", "results", "parameter_ablation")


def _set_knob(cfg: dict, key: str, value):
    """把旋钮值写入配置。支持顶层 key，以及 'backtest.xxx' 点路径。"""
    if "." in key:
        head, tail = key.split(".", 1)
        cfg.setdefault(head, {})
        _set_knob(cfg[head], tail, value)
    else:
        cfg[key] = value


def run_ablation(cfg: dict) -> dict:
    ab = cfg.get("ablate")
    if not ab or "key" not in ab or "values" not in ab:
        raise ValueError("配置缺少 ablate.{key,values}，这不是一个消融配置")

    key = ab["key"]
    values = ab["values"]
    labels = ab.get("labels") or [str(v) for v in values]
    base_name = cfg["name"]

    # linked：与主旋钮联动的额外旋钮（每档对应一个 {knob: value} 字典）
    # 例如频率消融里，改 freq 的同时联动 context_len / horizon / backtest.stride_hours
    linked_list = ab.get("linked") or [{} for _ in values]
    if len(linked_list) != len(values):
        raise ValueError(
            f"ablate.linked 长度（{len(linked_list)}）必须与 ablate.values（{len(values)}）一致"
        )

    print("=" * 70)
    print(f"消融实验：{base_name}　转动旋钮：{key}　档位：{values}")
    if any(linked_list):
        print(f"  联动旋钮：{[list(d.keys()) for d in linked_list if d]}")
    print("=" * 70)

    per_level = []          # [(label, summary_df), ...] 给画图
    per_level_records = []  # [records_df, ...] 给 DM test
    merged_rows = []        # 跨档总表

    for val, label, linked in zip(values, labels, linked_list):
        sub = copy.deepcopy(cfg)
        sub.pop("ablate", None)
        _set_knob(sub, key, val)
        # 联动旋钮：与主旋钮同步写入
        for lk, lv in linked.items():
            _set_knob(sub, lk, lv)
        # 每档单独落盘目录，避免互相覆盖
        sub["name"] = f"{base_name}/{key}={val}"

        print(f"\n{'─' * 60}\n▶ 档位 {key}={val}（{label}）\n{'─' * 60}")
        result = RE.run(sub)
        summ = result["summary"].copy()
        per_level.append((label, summ))
        per_level_records.append((label, result["records"].copy()))

        tagged = summ.copy()
        tagged.insert(0, "knob", key)
        # val 可能是列表（如协变量消融 []→[load]→…），pandas.insert 会把列表
        # 当数组匹配行数；统一转字符串存储。
        tagged.insert(1, "knob_value", str(val))
        tagged.insert(2, "knob_label", label)
        merged_rows.append(tagged)

    # ── 跨档总表 ──────────────────────────────────────────────────────────
    out_dir = os.path.join(RESULTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)
    merged = pd.concat(merged_rows, ignore_index=True)
    merged_path = os.path.join(out_dir, "ablation_summary.csv")
    merged.to_csv(merged_path, index=False)

    # ── 折线图 ────────────────────────────────────────────────────────────
    png_path = os.path.join(out_dir, f"ablation_{key.replace('.', '_')}.png")
    knob_label_axis = ab.get("axis_label", key)
    plot_ablation(
        [(lbl, df) for lbl, df in per_level],
        knob_label=knob_label_axis,
        out_png=png_path,
        title=f"消融：{key}（{base_name}）",
    )

    # ── DM test（消融配置间显著性检验，Lago checklist #7）──────────────────
    dm_result = _run_ablation_dm_tests(per_level_records, labels, out_dir, key)

    print(f"\n{'=' * 70}")
    print(f"✅ 消融完成")
    print(f"   跨档总表：{merged_path}")
    print(f"   折线图　：{png_path}")
    if dm_result is not None:
        print(f"   DM test ：{os.path.join(out_dir, 'ablation_dm_tests.csv')}")
    print("=" * 70)

    # 打印一张精简跨档对比（每模型在各档的 rMAE / Spike-F1）
    _print_pivot(merged, key)

    return {
        "merged": merged,
        "summary_path": merged_path,
        "png": png_path,
        "dm_tests": dm_result,
    }


def _print_pivot(merged: pd.DataFrame, key: str):
    # rMAE 优先（Lago 主指标），MAE 备用，Spike-F1 单独一张
    metrics = [
        ("rmae_mean", "rMAE"),
        ("mae_mean", "MAE"),
        ("spike_f1_mean_signal", "Spike-F1(mean)"),
    ]
    for metric, name in metrics:
        if metric not in merged.columns:
            continue
        piv = merged.pivot_table(index="model", columns="knob_value",
                                 values=metric, aggfunc="first")
        print(f"\n── {name} 随 {key} 变化（行=模型，列=档位）──")
        print(piv.round(4).to_string())


def _run_ablation_dm_tests(
    per_level_records: list,   # [(label, records_df), ...]
    labels: list,
    out_dir: str,
    key: str,
) -> "pd.DataFrame | None":
    """
    消融配置间 DM test（Lago checklist #7）。

    对每个模型，以第一档（baseline）为参照，依次对其余各档做 per-hour DM test。
    正 dm_stat → baseline 损失更大（challenger 更好）。
    负 dm_stat → baseline 更好。

    输出 ablation_dm_tests.csv：行 = (模型, 档位对)，含显著小时数和均值 p 值。
    """
    sys.path.insert(0, os.path.join(ROOT, "src", "evaluation"))
    try:
        from stat_tests import dm_test
    except ImportError:
        print("  ⚠️  stat_tests 未找到，跳过 DM test")
        return None

    if len(per_level_records) < 2:
        return None

    baseline_label, baseline_records = per_level_records[0]
    dm_rows = []

    for challenger_label, challenger_records in per_level_records[1:]:
        # 将两份 records 合并，用"虚拟"模型名区分，再逐模型做 DM test
        models = set(baseline_records["model"].unique()) & \
                 set(challenger_records["model"].unique())

        for model in sorted(models):
            rec_b = baseline_records[baseline_records["model"] == model].copy()
            rec_c = challenger_records[challenger_records["model"] == model].copy()
            # 重命名为虚拟名以供 dm_test 区分
            rec_b["model"] = f"baseline"
            rec_c["model"] = f"challenger"
            combined = pd.concat([rec_b, rec_c], ignore_index=True)

            try:
                dm_res = dm_test(combined, "baseline", "challenger")
                sig = dm_res[dm_res["p_value"] < 0.05]
                dm_rows.append({
                    "model":               model,
                    "baseline_config":     baseline_label,
                    "challenger_config":   challenger_label,
                    "sig_hours_p05":       int(len(sig)),
                    "baseline_wins":       int((sig["mean_loss_diff"] < 0).sum()),
                    "challenger_wins":     int((sig["mean_loss_diff"] > 0).sum()),
                    "mean_dm_stat":        round(float(dm_res["dm_stat"].mean()), 4),
                    "mean_p_value":        round(float(dm_res["p_value"].mean()), 4),
                })
            except Exception as e:
                dm_rows.append({
                    "model": model,
                    "baseline_config": baseline_label,
                    "challenger_config": challenger_label,
                    "error": str(e),
                })

    if not dm_rows:
        return None

    dm_df = pd.DataFrame(dm_rows)
    dm_path = os.path.join(out_dir, "ablation_dm_tests.csv")
    dm_df.to_csv(dm_path, index=False)

    # 终端摘要
    print(f"\n── DM test（{baseline_label} vs 各档，α=0.05）──")
    print(f"{'模型':20s}  {'对比档':20s}  {'显著h/24':8s}  {'base赢':6s}  "
          f"{'chall赢':7s}  {'均值p':7s}")
    for _, r in dm_df.iterrows():
        if "error" in r and pd.notna(r.get("error", None)):
            print(f"  {r['model']:20s}  {r['challenger_config']:20s}  ERROR: {r['error']}")
        else:
            print(f"  {r['model']:20s}  {r['challenger_config']:20s}  "
                  f"{r.get('sig_hours_p05','?'):8}  "
                  f"{r.get('baseline_wins','?'):6}  "
                  f"{r.get('challenger_wins','?'):7}  "
                  f"{r.get('mean_p_value','?'):7}")
    return dm_df


def main():
    """
    用法：
        python run_ablation.py <config.yaml> [--market PJM] [--nodes_group ablation]

    与 run_experiment.py 相同的 CLI 覆盖约定，便于多市场批量消融。
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--market",      default=None)
    parser.add_argument("--nodes_group", default=None)
    parser.add_argument("--suffix",      default=None)
    parser.add_argument("-h", "--help",  action="store_true")
    args, _ = parser.parse_known_args()

    if args.help:
        parser.print_help()
        return

    if not args.config:
        print("用法：python run_ablation.py <消融配置.yaml> [--market PJM]")
        sys.exit(1)

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(ROOT, cfg_path)
    cfg = RE._load_yaml(cfg_path)
    print(f"加载消融配置：{cfg_path}")

    # 多市场 CLI 覆盖
    if args.market:
        cfg["market"] = args.market
        suffix = args.suffix or args.market.lower()
        cfg["name"] = f"{cfg['name']}_{suffix}"
        print(f"  覆盖 market → {args.market}（name → {cfg['name']}）")
    if args.nodes_group:
        cfg["nodes_group"] = args.nodes_group
        print(f"  覆盖 nodes_group → {args.nodes_group}")

    run_ablation(cfg)


if __name__ == "__main__":
    main()
