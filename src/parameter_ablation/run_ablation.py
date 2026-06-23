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

RESULTS_DIR = os.path.join(ROOT, "data", "results")


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

    print("=" * 70)
    print(f"消融实验：{base_name}　转动旋钮：{key}　档位：{values}")
    print("=" * 70)

    per_level = []          # [(label, summary_df), ...] 给画图
    merged_rows = []        # 跨档总表

    for val, label in zip(values, labels):
        sub = copy.deepcopy(cfg)
        sub.pop("ablate", None)
        _set_knob(sub, key, val)
        # 每档单独落盘目录，避免互相覆盖
        sub["name"] = f"{base_name}/{key}={val}"

        print(f"\n{'─' * 60}\n▶ 档位 {key}={val}（{label}）\n{'─' * 60}")
        result = RE.run(sub)
        summ = result["summary"].copy()
        per_level.append((label, summ))

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

    print(f"\n{'=' * 70}")
    print(f"✅ 消融完成")
    print(f"   跨档总表：{merged_path}")
    print(f"   折线图　：{png_path}")
    print("=" * 70)

    # 打印一张精简跨档对比（每模型在各档的 MAE / Spike-F1）
    _print_pivot(merged, key)

    return {"merged": merged, "summary_path": merged_path, "png": png_path}


def _print_pivot(merged: pd.DataFrame, key: str):
    for metric, name in [("mae_mean", "MAE"), ("spike_f1_mean_signal", "Spike-F1")]:
        if metric not in merged.columns:
            continue
        piv = merged.pivot_table(index="model", columns="knob_value",
                                 values=metric, aggfunc="first")
        print(f"\n── {name} 随 {key} 变化（行=模型，列=档位）──")
        print(piv.round(3).to_string())


def main():
    if len(sys.argv) < 2:
        print("用法：python run_ablation.py <消融配置.yaml>")
        sys.exit(1)
    cfg_path = sys.argv[1]
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(ROOT, cfg_path)
    cfg = RE._load_yaml(cfg_path)
    print(f"加载消融配置：{cfg_path}")
    run_ablation(cfg)


if __name__ == "__main__":
    main()
