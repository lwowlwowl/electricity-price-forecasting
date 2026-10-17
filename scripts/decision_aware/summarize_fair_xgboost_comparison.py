#!/usr/bin/env python
"""生成无ID Transformer 与同数据 pooled LOMO XGBoost 的公平对照表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")
TASKS = ("da", "rt_at_da", "rt")


def macro_mae(path: Path) -> float:
    with np.load(path, allow_pickle=False) as archive:
        prediction = archive["prediction"].astype(np.float64)
        target_key = "target" if "target" in archive.files else (
            "target_da" if "target_da" in archive.files else "target_rt"
        )
        target = archive[target_key].astype(np.float64)
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError(f"{path}预测/真值形状不符合节点宏平均合同")
    axes = tuple(range(1, prediction.ndim))
    return float(np.mean(np.mean(np.abs(prediction - target), axis=axes)))


def build_rows(root: Path) -> list[dict]:
    rows = []
    for market in MARKETS:
        for task in TASKS:
            transformer_path = (
                root / "data/results" / f"universal_lomo_{task}_v1"
                / f"{market.lower()}_{task}_v1.npz"
            )
            xgboost_path = (
                root / "data/results/universal_lomo_xgboost_v1"
                / f"holdout_{market.lower()}_{task}_inner_seed0.npz"
            )
            transformer = macro_mae(transformer_path)
            xgboost = macro_mae(xgboost_path)
            rows.append({
                "market": market,
                "task": task,
                "transformer_mae": transformer,
                "pooled_lomo_xgboost_mae": xgboost,
                "winner": "transformer" if transformer < xgboost else "xgboost",
                "transformer_relative_improvement_pct": float(
                    100.0 * (xgboost - transformer) / xgboost
                ),
            })
    return rows


def paired_daily_profit(candidate: dict, reference: dict) -> dict:
    candidate_daily = candidate.get("macro_daily_net_revenue")
    reference_daily = reference.get("macro_daily_net_revenue")
    if not isinstance(candidate_daily, dict) or not isinstance(reference_daily, dict):
        raise ValueError("盈利配对需要macro_daily_net_revenue逐日序列")
    dates = sorted(reference_daily)
    if dates != sorted(candidate_daily):
        raise ValueError("候选系统与公平基线的逐日键不一致")
    difference = np.asarray(
        [candidate_daily[day] - reference_daily[day] for day in dates],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20261016)
    block = 7
    draws = 5000
    block_count = int(np.ceil(difference.size / block))
    starts = rng.integers(0, difference.size, size=(draws, block_count))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets) % difference.size
    samples = difference[indices.reshape(draws, -1)[:, : difference.size]]
    return {
        "mean_daily_difference": float(difference.mean()),
        "seven_day_block_bootstrap_95ci": [
            float(value) for value in np.quantile(samples.mean(axis=1), [0.025, 0.975])
        ],
    }


def daily_risk(item: dict) -> dict:
    daily = item.get("macro_daily_net_revenue")
    if not isinstance(daily, dict) or not daily:
        raise ValueError("风险汇总需要非空macro_daily_net_revenue逐日序列")
    values = np.asarray([daily[day] for day in sorted(daily)], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    midpoint = len(values) // 2

    def without_top(count: int) -> float | None:
        if len(values) <= count:
            return None
        return float(np.mean(np.sort(values)[:-count]))

    node_means = item.get("node_mean_daily_net_revenue")
    if isinstance(node_means, dict):
        node_values = np.asarray(list(node_means.values()), dtype=np.float64)
    elif isinstance(item.get("nodes"), dict):
        node_values = np.asarray([
            node["mean_daily_revenue"] for node in item["nodes"].values()
        ], dtype=np.float64)
    else:
        node_values = np.asarray([], dtype=np.float64)
    if node_values.size:
        rng = np.random.default_rng(20261017)
        samples = rng.choice(
            node_values, size=(5000, node_values.size), replace=True
        ).mean(axis=1)
        node_ci = [
            float(value) for value in np.quantile(samples, [0.025, 0.975])
        ]
    else:
        node_ci = None
    return {
        "positive_day_rate": float(np.mean(values > 0.0)),
        "maximum_drawdown_usd_per_node": float(drawdown.max()),
        "mean_without_top_5_days": without_top(5),
        "mean_without_top_10_days": without_top(10),
        "first_half_mean": float(values[:midpoint].mean()),
        "second_half_mean": float(values[midpoint:].mean()),
        "worst_node_mean_daily_revenue": item.get(
            "worst_node_mean_daily_net_revenue",
            item.get("worst_node_mean_daily_revenue"),
        ),
        "node_bootstrap_mean_95ci": node_ci,
    }


def profit_rows(root: Path) -> list[dict]:
    path = root / "data/results/tail_profit_diagnostics_v1/tail_profit_diagnostics.json"
    if not path.is_file():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))["ercot_profit"]
    pooled = report.get("xgboost_pooled", {}).get("inner")
    if pooled is None:
        return []
    candidates = []
    if "inner" in report.get("transformer", {}):
        candidates.append((
            "transformer", "无ID Transformer（seed 0）",
            report["transformer"]["inner"],
        ))
    multiseed_path = (
        root / "data/results/universal_lomo_ercot_multiseed_continuous_mpc_v1.json"
    )
    if multiseed_path.is_file():
        multiseed = json.loads(multiseed_path.read_text(encoding="utf-8"))
        ensemble = multiseed["systems"][
            "da_ensemble__rt_at_da_ensemble__rt_ensemble"
        ]
        candidates.append((
            "transformer_ensemble", "无ID Transformer（固定三seed等权集成）",
            ensemble,
        ))
    candidates.append((
        "xgboost_pooled", "pooled LOMO XGBoost（同数据公平基线）", pooled,
    ))
    if "inner" in report.get("xgboost_local", {}):
        candidates.append((
            "xgboost_local", "逐节点本地XGBoost（目标市场监督上界）",
            report["xgboost_local"]["inner"],
        ))
    rows = []
    for system, label, item in candidates:
        paired = (
            {"mean_daily_difference": 0.0,
             "seven_day_block_bootstrap_95ci": [0.0, 0.0]}
            if system == "xgboost_pooled"
            else paired_daily_profit(item, pooled)
        )
        rows.append({
            "system": system,
            "label": label,
            "ercot_2023_mean_daily_net_revenue": item.get(
                "macro_mean_daily_net_revenue",
                item.get("macro_mean_daily_revenue"),
            ),
            "paired_vs_pooled_lomo_xgboost": paired,
            "risk": daily_risk(item),
        })
    return rows


def render_markdown(summary: dict) -> str:
    lines = [
        "# 同数据公平XGBoost对照", "",
        "> 两者均只用三个源市场2020—2022训练，在第四市场2023零样本评价；未读取2025。", "",
        "| 留出市场 | 任务 | 无ID Transformer MAE | pooled LOMO XGBoost MAE | 胜者 | Transformer相对变化 |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in summary["forecast_rows"]:
        t = f"{row['transformer_mae']:.2f}"
        x = f"{row['pooled_lomo_xgboost_mae']:.2f}"
        if row["winner"] == "transformer":
            t = f"**{t}**"
            winner = "Transformer"
        else:
            x = f"**{x}**"
            winner = "XGBoost"
        lines.append(
            f"| {row['market']} | {row['task']} | {t} | {x} | {winner} | "
            f"{row['transformer_relative_improvement_pct']:+.1f}% |"
        )
    lines.extend((
        "", f"预测MAE上，Transformer赢得 **{summary['transformer_wins']}/12** 个市场×任务组合。",
        "", "## ERCOT 2023同合同盈利", "",
        "| 系统 | 日均净收益（USD/节点·日） | 相对pooled XGBoost | 配对7日块95%区间 |",
        "|---|---:|---:|---|",
    ))
    best = max((row["ercot_2023_mean_daily_net_revenue"] for row in summary["profit_rows"]), default=None)
    for row in summary["profit_rows"]:
        value = f"{row['ercot_2023_mean_daily_net_revenue']:.2f}"
        if best is not None and row["ercot_2023_mean_daily_net_revenue"] == best:
            value = f"**{value}**"
        paired = row["paired_vs_pooled_lomo_xgboost"]
        ci = paired["seven_day_block_bootstrap_95ci"]
        lines.append(
            f"| {row['label']} | {value} | {paired['mean_daily_difference']:+.2f} | "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] |"
        )
    lines.extend((
        "", "MAE胜负与盈利胜负不是同一概念；本表不以12项MAE多数胜出代替盈利结论。",
        "逐节点本地XGBoost看过目标市场标签，只作为实际监督强上界，不属于同数据架构因果对照。",
        "", "## ERCOT 2023盈利风险", "",
        "| 系统 | 正收益日 | 最大回撤 | 去掉最高5日 | 去掉最高10日 | 前半年/后半年 | 最差节点 | 节点Bootstrap 95%区间 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ))
    for row in summary["profit_rows"]:
        risk = row["risk"]
        worst = risk["worst_node_mean_daily_revenue"]
        worst_text = "-" if worst is None else f"{worst:.2f}"
        node_ci = risk["node_bootstrap_mean_95ci"]
        node_ci_text = (
            "-" if node_ci is None else f"[{node_ci[0]:.2f}, {node_ci[1]:.2f}]"
        )
        lines.append(
            f"| {row['label']} | {100*risk['positive_day_rate']:.1f}% | "
            f"{risk['maximum_drawdown_usd_per_node']:.2f} | "
            f"{risk['mean_without_top_5_days']:.2f} | "
            f"{risk['mean_without_top_10_days']:.2f} | "
            f"{risk['first_half_mean']:.2f}/{risk['second_half_mean']:.2f} | "
            f"{worst_text} | {node_ci_text} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="data/results/fair_xgboost_comparison_v1"
    )
    args = parser.parse_args()
    rows = build_rows(ROOT)
    summary = {
        "experiment": "fair_pooled_lomo_xgboost_comparison_v1",
        "status": "2023_inner_zero_shot_same_training_markets_and_years",
        "test_2025_read": False,
        "forecast_rows": rows,
        "transformer_wins": sum(row["winner"] == "transformer" for row in rows),
        "xgboost_wins": sum(row["winner"] == "xgboost" for row in rows),
        "profit_rows": profit_rows(ROOT),
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({
        "event": "fair_xgboost_comparison_complete",
        "transformer_wins": summary["transformer_wins"],
        "xgboost_wins": summary["xgboost_wins"],
        "output_dir": args.output_dir,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
