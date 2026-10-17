#!/usr/bin/env python
"""增量汇总四市场cutoff-safe LEAR与现有零样本基线。

LEAR只适用于两个24小时任务；脚本不会为滚动4小时RT伪造结果，也不会读取2025。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.summarize_fair_xgboost_comparison import macro_mae


MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")
TASKS = ("da", "rt_at_da")
EXPECTED_NODE_COUNTS = {"CAISO": 3, "ERCOT": 15, "NYISO": 11, "PJM": 35}


def lear_result(root: Path, market: str, task: str) -> dict:
    base = root / "data/results/global_baselines_v1"
    # ERCOT是该入口最早支持的默认市场，为保持历史目录兼容不增加ercot层。
    path = (
        base / f"{task}_inner/summary.json"
        if market == "ERCOT"
        else base / market.lower() / f"{task}_inner/summary.json"
    )
    expected_nodes = EXPECTED_NODE_COUNTS[market]
    cached_nodes = len(list(path.parent.glob("lear_*_seed0.npz")))
    pending = {
        "status": "pending",
        "mae": None,
        "elapsed_seconds": None,
        "cached_nodes": cached_nodes,
        "expected_nodes": expected_nodes,
    }
    if not path.is_file():
        return pending
    report = json.loads(path.read_text(encoding="utf-8"))
    item = report.get("models", {}).get("lear")
    if (
        not isinstance(item, dict)
        or not item.get("complete")
        or int(item.get("node_count", 0)) != expected_nodes
        or cached_nodes != expected_nodes
    ):
        return pending
    return {
        "status": "complete",
        "mae": float(item["macro_mae"]),
        "elapsed_seconds": float(item.get("elapsed_seconds", 0.0)),
        "nodes": int(item["node_count"]),
        "cached_nodes": cached_nodes,
        "expected_nodes": expected_nodes,
        "protocol": item.get("protocol"),
    }


def build_rows(root: Path) -> list[dict]:
    rows = []
    for market in MARKETS:
        for task in TASKS:
            lear = lear_result(root, market, task)
            transformer = macro_mae(
                root / "data/results" / f"universal_lomo_{task}_v1"
                / f"{market.lower()}_{task}_v1.npz"
            )
            pooled = macro_mae(
                root / "data/results/universal_lomo_xgboost_v1"
                / f"holdout_{market.lower()}_{task}_inner_seed0.npz"
            )
            available = {"transformer": transformer, "pooled_xgboost": pooled}
            if lear["mae"] is not None:
                available["lear"] = lear["mae"]
            winner = min(available, key=available.get)
            rows.append({
                "market": market,
                "task": task,
                "lear": lear,
                "transformer_mae": transformer,
                "pooled_lomo_xgboost_mae": pooled,
                "winner_among_available": winner,
            })
    return rows


def render(summary: dict) -> str:
    lines = [
        "# 四市场逐节点本地cutoff-safe LEAR增量汇总", "",
        "> LEAR使用目标节点2020—2022历史并每天按截止时点滚动重估，只用于DA和RT-at-DA；它是目标市场监督强基线，不是同数据零样本架构对照。未读取2025。", "",
        "| 市场 | 任务 | 本地LEAR MAE | 无ID Transformer MAE | pooled LOMO XGBoost MAE | 当前可用最低MAE | LEAR墙钟 |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    labels = {
        "lear": "本地LEAR", "transformer": "Transformer",
        "pooled_xgboost": "pooled XGBoost",
    }
    for row in summary["rows"]:
        winner = row["winner_among_available"]
        lear = row["lear"]
        lear_text = "-" if lear["mae"] is None else f"{lear['mae']:.2f}"
        if lear["mae"] is None and lear.get("cached_nodes", 0):
            lear_text = (
                f"-（缓存{lear['cached_nodes']}/{lear['expected_nodes']}节点）"
            )
        transformer = f"{row['transformer_mae']:.2f}"
        pooled = f"{row['pooled_lomo_xgboost_mae']:.2f}"
        if winner == "lear":
            lear_text = f"**{lear_text}**"
        elif winner == "transformer":
            transformer = f"**{transformer}**"
        else:
            pooled = f"**{pooled}**"
        elapsed = (
            "-" if lear["elapsed_seconds"] is None
            else f"{lear['elapsed_seconds'] / 60:.1f}分"
        )
        lines.append(
            f"| {row['market']} | {row['task']} | {lear_text} | {transformer} | "
            f"{pooled} | {labels[winner]} | {elapsed} |"
        )
    lines.extend((
        "",
        f"当前完成 **{summary['complete_rows']}/8** 个市场×任务；全部完成前不写总排名。",
        "滚动RT不在LEAR定义内，保留lag-168、pooled XGBoost、TSFM和当前Transformer对照。",
    ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="data/results/lear_multimarket_summary_v1"
    )
    args = parser.parse_args()
    rows = build_rows(ROOT)
    summary = {
        "experiment": "cutoff_safe_lear_multimarket_incremental_summary_v1",
        "status": "complete" if all(
            row["lear"]["status"] == "complete" for row in rows
        ) else "running_partial",
        "test_2025_read": False,
        "complete_rows": sum(
            row["lear"]["status"] == "complete" for row in rows
        ),
        "rows": rows,
    }
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(render(summary), encoding="utf-8")
    print(json.dumps({
        "event": "lear_summary_complete",
        "status": summary["status"],
        "complete_rows": summary["complete_rows"],
        "output_dir": args.output_dir,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
