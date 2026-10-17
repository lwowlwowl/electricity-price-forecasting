"""把云端status.tsv转换为可审计的阶段墙钟表，不估算缺失的显存或推理时间。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_status(path: Path) -> list[dict]:
    pending: dict[tuple[str, str], list[datetime]] = {}
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        timestamp_text, stage, event = raw.split("\t")
        timestamp = datetime.fromisoformat(timestamp_text)
        if event in {"starting", "training_starting", "holdout_starting", "preparing"}:
            kind = (
                event.removesuffix("_starting")
                if event.endswith("_starting")
                else "run"
                if event == "starting"
                else event
            )
            pending.setdefault((stage, kind), []).append(timestamp)
            continue
        if event not in {"complete", "training_complete", "holdout_complete"}:
            continue
        kind = event.removesuffix("_complete") if event.endswith("_complete") else "run"
        key = (stage, kind)
        starts = pending.get(key, [])
        if not starts:
            continue
        started = starts.pop(0)
        rows.append({
            "stage": stage,
            "kind": kind,
            "started_at": started.isoformat(),
            "completed_at": timestamp.isoformat(),
            "elapsed_seconds": (timestamp - started).total_seconds(),
        })
    return rows


def build_summary(inputs: list[Path]) -> dict:
    files = []
    for path in inputs:
        rows = parse_status(path)
        files.append({
            "path": path.as_posix(),
            "rows": rows,
            "paired_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
        })
    return {
        "experiment": "paper_compute_cost_status_audit_v1",
        "source": "recorded wall-clock timestamps; no hardware-normalized estimates",
        "files": files,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# 论文补实验墙钟审计",
        "",
        "> 仅汇总status.tsv中能够首尾配对的实测阶段；不推断峰值显存、推理延迟或缺失阶段。",
        "",
        "| 来源 | 阶段 | 类型 | 墙钟 |",
        "|---|---|---|---:|",
    ]
    for file_record in summary["files"]:
        source = Path(file_record["path"]).parent.name
        for row in file_record["rows"]:
            seconds = row["elapsed_seconds"]
            lines.append(
                f"| {source} | {row['stage']} | {row['kind']} | "
                f"{int(seconds // 3600)}时{int(seconds % 3600 // 60)}分{int(seconds % 60)}秒 |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--output-dir", default="data/results/paper_compute_cost_audit_v1"
    )
    args = parser.parse_args()
    inputs = [ROOT / value for value in args.inputs]
    summary = build_summary(inputs)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
