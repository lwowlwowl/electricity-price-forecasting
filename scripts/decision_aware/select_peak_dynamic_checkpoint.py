#!/usr/bin/env python
"""按预注册源市场规则冻结DA局部动态checkpoint。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.peak_dynamic_selection import (  # noqa: E402
    select_peak_dynamic_candidate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", default="selected_peak_dynamic.pt")
    parser.add_argument("--summary", default="peak_dynamic_selection.json")
    parser.add_argument("--mae-guard-fraction", type=float, default=0.02)
    args = parser.parse_args()

    checkpoint_dir = ROOT / args.checkpoint_dir
    candidate_paths = sorted(checkpoint_dir.glob("candidate_epoch_*.pt"))
    if not candidate_paths:
        raise FileNotFoundError(f"没有候选checkpoint: {checkpoint_dir}")
    records: list[dict] = []
    paths_by_epoch: dict[int, Path] = {}
    for path in candidate_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("task") != "da" or checkpoint.get("phase") != "inner":
            raise ValueError(f"候选不是DA inner checkpoint: {path}")
        epoch = int(checkpoint["epoch"])
        if epoch in paths_by_epoch:
            raise ValueError(f"重复epoch候选: {epoch}")
        history = [
            item for item in checkpoint.get("history", [])
            if int(item.get("epoch", -1)) == epoch
        ]
        if len(history) != 1:
            raise ValueError(f"候选历史缺少epoch {epoch}: {path}")
        records.append(history[0])
        paths_by_epoch[epoch] = path

    selection = select_peak_dynamic_candidate(
        records, mae_guard_fraction=args.mae_guard_fraction
    )
    source = paths_by_epoch[int(selection["epoch"])]
    output = checkpoint_dir / args.output
    summary_path = checkpoint_dir / args.summary
    shutil.copy2(source, output)
    selection.update({
        "source_checkpoint": source.relative_to(ROOT).as_posix(),
        "selected_checkpoint": output.relative_to(ROOT).as_posix(),
        "selected_checkpoint_sha256": _sha256(output),
        "candidate_count": len(candidate_paths),
        "rule": (
            "source-2023 market-macro MAE <= epoch0 * 1.02; then minimum "
            "source-2023 market-macro daily peak-excursion MAE; ties by MAE, epoch"
        ),
    })
    summary_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
