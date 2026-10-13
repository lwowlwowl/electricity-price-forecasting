#!/usr/bin/env python
"""核对严格外层Global DA训练没有用2024做早停、调度或选权重。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inner-root",
        default="data/checkpoints/global_ercot_huber_f1_stable_v2",
    )
    parser.add_argument(
        "--outer-root",
        default="data/checkpoints/global_ercot_huber_f1_outer_frozen_v1",
    )
    parser.add_argument(
        "--output",
        default="data/results/global_da_outer_frozen_training_audit_v1.json",
    )
    args = parser.parse_args()

    records = []
    for seed in (0, 1, 2):
        inner_path = ROOT / args.inner_root / f"da_inner_seed{seed}" / "latest.pt"
        outer_path = ROOT / args.outer_root / f"da_outer_seed{seed}" / "best.pt"
        inner = torch.load(inner_path, map_location="cpu", weights_only=False)
        outer = torch.load(outer_path, map_location="cpu", weights_only=False)
        expected_epoch = int(inner["best_epoch"])
        history = list(outer["history"])
        evaluated_epochs = [
            int(item["epoch"])
            for item in history
            if isinstance(item.get("validation"), dict)
        ]
        expected_lr_after = {
            int(item["epoch"]): float(item["learning_rate_after_scheduler"])
            for item in inner["history"]
            if 1 <= int(item.get("epoch", -1)) <= expected_epoch
        }
        actual_lr_after = {
            int(item["epoch"]): float(item["learning_rate_after_scheduler"])
            for item in history
        }
        metadata = outer.get("frozen_training_plan", {})
        checks = {
            "outer_phase": outer.get("phase") == "outer",
            "same_seed": int(outer["config"]["seed"]) == seed,
            "final_epoch_matches_inner_best": int(outer["epoch"]) == expected_epoch,
            "best_is_final_epoch": int(outer["best_epoch"]) == expected_epoch,
            "history_has_exact_epoch_count": len(history) == expected_epoch,
            "outer_evaluated_only_at_final_epoch": evaluated_epochs == [expected_epoch],
            "learning_rate_history_matches_inner": actual_lr_after == expected_lr_after,
            "plan_hash_matches_inner_checkpoint": (
                metadata.get("sha256") == _sha256(inner_path)
            ),
            "metadata_declares_final_only": (
                metadata.get("outer_validation_usage")
                == "final_epoch_evaluation_only"
            ),
        }
        records.append({
            "seed": seed,
            "expected_final_epoch_from_2023": expected_epoch,
            "evaluated_outer_epochs": evaluated_epochs,
            "checks": checks,
            "passed": all(checks.values()),
            "inner_checkpoint": inner_path.relative_to(ROOT).as_posix(),
            "outer_checkpoint": outer_path.relative_to(ROOT).as_posix(),
        })

    report = {
        "experiment": "global_da_outer_frozen_training_leakage_audit",
        "status": "passed" if all(item["passed"] for item in records) else "failed",
        "2025_used": False,
        "records": records,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
