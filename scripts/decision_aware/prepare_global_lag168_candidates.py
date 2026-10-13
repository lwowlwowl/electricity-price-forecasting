#!/usr/bin/env python
"""准备 lag168 DA/RT-at-DA 候选的完整结算评估清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("da", "rt_at_da", "rt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone_archive(
    source: Path, destination: Path, family: str, task: str, root: Path
) -> None:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata.update({
        "family": family,
        "task": task,
        "cloned_template_source": source.relative_to(root).as_posix(),
    })
    arrays["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def _select(run_dir: Path, keep: int, root: Path) -> list[dict]:
    manifest = json.loads(
        (run_dir / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    candidates = []
    seen = set()
    for item in sorted(
        manifest, key=lambda value: (float(value["macro_mae"]), int(value["epoch"]))
    ):
        checkpoint = run_dir / item["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        digest = _sha256(checkpoint)
        if digest in seen:
            continue
        seen.add(digest)
        candidates.append({
            "epoch": int(item["epoch"]),
            "macro_mae": float(item["macro_mae"]),
            "checkpoint": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": digest,
        })
        if len(candidates) == keep:
            break
    if len(candidates) < keep:
        raise ValueError(f"{run_dir}只有{len(candidates)}个不重复候选，少于keep={keep}")
    return candidates


def prepare(
    *,
    root: Path,
    da_run_dir: Path,
    rt_at_da_run_dir: Path,
    template_dir: Path,
    prediction_dir: Path,
    keep: int,
) -> dict:
    if keep < 1:
        raise ValueError("keep必须至少为1")
    selected = {
        "da": _select(da_run_dir, keep, root),
        "rt_at_da": _select(rt_at_da_run_dir, keep, root),
    }
    families: dict[str, dict[str, str]] = {}
    export_groups = {"da": [], "rt_at_da": []}

    for task, prefix in (("da", "lagda"), ("rt_at_da", "lagrtda")):
        for rank, item in enumerate(selected[task]):
            family = f"{prefix}{rank}"
            families[family] = {task: item["checkpoint"]}
            export_groups[task].append(family)
            for filler_task, template_family in (
                ("da", "c0"), ("rt_at_da", "c1"), ("rt", "c0")
            ):
                if filler_task == task:
                    continue
                _clone_archive(
                    template_dir / f"{template_family}_{filler_task}_inner_seed0.npz",
                    prediction_dir / f"{family}_{filler_task}_inner_seed0.npz",
                    family,
                    filler_task,
                    root,
                )

    # 冻结当前稳定版的滚动RT；stable家族也作为可直接比较的完整系统。
    stable_templates = {"da": "c0", "rt_at_da": "c1", "rt": "c0"}
    for task, template_family in stable_templates.items():
        _clone_archive(
            template_dir / f"{template_family}_{task}_inner_seed0.npz",
            prediction_dir / f"stable_{task}_inner_seed0.npz",
            "stable",
            task,
            root,
        )

    triplets = [
        f"da_lagda{da_rank}__rt_at_da_lagrtda{rtda_rank}__rt_stable"
        for da_rank in range(keep)
        for rtda_rank in range(keep)
    ]
    triplets.append("da_stable__rt_at_da_stable__rt_stable")
    return {
        "experiment": "global_lag168_candidate_pool_v1",
        "phase": "inner",
        "selection_rule": (
            "retain unique checkpoints by 2023 node-macro MAE; choose the final "
            "DA/RT-at-DA pair only with frozen full dual settlement"
        ),
        "selected_by_task": selected,
        "families": families,
        "export_groups": export_groups,
        "prediction_dir": prediction_dir.relative_to(root).as_posix(),
        "frozen_rt_family": "stable/c0",
        "requested_triplets": triplets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--da-run-dir",
        default="data/checkpoints/global_ercot_lag168_da_mae_v1/da_inner_seed0",
    )
    parser.add_argument(
        "--rt-at-da-run-dir",
        default=(
            "data/checkpoints/global_ercot_lag168_rt_at_da_pseudohuber_v1/"
            "rt_at_da_inner_seed0"
        ),
    )
    parser.add_argument(
        "--template-dir", default="data/results/global_f0_f1_stable_v2_predictions"
    )
    parser.add_argument(
        "--prediction-dir", default="data/results/global_lag168_predictions_v1"
    )
    parser.add_argument(
        "--output", default="data/results/global_lag168_candidate_manifest_v1.json"
    )
    parser.add_argument("--keep", type=int, default=3)
    args = parser.parse_args()

    report = prepare(
        root=ROOT,
        da_run_dir=ROOT / args.da_run_dir,
        rt_at_da_run_dir=ROOT / args.rt_at_da_run_dir,
        template_dir=ROOT / args.template_dir,
        prediction_dir=ROOT / args.prediction_dir,
        keep=args.keep,
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "lag168_candidate_manifest_saved",
        "path": output.relative_to(ROOT).as_posix(),
        "triplet_count": len(report["requested_triplets"]),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
