#!/usr/bin/env python
"""在云端把本轮三项论文补实验打成可审计归档。

只在三份云端 status.tsv 都以 ``all complete`` 结束时运行。归档仅包含正式结果、
最佳 checkpoint、日志和清单，不包含 latest.pt、缓存或原始市场数据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULT_GROUPS = (
    "data/results/cloud_universal_lomo_multiseed_v1",
    "data/results/universal_lomo_multiseed_v1",
    "data/results/universal_lomo_ercot_multiseed_profit_predictions_v1",
    "data/results/universal_lomo_ercot_multiseed_continuous_mpc_v1.json",
    "data/results/cloud_universal_lomo_xgboost_v1",
    "data/results/universal_lomo_xgboost_v1",
    "data/results/cloud_universal_ercot_ablation_v1",
    "data/results/universal_lomo_ercot_ablation_predictions_v1",
    "data/results/universal_lomo_ercot_ablation_profit_predictions_v1",
    "data/results/universal_lomo_ercot_ablation_continuous_mpc_v1.json",
)
STATUS_FILES = (
    "data/results/cloud_universal_lomo_multiseed_v1/status.tsv",
    "data/results/cloud_universal_lomo_xgboost_v1/status.tsv",
    "data/results/cloud_universal_ercot_ablation_v1/status.tsv",
)


def _checkpoint_files(root: Path) -> Iterable[Path]:
    patterns = (
        "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed[12]/best.pt",
        "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed[12]/summary.json",
        "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/rt_inner_seed[12]/best.pt",
        "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/rt_inner_seed[12]/summary.json",
        "data/checkpoints/universal_lomo_holdout_ercot_rt_at_da_v1/rt_at_da_inner_seed[12]/best.pt",
        "data/checkpoints/universal_lomo_holdout_ercot_rt_at_da_v1/rt_at_da_inner_seed[12]/summary.json",
        "data/checkpoints/universal_lomo_ablation_ercot_all_f0_v1/*_inner_seed0/best.pt",
        "data/checkpoints/universal_lomo_ablation_ercot_all_f0_v1/*_inner_seed0/summary.json",
        "data/checkpoints/universal_lomo_ablation_ercot_all_f1_v1/*_inner_seed0/best.pt",
        "data/checkpoints/universal_lomo_ablation_ercot_all_f1_v1/*_inner_seed0/summary.json",
    )
    for pattern in patterns:
        yield from root.glob(pattern)


def _files_under(root: Path, item: str) -> Iterable[Path]:
    path = root / item
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (candidate for candidate in path.rglob("*") if candidate.is_file())


def collect_files(root: Path) -> list[Path]:
    files = [path for item in RESULT_GROUPS for path in _files_under(root, item)]
    files.extend(_checkpoint_files(root))
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_complete(root: Path) -> None:
    for relative in STATUS_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = [line.strip().split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or len(rows[-1]) < 3 or rows[-1][1:] != ["all", "complete"]:
            raise RuntimeError(f"云端任务尚未完成: {relative}; 最后一行={rows[-1] if rows else None}")


def build_bundle(root: Path, output: Path, manifest: Path) -> dict:
    validate_complete(root)
    files = collect_files(root)
    if not files:
        raise RuntimeError("没有找到可归档文件")
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    payload = {
        "bundle": "paper_gap_cloud_results_20261016_v1",
        "complete_status_files": list(STATUS_FILES),
        "excludes": ["latest.pt", "raw market data", "rebuildable caches"],
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    files_with_manifest = [*files, manifest]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in files_with_manifest:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    payload["archive"] = output.relative_to(root).as_posix()
    payload["archive_sha256"] = _sha256(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/results/paper_gap_cloud_results_20261016_v1.tar.gz",
    )
    parser.add_argument(
        "--manifest",
        default="data/results/paper_gap_cloud_results_20261016_v1.manifest.json",
    )
    args = parser.parse_args()
    report = build_bundle(ROOT, ROOT / args.output, ROOT / args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
