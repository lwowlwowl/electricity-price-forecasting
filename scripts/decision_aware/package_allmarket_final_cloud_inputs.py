#!/usr/bin/env python
"""打包最终云端两阶段所需的冻结代码、协议、checkpoint与本地XGBoost缓存。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = (
    Path("data/results/allmarket_final_2025_preregistration_v1/protocol.json"),
    Path("data/results/allmarket_final_2025_baseline_preregistration_v1/protocol.json"),
)
TASK_CHECKPOINTS = {
    "da": "data/checkpoints/allmarket_final_da_mae_v1/da_final_seed{seed}/best.pt",
    "rt_at_da": "data/checkpoints/allmarket_final_rt_at_da_pseudohuber_v1/rt_at_da_final_seed{seed}/best.pt",
    "rt": "data/checkpoints/allmarket_final_rt_mae_v1/rt_final_seed{seed}/best.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_files() -> set[Path]:
    files: set[Path] = set()
    for rel in PROTOCOLS:
        path = ROOT / rel
        payload = json.loads(path.read_text(encoding="utf-8"))
        files.update(Path(item) for item in payload["file_sha256"])
        files.add(rel)
        files.add(rel.with_suffix(".sha256"))
        for item, expected in payload["file_sha256"].items():
            actual = sha256(ROOT / item)
            if actual != expected:
                raise ValueError(f"冻结文件哈希变化: {item}: {actual} != {expected}")
    return files


def collect(*, require_local_xgboost: bool) -> list[Path]:
    if (ROOT / PROTOCOLS[0].parent / "LOCKED_2025_OPENED.json").exists():
        raise RuntimeError("2025已经打开，拒绝重新制作训练输入包")
    files = _protocol_files()
    files.update({
        Path("scripts/decision_aware/freeze_allmarket_2025_protocol.py"),
        Path("scripts/decision_aware/freeze_allmarket_2025_baseline_protocol.py"),
        Path("configs/nodes.yaml"),
    })
    for pattern in TASK_CHECKPOINTS.values():
        for seed in (0, 1, 2):
            files.add(Path(pattern.format(seed=seed)))

    # Only package the frozen, numerically stable per-node objective.  Older
    # Pseudo-Huber caches may coexist locally after the pre-2025 audit and must
    # never be transferred as candidates for the locked evaluation.
    xgb_checkpoints = sorted(
        (ROOT / "data/checkpoints/global_baselines_v1").glob(
            "**/*_final/xgboost_absolute_*_seed0.joblib"
        )
    )
    xgb_predictions = sorted(
        (ROOT / "data/results/global_baselines_v1").glob(
            "**/*_final/xgboost_absolute_*_seed0.npz"
        )
    )
    xgb_summaries = sorted((ROOT / "data/results/global_baselines_v1").glob("**/*_final/summary.json"))
    if require_local_xgboost:
        if len(xgb_checkpoints) != 192 or len(xgb_predictions) != 192 or len(xgb_summaries) != 12:
            raise RuntimeError(
                "本地逐节点XGBoost尚未完整："
                f"checkpoint={len(xgb_checkpoints)}/192, prediction={len(xgb_predictions)}/192, "
                f"summary={len(xgb_summaries)}/12"
            )
    files.update(path.relative_to(ROOT) for path in xgb_checkpoints)
    files.update(path.relative_to(ROOT) for path in xgb_predictions)
    files.update(path.relative_to(ROOT) for path in xgb_summaries)
    missing = [path.as_posix() for path in files if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"上传包缺少{len(missing)}个文件: {missing[:10]}")
    return sorted(files, key=lambda path: path.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="data/results/allmarket_final_cloud_inputs_20261017.tar.gz"
    )
    parser.add_argument(
        "--manifest", default="data/results/allmarket_final_cloud_inputs_20261017.manifest.json"
    )
    parser.add_argument("--allow-incomplete-local-xgboost", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    files = collect(require_local_xgboost=not args.allow_incomplete_local_xgboost)
    manifest_path = ROOT / args.manifest
    archive_path = ROOT / args.output
    records = [
        {
            "path": path.as_posix(),
            "bytes": (ROOT / path).stat().st_size,
            "sha256": sha256(ROOT / path),
        }
        for path in files
    ]
    manifest = {
        "protocol": "allmarket_final_cloud_input_bundle_v1",
        "locked_2025_read": False,
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "files": records,
    }
    if args.preflight_only:
        print(json.dumps({key: manifest[key] for key in ("protocol", "file_count", "total_bytes")}))
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in files:
            archive.add(ROOT / path, arcname=path.as_posix())
        archive.add(manifest_path, arcname=manifest_path.relative_to(ROOT).as_posix())
    archive_sha = sha256(archive_path)
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{archive_sha}  {archive_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "allmarket_final_cloud_inputs_packaged",
        "archive": archive_path.relative_to(ROOT).as_posix(),
        "archive_sha256": archive_sha,
        "file_count": len(records),
        "total_bytes": manifest["total_bytes"],
    }))


if __name__ == "__main__":
    main()
