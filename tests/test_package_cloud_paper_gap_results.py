import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from scripts.decision_aware import package_cloud_paper_gap_results as module


def _statuses(root: Path, complete: bool = True) -> None:
    for relative in module.STATUS_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        state = "complete" if complete else "starting"
        path.write_text(f"2026-10-16T04:00:00+08:00\tall\t{state}\n", encoding="utf-8")


def test_validate_complete_rejects_running_queue(tmp_path):
    _statuses(tmp_path, complete=False)
    with pytest.raises(RuntimeError, match="尚未完成"):
        module.validate_complete(tmp_path)


def test_bundle_contains_manifest_and_excludes_latest(tmp_path, monkeypatch):
    _statuses(tmp_path)
    result_dir = tmp_path / "data/results/universal_lomo_multiseed_v1"
    result_dir.mkdir(parents=True)
    (result_dir / "prediction.npz").write_bytes(b"prediction")
    checkpoint = (
        tmp_path
        / "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed1"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "best.pt").write_bytes(b"best")
    (checkpoint / "summary.json").write_text("{}", encoding="utf-8")
    (checkpoint / "latest.pt").write_bytes(b"latest")
    monkeypatch.setattr(
        module,
        "RESULT_GROUPS",
        (
            "data/results/cloud_universal_lomo_multiseed_v1",
            "data/results/cloud_universal_lomo_xgboost_v1",
            "data/results/cloud_universal_ercot_ablation_v1",
            "data/results/universal_lomo_multiseed_v1",
        ),
    )
    output = tmp_path / "data/results/bundle.tar.gz"
    manifest = tmp_path / "data/results/manifest.json"
    report = module.build_bundle(tmp_path, output, manifest)
    assert report["archive_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    paths = {item["path"] for item in payload["files"]}
    assert any(path.endswith("best.pt") for path in paths)
    assert not any(path.endswith("latest.pt") for path in paths)
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "data/results/manifest.json" in names
