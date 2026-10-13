import json
from pathlib import Path

import numpy as np

from scripts.decision_aware.prepare_global_lag168_candidates import prepare


def _run(root: Path, name: str) -> Path:
    run = root / "data" / "checkpoints" / name
    run.mkdir(parents=True)
    entries = []
    for epoch, mae in ((1, 4.0), (2, 2.0), (3, 3.0)):
        checkpoint = run / f"candidate_epoch_{epoch:03d}.pt"
        checkpoint.write_bytes(f"{name}-{epoch}".encode())
        entries.append({
            "epoch": epoch,
            "macro_mae": mae,
            "checkpoint": checkpoint.name,
        })
    (run / "candidate_manifest.json").write_text(
        json.dumps(entries), encoding="utf-8"
    )
    return run


def _templates(root: Path) -> Path:
    directory = root / "data" / "results" / "templates"
    directory.mkdir(parents=True)
    for family, task in (("c0", "da"), ("c1", "rt_at_da"), ("c0", "rt")):
        np.savez_compressed(
            directory / f"{family}_{task}_inner_seed0.npz",
            prediction=np.zeros((1, 1), dtype=np.float32),
            metadata=json.dumps({"family": family, "task": task}),
        )
    return directory


def test_prepare_lag168_candidates_builds_only_requested_cartesian_grid(tmp_path):
    da_run = _run(tmp_path, "da")
    rtda_run = _run(tmp_path, "rtda")
    prediction_dir = tmp_path / "data" / "results" / "predictions"
    report = prepare(
        root=tmp_path,
        da_run_dir=da_run,
        rt_at_da_run_dir=rtda_run,
        template_dir=_templates(tmp_path),
        prediction_dir=prediction_dir,
        keep=2,
    )

    assert [item["epoch"] for item in report["selected_by_task"]["da"]] == [2, 3]
    assert report["export_groups"] == {
        "da": ["lagda0", "lagda1"],
        "rt_at_da": ["lagrtda0", "lagrtda1"],
    }
    assert report["requested_triplets"] == [
        "da_lagda0__rt_at_da_lagrtda0__rt_stable",
        "da_lagda0__rt_at_da_lagrtda1__rt_stable",
        "da_lagda1__rt_at_da_lagrtda0__rt_stable",
        "da_lagda1__rt_at_da_lagrtda1__rt_stable",
        "da_stable__rt_at_da_stable__rt_stable",
    ]
    with np.load(
        prediction_dir / "lagda0_rt_at_da_inner_seed0.npz", allow_pickle=False
    ) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["family"] == "lagda0"
    assert metadata["task"] == "rt_at_da"
