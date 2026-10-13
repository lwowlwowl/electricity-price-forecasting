import json
from pathlib import Path

from scripts.decision_aware.prepare_global_taskloss_candidates import prepare


def _run(root: Path, task: str) -> str:
    relative = f"runs/{task}"
    run = root / relative
    run.mkdir(parents=True)
    rows = []
    for epoch, mae in ((3, 8.0), (1, 7.0), (2, 7.0)):
        checkpoint = run / f"candidate_epoch_{epoch:03d}.pt"
        checkpoint.write_bytes(f"{task}-{epoch}".encode())
        rows.append({
            "epoch": epoch,
            "macro_mae": mae,
            "checkpoint": checkpoint.name,
        })
    (run / "candidate_manifest.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    return relative


def test_prepare_taskloss_candidates_keeps_top_epochs_and_builds_families(tmp_path):
    run_dirs = {task: _run(tmp_path, task) for task in ("da", "rt_at_da", "rt")}
    report = prepare(
        root=tmp_path, run_dirs=run_dirs, keep=2, family_prefix="taskloss"
    )

    assert report["losses"] == {
        "da": "mae", "rt_at_da": "pseudo_huber", "rt": "mae"
    }
    assert report["2024_used"] is False
    assert report["2025_used"] is False
    for task in run_dirs:
        assert [item["epoch"] for item in report["selected_by_task"][task]] == [1, 2]
        assert report["families"]["taskloss0"][task].endswith(
            "candidate_epoch_001.pt"
        )
        assert report["families"]["taskloss1"][task].endswith(
            "candidate_epoch_002.pt"
        )
