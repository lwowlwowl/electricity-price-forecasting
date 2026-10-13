import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware import select_global_stable_candidates as module  # noqa: E402


def test_select_candidates_keeps_overall_top_two_per_task(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    roots = {"f0": "f0", "f1": "f1"}
    for architecture, base in (("f0", 10.0), ("f1", 5.0)):
        for task in module.TASKS:
            run = tmp_path / architecture / f"{task}_inner_seed0"
            run.mkdir(parents=True)
            rows = []
            for epoch, offset in ((1, 0.0), (2, 1.0)):
                checkpoint = run / f"candidate_epoch_{epoch:03d}.pt"
                checkpoint.write_bytes(f"{architecture}-{task}-{epoch}".encode())
                rows.append({
                    "epoch": epoch,
                    "macro_mae": base + offset,
                    "checkpoint": checkpoint.name,
                })
            (run / "candidate_manifest.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
    report = module.select_candidates(roots, keep=2)
    for task in module.TASKS:
        assert [item["architecture"] for item in report["selected_by_task"][task]] == [
            "f1", "f1"
        ]
        assert report["families"]["c0"][task].endswith("candidate_epoch_001.pt")
        assert report["families"]["c1"][task].endswith("candidate_epoch_002.pt")
