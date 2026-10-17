from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.summarize_compute_costs import parse_status


def test_parse_status_pairs_training_and_holdout(tmp_path):
    path = tmp_path / "status.tsv"
    path.write_text(
        "2026-10-15T16:00:00+08:00\tjob\ttraining_starting\n"
        "2026-10-15T16:30:00+08:00\tjob\ttraining_complete\n"
        "2026-10-15T16:30:00+08:00\tjob\tholdout_starting\n"
        "2026-10-15T16:31:15+08:00\tjob\tholdout_complete\n",
        encoding="utf-8",
    )
    rows = parse_status(path)
    assert [(row["kind"], row["elapsed_seconds"]) for row in rows] == [
        ("training", 1800.0),
        ("holdout", 75.0),
    ]


def test_parse_status_ignores_unpaired_complete(tmp_path):
    path = tmp_path / "status.tsv"
    path.write_text(
        "2026-10-15T16:00:00+08:00\tall\tcomplete\n",
        encoding="utf-8",
    )
    assert parse_status(path) == []
