import json

from scripts.decision_aware.prepare_global_joint_grid_manifest import combine


def test_combine_joint_grid_manifests(tmp_path):
    directories = []
    for family in ("a", "b"):
        directory = tmp_path / "data" / "checkpoints" / family
        directory.mkdir(parents=True)
        paths = {}
        for task in ("da", "rt_at_da", "rt"):
            checkpoint = directory / f"{task}.pt"
            checkpoint.write_bytes(task.encode())
            paths[task] = checkpoint.relative_to(tmp_path).as_posix()
        (directory / "candidate_manifest.json").write_text(
            json.dumps({"families": {family: paths}}), encoding="utf-8"
        )
        directories.append(directory)

    report = combine(tmp_path, directories)
    assert sorted(report["families"]) == ["a", "b"]
    assert set(report["families"]["a"]) == {"da", "rt_at_da", "rt"}
