from scripts.decision_aware.prepare_global_taskloss_mixed_manifest import prepare


def _report(names):
    return {
        "families": {
            name: {task: f"{name}-{task}.pt" for task in ("da", "rt_at_da", "rt")}
            for name in names
        }
    }


def test_mixed_manifest_is_limited_to_three_full_families():
    report = prepare(_report(("taskloss0", "taskloss1")), _report(("c0", "c1")))
    assert set(report["families"]) == {"new0", "new1", "stable0"}
    assert report["families"]["new0"]["da"] == "taskloss0-da.pt"
    assert report["families"]["stable0"]["rt"] == "c0-rt.pt"
    assert report["2024_used"] is False
    assert report["2025_used"] is False
