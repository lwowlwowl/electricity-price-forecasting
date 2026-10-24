import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.prepare_fixed_family_ensemble import build_ensemble
from scripts.decision_aware.summarize_implicit_qp_multiseed import _label, summarize


def test_fixed_ensemble_averages_predictions_and_preserves_truth(tmp_path):
    for family, value in (("seed0", 1.0), ("seed1", 2.0), ("seed2", 3.0)):
        for task in ("da", "rt_at_da", "rt"):
            arrays = {
                "nodes": np.array(["N"]),
                "dates": np.array(["2023-01-01"]),
                "prediction": np.full((1, 1, 2), value, dtype=np.float32),
                "target_rt": np.ones((1, 1, 2), dtype=np.float32),
                "metadata": json.dumps({"family": family, "task": task}),
            }
            if task == "da":
                arrays["target_da"] = np.zeros((1, 1, 2), dtype=np.float32)
            if task == "rt":
                arrays["executed_timestamps"] = np.array([["2023-01-01T00:00:00Z"]])
                arrays["timezone"] = np.array("UTC")
            np.savez(tmp_path / f"{family}_{task}_inner_seed0.npz", **arrays)

    build_ensemble(tmp_path, ("seed0", "seed1", "seed2"), "ensemble")
    with np.load(tmp_path / "ensemble_da_inner_seed0.npz") as archive:
        np.testing.assert_allclose(archive["prediction"], 2.0)
        np.testing.assert_allclose(archive["target_da"], 0.0)
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["seed_selection_by_profit"] is False
    assert metadata["members"] == ["seed0", "seed1", "seed2"]


def test_fixed_ensemble_supports_explicit_convex_weights(tmp_path):
    for family, value in (("stable", 2.0), ("strong", 8.0)):
        for task in ("da", "rt_at_da", "rt"):
            arrays = {
                "nodes": np.array(["N"]),
                "dates": np.array(["2023-01-01"]),
                "prediction": np.full((1, 1, 2), value, dtype=np.float32),
                "target_rt": np.ones((1, 1, 2), dtype=np.float32),
            }
            if task == "da":
                arrays["target_da"] = np.zeros((1, 1, 2), dtype=np.float32)
            if task == "rt":
                arrays["executed_timestamps"] = np.array([["2023-01-01T00:00:00Z"]])
                arrays["timezone"] = np.array("UTC")
            np.savez(tmp_path / f"{family}_{task}_inner_seed0.npz", **arrays)

    build_ensemble(
        tmp_path,
        ("stable", "strong"),
        "weighted",
        weights=(1.0, 2.0),
    )
    with np.load(tmp_path / "weighted_da_inner_seed0.npz") as archive:
        np.testing.assert_allclose(archive["prediction"], 6.0)
        metadata = json.loads(str(archive["metadata"].item()))
    np.testing.assert_allclose(metadata["weights"], [1.0 / 3.0, 2.0 / 3.0])


def _system(value):
    daily = {f"2023-01-{day:02d}": value + day for day in range(1, 22)}
    return {
        "macro_mean_daily_revenue": float(np.mean(list(daily.values()))),
        "macro_daily_net_revenue": daily,
        "nodes": {
            "A": {"mean_daily_revenue": value},
            "B": {"mean_daily_revenue": value + 1.0},
        },
    }


def test_multiseed_summary_uses_all_fixed_seeds_and_ensemble():
    seeds = ("seed0", "seed1", "seed2")
    systems = {_label(name): _system(10.0 + index) for index, name in enumerate(seeds)}
    systems[_label("ensemble")] = _system(11.5)
    systems[_label("xgbpooled")] = _system(8.0)
    result = summarize({"systems": systems}, seeds, "ensemble", "xgbpooled")
    assert result["seed_selection_by_profit"] is False
    assert result["across_seed_mean_daily_revenue"] == np.mean([21.0, 22.0, 23.0])
    assert result["fixed_equal_weight_ensemble"]["mean_daily_difference"] == 3.5
