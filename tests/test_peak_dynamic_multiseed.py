from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "decision_aware"))

from audit_peak_dynamic_actions import _counts, _remove_top_days, _state  # noqa: E402
from build_prediction_median_ensemble import build_median  # noqa: E402
from summarize_peak_dynamic_multiseed import _gate  # noqa: E402


def _archive(path: Path, prediction: np.ndarray, seed: int) -> None:
    np.savez_compressed(
        path,
        nodes=np.asarray(["N1"]),
        dates=np.asarray(["2023-01-01"]),
        prediction=prediction.astype(np.float32),
        target=np.zeros_like(prediction, dtype=np.float32),
        metadata=json.dumps({"seed": seed}),
    )


def test_fixed_hourly_median_ensemble(tmp_path: Path) -> None:
    sources = []
    for seed, value in enumerate((1.0, 100.0, 3.0)):
        path = tmp_path / f"seed{seed}.npz"
        _archive(path, np.full((1, 1, 2), value), seed)
        sources.append(path)
    output = tmp_path / "median.npz"
    build_median(sources, output, label="frozen_median")
    with np.load(output, allow_pickle=False) as archive:
        assert np.array_equal(archive["prediction"], np.full((1, 1, 2), 3.0))
        metadata = json.loads(str(archive["metadata"]))
    assert metadata["seeds"] == [0, 1, 2]
    assert metadata["selection_by_target_market"] is False


def test_action_state_and_top_day_removal() -> None:
    values = np.asarray([-1.0, 0.0, 1.0, 1e-8])
    assert _state(values).tolist() == [-1, 0, 1, 0]
    assert _counts(values) == {"charge_hours": 1, "idle_hours": 2, "discharge_hours": 1}
    result = _remove_top_days({str(index): float(index) for index in range(12)})
    assert result["all_days"] == 5.5
    assert result["without_top_1_delta_days"] == 5.0
    assert result["without_top_10_delta_days"] == 0.5


def _phase(deltas: list[float], ensemble_ci: list[float]) -> dict:
    runs = {}
    for seed, delta in enumerate(deltas):
        runs[str(seed)] = {
            "profit": {"paired_candidate_minus_original": {
                "mean_daily_difference": delta,
                "seven_day_block_bootstrap_95ci": [delta - 1, delta + 1],
            }}
        }
    runs["median"] = {
        "profit": {"paired_candidate_minus_original": {
            "mean_daily_difference": float(np.mean(deltas)),
            "seven_day_block_bootstrap_95ci": ensemble_ci,
        }}
    }
    return {"runs": runs}


def test_stability_gate_requires_two_consistent_seeds_and_positive_ensemble_ci() -> None:
    passed = _gate(_phase([2, 1, -1], [0.1, 2]), _phase([3, 2, -2], [0.2, 3]))
    assert passed["passed"] is True
    assert passed["consistent_positive_seeds_both_years"] == [0, 1]
    failed = _gate(_phase([2, 1, -1], [-0.1, 2]), _phase([3, 2, -2], [0.2, 3]))
    assert failed["passed"] is False
    assert failed["stop_and_do_not_retune"] is True
