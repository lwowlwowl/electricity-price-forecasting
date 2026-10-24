from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.positive_spike_probability_gate import (
    THRESHOLD_GRID,
    expected_calibration_error,
    fit_platt_calibrator,
    metrics_with_optional_threshold,
    node_equal_metrics,
    pe_market_gate,
    select_fpr_budget_threshold,
)


def test_threshold_grid_is_frozen() -> None:
    assert len(THRESHOLD_GRID) == 99
    assert THRESHOLD_GRID[0] == 0.01
    assert THRESHOLD_GRID[-1] == 0.99
    assert np.allclose(np.diff(THRESHOLD_GRID), 0.01)


def test_threshold_requires_each_source_market_fpr_and_recall() -> None:
    probability = np.asarray([0.9, 0.8, 0.04, 0.03, 0.7, 0.6, 0.02, 0.01])
    truth = np.asarray([True, True, False, False, True, True, False, False])
    markets = np.asarray(["A"] * 4 + ["B"] * 4)
    selected, rows = select_fpr_budget_threshold(
        probability, truth, markets, candidates=(0.05, 0.50, 0.95)
    )
    assert selected == 0.05
    assert rows[0]["feasible"] is True


def test_no_feasible_threshold_is_explicit_failure() -> None:
    probability = np.asarray([0.9, 0.8, 0.7, 0.6])
    truth = np.asarray([False, False, True, True])
    markets = np.asarray(["A", "A", "B", "B"])
    selected, _ = select_fpr_budget_threshold(
        probability, truth, markets, candidates=(0.5, 0.95)
    )
    assert selected is None
    metrics = metrics_with_optional_threshold(probability, truth, selected)
    assert metrics["recall"] is None
    assert metrics["brier"] >= 0.0


def test_platt_calibrator_outputs_probabilities() -> None:
    score = np.asarray([-3, -2, -1, 1, 2, 3], dtype=float)
    truth = np.asarray([False, False, False, True, True, True])
    markets = np.asarray(["A", "A", "A", "B", "B", "B"])
    fitted = fit_platt_calibrator(score, truth, markets)
    probability = fitted.predict(score)
    assert np.all((probability > 0) & (probability < 1))
    assert np.all(np.diff(probability) > 0)


def test_node_equal_metrics_and_gate() -> None:
    probability = np.asarray([0.9, 0.1, 0.8, 0.2])
    truth = np.asarray([True, False, True, False])
    node_ids = np.asarray([0, 0, 1, 1])
    equal, rows = node_equal_metrics(
        probability, truth, node_ids, np.asarray(["N0", "N1"]), 0.5
    )
    assert len(rows) == 2
    assert equal["false_positive_rate"] == 0.0
    gate = pe_market_gate(
        metrics_with_optional_threshold(probability, truth, 0.5),
        equal,
        {"ci95": [0.1, 0.2]},
        0.5,
    )
    assert gate["passed"] is True


def test_expected_calibration_error() -> None:
    rows = [
        {"samples": 5, "mean_probability": 0.2, "observed_rate": 0.4},
        {"samples": 5, "mean_probability": 0.8, "observed_rate": 0.8},
    ]
    assert np.isclose(expected_calibration_error(rows), 0.1)
