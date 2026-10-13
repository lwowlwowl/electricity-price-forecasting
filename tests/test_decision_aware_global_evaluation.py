import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.global_evaluation import (  # noqa: E402
    align_node_task_predictions,
    enumerate_fusion_triplets,
)
from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _aggregate,
    _process_safe,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
)


def _records():
    da_dates = np.asarray(["2023-01-01", "2023-01-02", "2023-01-03"])
    rt_dates = np.asarray(["2023-01-02", "2023-01-03"])
    truth = np.arange(48, dtype=np.float32).reshape(2, 24)
    da_truth = np.concatenate([np.full((1, 24), -1, np.float32), truth])
    stamps = np.asarray([
        [f"{day}T{hour:02d}:00:00+00:00" for hour in range(24)]
        for day in rt_dates
    ])
    return (
        {
            "dates": da_dates,
            "prediction": da_truth + 1,
            "target_da": da_truth,
            "target_rt": da_truth,
        },
        {
            "dates": rt_dates,
            "prediction": truth + 2,
            "target_rt": truth,
        },
        {
            "dates": rt_dates,
            "prediction": np.repeat((truth + 3)[..., None], 4, axis=2),
            "target_rt": np.repeat(truth[..., None], 4, axis=2),
            "executed_timestamps": stamps,
            "timezone": "UTC",
        },
    )


def test_enumerates_all_eight_f0_f1_triplets():
    combinations = enumerate_fusion_triplets()
    assert len(combinations) == 8
    assert len({tuple(item.values()) for item in combinations}) == 8


def test_aligns_three_tasks_by_delivery_date_not_array_position():
    aligned = align_node_task_predictions(*_records())
    assert aligned["dates"] == ["2023-01-02", "2023-01-03"]
    assert tuple(aligned["rt_price_forecasts"].shape) == (48, 4)
    assert tuple(aligned["realized_da_price"].shape) == (48,)
    assert len(aligned["local_timestamps"]) == 48
    assert float(aligned["da_price_forecasts"][np.datetime64("2023-01-02").astype(object)][0]) == 1.0


def test_rejects_truth_mismatch_between_tasks():
    da, rt_at_da, rt = _records()
    rt["target_rt"] = rt["target_rt"].copy()
    rt["target_rt"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="真实价格不一致"):
        align_node_task_predictions(da, rt_at_da, rt)


def test_continuous_evaluator_converts_tensors_before_process_pool():
    original = {
        "curve": torch.arange(4, dtype=torch.float32),
        "nested": {"value": torch.tensor([5.0])},
        "unchanged": "UTC",
    }
    converted = _process_safe(original)
    assert isinstance(converted["curve"], np.ndarray)
    assert isinstance(converted["nested"]["value"], np.ndarray)
    assert converted["curve"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert converted["unchanged"] == "UTC"


def test_load_archive_uses_requested_outer_phase(tmp_path):
    path = tmp_path / "candidate_da_outer_seed0.npz"
    np.savez_compressed(
        path,
        nodes=np.asarray(["N1"]),
        dates=np.asarray(["2024-01-01"]),
        prediction=np.zeros((1, 1, 24), dtype=np.float32),
        target_da=np.ones((1, 1, 24), dtype=np.float32),
        target_rt=np.ones((1, 1, 24), dtype=np.float32),
        metadata=json.dumps({
            "family": "candidate", "task": "da", "phase": "outer",
        }),
    )
    loaded = _load_archive(tmp_path, "candidate", "da", "outer")
    assert loaded["metadata"]["phase"] == "outer"
    assert loaded["prediction"].shape == (1, 1, 24)


def test_aggregate_reports_distribution_costs_and_reconstruction():
    nodes = {}
    for offset, node in enumerate(("N1", "N2")):
        daily = {
            f"2024-01-{day:02d}": float(day + 5 + offset)
            for day in range(1, 7)
        }
        nodes[node] = {
            "daily_net_revenue": daily,
            "days": 6,
            "actual_throughput_mwh": 12.0,
            "soc_violation_count": offset,
            "revenue_components": {
                "da_leg": 60.0 + 6.0 * offset,
                "rt_deviation_leg": 6.0,
                "degradation_cost": 12.0,
                "deviation_penalty": 3.0,
            },
        }
    result = _aggregate(nodes)
    assert result["macro_mean_daily_revenue"] == pytest.approx(9.0)
    assert result["macro_median_daily_revenue"] == pytest.approx(9.0)
    assert result["macro_mean_daily_actual_throughput_mwh"] == pytest.approx(2.0)
    assert result["total_soc_violation_count"] == 1
    components = result["macro_mean_daily_revenue_components"]
    assert components == pytest.approx({
        "da_leg": 10.5,
        "rt_deviation_leg": 1.0,
        "degradation_cost": 2.0,
        "deviation_penalty": 0.5,
    })
    assert result["macro_mean_daily_revenue_reconstructed"] == pytest.approx(9.0)
