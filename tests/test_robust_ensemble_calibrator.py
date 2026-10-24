from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from decision_aware.robust_ensemble import (
    EconomicEnsembleCalibrator,
    EnsembleCalibratorConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_initialization_is_exact_equal_weight_ensemble() -> None:
    generator = torch.Generator().manual_seed(17)
    da = torch.randn(5, 3, 24, generator=generator)
    rt_at_da = torch.randn(5, 3, 24, generator=generator)
    rt = torch.randn(5, 3, 24, 7, generator=generator)
    calibrator = EconomicEnsembleCalibrator()

    actual_da, actual_rt_at_da, actual_rt = calibrator(da, rt_at_da, rt)

    torch.testing.assert_close(actual_da, da.mean(dim=1))
    torch.testing.assert_close(actual_rt_at_da, rt_at_da.mean(dim=1))
    torch.testing.assert_close(actual_rt, rt.mean(dim=1))
    torch.testing.assert_close(
        calibrator.weights(), torch.full((3, 3), 1.0 / 3.0)
    )


def test_corrections_are_bounded_and_gradients_reach_all_parameters() -> None:
    config = EnsembleCalibratorConfig(
        maximum_center_bias=2.0,
        maximum_spread_bias=3.0,
        maximum_spread_log_scale=0.2,
        maximum_rt_bias=4.0,
        maximum_rt_log_scale=0.1,
    )
    calibrator = EconomicEnsembleCalibrator(config)
    with torch.no_grad():
        for parameter in calibrator.parameters():
            parameter.fill_(100.0)
    interpretation = calibrator.interpretation()
    assert max(np.abs(interpretation["center_bias_usd_per_mwh"])) <= 2.0
    assert max(np.abs(interpretation["spread_bias_usd_per_mwh"])) <= 3.0
    assert max(np.abs(interpretation["rt_bias_usd_per_mwh"])) <= 4.0
    assert max(interpretation["spread_scale"]) <= np.exp(0.2) + 1e-6
    assert max(interpretation["rt_scale"]) <= np.exp(0.1) + 1e-6

    calibrator.zero_grad(set_to_none=True)
    with torch.no_grad():
        for parameter in calibrator.parameters():
            parameter.zero_()
        calibrator.seed_logits.copy_(torch.tensor([
            [0.2, -0.1, 0.0], [0.0, 0.1, -0.2], [-0.1, 0.3, 0.0]
        ]))
    da = torch.randn(2, 3, 24)
    rt_at_da = torch.randn(2, 3, 24)
    rt = torch.randn(2, 3, 24, 4)
    outputs = calibrator(da, rt_at_da, rt)
    loss = sum(output.square().mean() for output in outputs)
    loss.backward()
    assert all(parameter.grad is not None for parameter in calibrator.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in calibrator.parameters())


def test_balanced_indices_draw_exact_count_per_market() -> None:
    train = _load_script(
        "train_global_robust_ensemble_calibrator",
        "scripts/decision_aware/train_global_robust_ensemble_calibrator.py",
    )

    class Dataset:
        entities = [("PJM", "p0"), ("CAISO", "c0"), ("PJM", "p1"), ("NYISO", "n0")]
        offsets = [0, 7, 16, 23, 31]

    indices = train._balanced_indices(Dataset(), per_market=5, seed=29)
    assert len(indices) == 15
    assert len(set(indices)) == 15
    counts = {"CAISO": 0, "NYISO": 0, "PJM": 0}
    for index in indices:
        entity = max(i for i, start in enumerate(Dataset.offsets[:-1]) if start <= index)
        counts[Dataset.entities[entity][0]] += 1
    assert counts == {"CAISO": 5, "NYISO": 5, "PJM": 5}


def test_balanced_final_indices_are_disjoint_and_exact() -> None:
    train = _load_script(
        "train_global_robust_ensemble_calibrator_disjoint",
        "scripts/decision_aware/train_global_robust_ensemble_calibrator.py",
    )

    class Dataset:
        entities = [
            ("CAISO", "c0"),
            ("ERCOT", "e0"),
            ("NYISO", "n0"),
            ("PJM", "p0"),
        ]
        offsets = [0, 20, 40, 60, 80]

    selected_train, selected_validation = train._balanced_disjoint_indices(
        Dataset(), train_per_market=5, validation_per_market=3, seed=31
    )
    assert len(selected_train) == 20
    assert len(selected_validation) == 12
    assert not (set(selected_train) & set(selected_validation))


def test_apply_zero_calibrator_writes_member_mean(tmp_path: Path) -> None:
    apply_module = _load_script(
        "apply_robust_ensemble_calibrator",
        "scripts/decision_aware/apply_robust_ensemble_calibrator.py",
    )
    members = ("member0", "member1", "member2")
    nodes = np.asarray(["node-a"])
    dates = np.asarray([["2023-01-01", "2023-01-02"]])
    for task in ("da", "rt_at_da", "rt"):
        for member_index, member in enumerate(members):
            prediction = np.full(
                (1, 1, 24, 2) if task == "rt" else (1, 2, 24),
                member_index + 1,
                dtype=np.float32,
            )
            task_dates = dates[:, :1] if task == "rt" else dates
            np.savez_compressed(
                tmp_path / f"{member}_{task}_inner_seed0.npz",
                nodes=nodes,
                dates=task_dates,
                prediction=prediction,
                target_da=np.zeros((1, 2, 24), dtype=np.float32),
                target_rt=np.zeros((1, 2, 24), dtype=np.float32),
            )
    calibrator = EconomicEnsembleCalibrator()
    checkpoint = tmp_path / "calibrator.pt"
    torch.save({
        "source_validation_improved": True,
        "members": list(members),
        "calibrator_config": calibrator.config.to_dict(),
        "model_state": calibrator.state_dict(),
    }, checkpoint)

    apply_module.apply(tmp_path, checkpoint, "combined")

    for task in ("da", "rt_at_da", "rt"):
        with np.load(tmp_path / f"combined_{task}_inner_seed0.npz") as archive:
            np.testing.assert_allclose(archive["prediction"], 2.0)

