import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware import summarize_universal_lomo_multiseed as module


def test_macro_mae_is_node_equal_not_element_pooling():
    target = np.zeros((2, 1, 2), dtype=np.float32)
    prediction = np.array([[[1.0, 1.0]], [[3.0, 3.0]]], dtype=np.float32)
    assert module._macro_mae(target, prediction) == pytest.approx(2.0)


def test_summarize_rejects_mismatched_seed_truth(tmp_path, monkeypatch):
    result_root = tmp_path / "results"
    result_root.mkdir()
    seed0 = tmp_path / "seed0.npz"
    common = {
        "nodes": np.array(["N"]),
        "dates": np.array(["2023-01-01"]),
        "prediction": np.zeros((1, 1, 2), dtype=np.float32),
    }
    np.savez(seed0, **common, target=np.zeros((1, 1, 2), dtype=np.float32))
    np.savez(
        result_root / "ercot_da_inner_seed1.npz",
        **common,
        target=np.ones((1, 1, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        module,
        "_seed_path",
        lambda root, holdout, task, seed: seed0 if seed == 0 else root / f"ercot_da_inner_seed{seed}.npz",
    )
    with pytest.raises(ValueError, match="真值不一致"):
        module.summarize(result_root, ("ercot",), ("da",), (0, 1))


def test_summarize_writes_fixed_equal_weight_ensemble(tmp_path, monkeypatch):
    result_root = tmp_path / "results"
    ensemble_root = tmp_path / "ensemble"
    result_root.mkdir()
    paths = {}
    for seed, value in enumerate((1.0, 2.0, 3.0)):
        path = tmp_path / f"seed{seed}.npz"
        np.savez(
            path,
            nodes=np.array(["N"]),
            dates=np.array(["2023-01-01"]),
            prediction=np.full((1, 1, 2), value, dtype=np.float32),
            target=np.zeros((1, 1, 2), dtype=np.float32),
            target_da=np.ones((1, 1, 2), dtype=np.float32),
        )
        paths[seed] = path
    monkeypatch.setattr(
        module,
        "_seed_path",
        lambda root, holdout, task, seed: paths[seed],
    )
    report = module.summarize(
        result_root,
        ("ercot",),
        ("rt_at_da",),
        (0, 1, 2),
        ensemble_output_root=ensemble_root,
    )
    path = ensemble_root / "ercot_rt_at_da_inner_ensemble.npz"
    assert path.exists()
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["prediction"], 2.0)
        np.testing.assert_allclose(archive["target_da"], 1.0)
    assert report["results"]["ercot_rt_at_da"][
        "equal_weight_ensemble_path"
    ].endswith("ercot_rt_at_da_inner_ensemble.npz")
