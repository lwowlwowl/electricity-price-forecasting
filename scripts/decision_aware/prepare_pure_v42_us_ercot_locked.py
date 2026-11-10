#!/usr/bin/env python
"""Project strengthened-v4 ERCOT DA prices onto the official-v4 decision cone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.inverse_lp_projection import project_da_to_target_normal_cone  # noqa: E402
from decision_aware.policy import BESSSimulator  # noqa: E402


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _save(payload: dict[str, np.ndarray], path: Path, family: str, task: str) -> None:
    output = dict(payload)
    metadata = json.loads(str(output["metadata"].item()))
    metadata.update(
        {
            "family": family,
            "task": task,
            "phase": "inner",
            "source_family": "v4_official",
            "decision_preserving_projection": task == "da",
        }
    )
    output["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="data/results/pure_v42_memory_us_ercot_predictions_v1"
    )
    parser.add_argument(
        "--output-dir", default="data/results/pure_v42_memory_locked_us_ercot_predictions_v1"
    )
    parser.add_argument(
        "--config", default="configs/decision_aware/global_ercot_huber_v1.yaml"
    )
    args = parser.parse_args()

    source = ROOT / args.input_dir
    base_da = _load(source / "v4_official_da_inner_seed0.npz")
    raw_da = _load(source / "pure_v42_memory_da_inner_seed0.npz")
    base_rtda = _load(source / "v4_official_rt_at_da_inner_seed0.npz")
    if not np.array_equal(base_da["nodes"], base_rtda["nodes"]):
        raise ValueError("DA and RT-at-DA nodes differ")
    if not np.array_equal(base_da["dates"], base_rtda["dates"]):
        raise ValueError("DA and RT-at-DA dates differ")

    cfg = PilotConfig.from_yaml(ROOT / args.config)
    simulator = BESSSimulator(
        cfg.bess_power_mw,
        cfg.bess_energy_mwh,
        cfg.bess_eta,
        cfg.bess_init_soc_frac,
        dt=1.0,
        kappa=cfg.bess_kappa,
        soc_min=cfg.bess_soc_min,
        soc_max=cfg.bess_soc_max,
        e_cyc=cfg.bess_e_cyc,
    )
    initial_soc = cfg.bess_energy_mwh * cfg.bess_init_soc_frac
    dates = base_da["dates"].astype("datetime64[D]")
    projected = raw_da["prediction"].astype(np.float64).copy()
    maximum_gap = 0.0
    maximum_plan_difference = 0.0
    maximum_stationarity = 0.0
    for node, name in enumerate(base_da["nodes"].astype(str)):
        plan_soc = float(initial_soc)
        previous = None
        for day, date in enumerate(dates):
            if previous is None or int((date - previous).astype(int)) != 1:
                plan_soc = float(initial_soc)
            result = project_da_to_target_normal_cone(
                projected[node, day],
                base_da["prediction"][node, day].astype(np.float64),
                base_rtda["prediction"][node, day].astype(np.float64),
                simulator,
                initial_plan_soc_mwh=plan_soc,
                initial_actual_soc_mwh=plan_soc,
            )
            projected[node, day] = result.projected_da
            maximum_gap = max(maximum_gap, abs(result.target_objective_gap))
            maximum_plan_difference = max(
                maximum_plan_difference, result.max_plan_difference_mwh
            )
            maximum_stationarity = max(
                maximum_stationarity, result.max_stationarity_residual
            )
            horizon = len(result.projected_da)
            discharge = result.target_solution[:horizon]
            charge = result.target_solution[horizon : 2 * horizon]
            plan_soc += float(np.sum(-discharge / simulator.eta + charge * simulator.eta))
            plan_soc = float(np.clip(plan_soc, simulator.s_min, simulator.s_max))
            previous = date
        print(json.dumps({"event": "node_projected", "node": name}), flush=True)

    output = ROOT / args.output_dir
    family = "pure_v42_memory_locked"
    da_payload = dict(raw_da)
    da_payload["prediction"] = projected.astype(np.float32)
    _save(da_payload, output / f"{family}_da_inner_seed0.npz", family, "da")
    for task in ("rt_at_da", "rt"):
        payload = _load(source / f"v4_official_{task}_inner_seed0.npz")
        _save(payload, output / f"{family}_{task}_inner_seed0.npz", family, task)

    target = base_da["target_da"].astype(np.float64)
    base = base_da["prediction"].astype(np.float64)
    raw = raw_da["prediction"].astype(np.float64)
    report = {
        "experiment": "pure_v42_memory_locked_us_ercot_predictions_v1",
        "year": 2023,
        "base_mae_usd_per_mwh": float(np.mean(np.abs(base - target))),
        "raw_candidate_mae_usd_per_mwh": float(np.mean(np.abs(raw - target))),
        "locked_candidate_mae_usd_per_mwh": float(np.mean(np.abs(projected - target))),
        "maximum_target_objective_gap": maximum_gap,
        "maximum_plan_difference_mwh": maximum_plan_difference,
        "maximum_stationarity_residual": maximum_stationarity,
        "nodes": int(len(base_da["nodes"])),
        "days": int(len(dates)),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
