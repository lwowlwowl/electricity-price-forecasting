from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.inverse_lp_projection import (  # noqa: E402
    build_joint_lp,
    project_da_to_target_normal_cone,
    solve_lexicographic_nearest_solution,
    solve_joint_lp,
)
from decision_aware.policy import BESSSimulator  # noqa: E402


def simulator() -> BESSSimulator:
    return BESSSimulator(
        power_mw=1.0,
        energy_mwh=4.0,
        eta=0.95,
        init_soc_frac=0.5,
        dt=0.25,
        kappa=5.0,
        soc_min=0.4,
        soc_max=3.6,
        e_cyc=4.0,
    )


def test_inverse_projection_keeps_target_solution_optimal() -> None:
    v4 = np.asarray([20.0, 80.0, 10.0, 100.0])
    rt = np.asarray([40.0, 40.0, 40.0, 40.0])
    anchor = np.asarray([35.0, 45.0, 35.0, 45.0])
    result = project_da_to_target_normal_cone(
        anchor,
        v4,
        rt,
        simulator(),
        initial_plan_soc_mwh=2.0,
    )
    assert result.target_objective_gap <= 1e-7
    assert result.max_stationarity_residual <= 1e-7
    projected = build_joint_lp(
        result.projected_da,
        rt,
        simulator(),
        initial_plan_soc_mwh=2.0,
    )
    solved = solve_joint_lp(projected)
    np.testing.assert_allclose(
        projected.objective @ result.target_solution,
        projected.objective @ solved,
        atol=1e-7,
    )


def test_lexicographic_tiebreak_recovers_primary_optimal_reference() -> None:
    problem = build_joint_lp(
        np.asarray([20.0, 80.0, 10.0, 100.0]),
        np.asarray([40.0, 40.0, 40.0, 40.0]),
        simulator(),
        initial_plan_soc_mwh=2.0,
    )
    reference = solve_joint_lp(problem)
    selected = solve_lexicographic_nearest_solution(problem, reference)
    np.testing.assert_allclose(selected, reference, atol=1e-8)


def test_projection_generator_has_no_teacher_model_or_truth_dependency() -> None:
    sources = [
        ROOT / "src/decision_aware/inverse_lp_projection.py",
        ROOT / "scripts/decision_aware/prepare_austria_v4_inverse_lp_exaa.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in sources)
    for forbidden in ("timesfm", "chronos", "brim", "prediction truth"):
        assert forbidden not in text
    assert "日前价格 (eur/mwh)" not in text
