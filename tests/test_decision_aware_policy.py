from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.policy import (  # noqa: E402
    BESSSimulator,
    LPOracleSolveError,
    STEPolicy,
    compute_regret,
    greedy_hindsight_revenue,
    lp_oracle_revenue,
)


class LPOracleTests(unittest.TestCase):
    def setUp(self):
        self.simulator = BESSSimulator(
            power_mw=1.0,
            energy_mwh=4.0,
            eta=0.95,
            init_soc_frac=0.5,
            kappa=0.0,
            soc_min=0.4,
            soc_max=3.6,
            e_cyc=4.0,
        )
        self.price = torch.tensor([[10.0, 20.0, 50.0, 5.0]])

    def test_lp_returns_auditable_success_status(self):
        revenue, statuses = lp_oracle_revenue(
            self.price, self.simulator, return_status=True
        )

        self.assertEqual(revenue.shape, (1,))
        self.assertEqual(len(statuses), 1)
        self.assertTrue(statuses[0].success)
        self.assertEqual(statuses[0].sample_index, 0)
        self.assertEqual(statuses[0].leg, "single")

    def test_lp_failure_raises_instead_of_falling_back_to_greedy(self):
        failed = SimpleNamespace(
            success=False,
            status=2,
            message="infeasible test problem",
            fun=None,
        )
        with mock.patch("scipy.optimize.linprog", return_value=failed):
            with self.assertRaises(LPOracleSolveError) as ctx:
                lp_oracle_revenue(self.price, self.simulator)

        self.assertEqual(ctx.exception.solve_status.status_code, 2)
        self.assertIn("infeasible test problem", str(ctx.exception))

    def test_greedy_reference_cannot_be_used_as_regret_oracle(self):
        baseline = greedy_hindsight_revenue(self.price, self.simulator)
        self.assertEqual(baseline.shape, (1,))

        with self.assertRaisesRegex(ValueError, "only accepts oracle='lp'"):
            compute_regret(
                self.price,
                self.price,
                self.simulator,
                STEPolicy(),
                oracle="greedy",
            )


if __name__ == "__main__":
    unittest.main()
