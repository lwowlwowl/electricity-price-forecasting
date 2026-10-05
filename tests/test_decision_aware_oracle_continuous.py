import unittest
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.oracle_continuous import solve_dual_penalty_milp_segment
from decision_aware.policy import BESSSimulator, solve_dual_penalty_milp_day


class ContinuousPenaltyOracleTests(unittest.TestCase):
    def setUp(self):
        self.simulator = BESSSimulator(
            power_mw=1.0,
            energy_mwh=4.0,
            eta=0.95,
            init_soc_frac=0.5,
            kappa=5.7,
            soc_min=0.4,
            soc_max=3.6,
            e_cyc=4.0,
        )

    def test_one_day_sparse_segment_matches_small_day_oracle(self):
        da = np.asarray([15.0, 90.0, -12.0, 45.0], dtype=np.float64)
        rt = np.asarray([20.0, 82.0, -30.0, 60.0], dtype=np.float64)
        expected = solve_dual_penalty_milp_day(da, rt, self.simulator)
        actual = solve_dual_penalty_milp_segment(
            da.reshape(1, -1), rt.reshape(1, -1), self.simulator
        )
        self.assertTrue(actual.status.success)
        self.assertAlmostEqual(actual.revenue, expected.revenue, places=6)
        self.assertEqual(actual.dispatch.daily_revenue.shape, (1,))

    def test_two_day_soc_path_is_one_continuous_recurrence(self):
        da = np.asarray(
            [[-30.0, 10.0, 20.0, 25.0], [100.0, 70.0, -5.0, 5.0]],
            dtype=np.float64,
        )
        rt = np.asarray(
            [[-25.0, 8.0, 18.0, 22.0], [90.0, 65.0, -8.0, 3.0]],
            dtype=np.float64,
        )
        solved = solve_dual_penalty_milp_segment(da, rt, self.simulator)
        dispatch = solved.dispatch
        self.assertEqual(dispatch.da_soc_path_mwh.shape, (9,))
        self.assertEqual(dispatch.rt_soc_path_mwh.shape, (9,))
        self.assertEqual(dispatch.daily_revenue.shape, (2,))

        for net, soc_path in (
            (dispatch.da_net_mwh.reshape(-1), dispatch.da_soc_path_mwh),
            (dispatch.rt_net_mwh.reshape(-1), dispatch.rt_soc_path_mwh),
        ):
            changes = np.where(
                net >= 0.0,
                -net / self.simulator.eta,
                -net * self.simulator.eta,
            )
            np.testing.assert_allclose(
                soc_path[1:], soc_path[0] + np.cumsum(changes), atol=1e-7
            )
            self.assertGreaterEqual(float(np.min(soc_path)), self.simulator.s_min - 1e-7)
            self.assertLessEqual(float(np.max(soc_path)), self.simulator.s_max + 1e-7)


if __name__ == "__main__":
    unittest.main()
