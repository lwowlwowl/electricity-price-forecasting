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
    LookaheadMPCPolicy,
    STEPolicy,
    compute_regret,
    greedy_hindsight_revenue,
    lp_oracle_revenue,
)


class LookaheadMPCPolicyTests(unittest.TestCase):
    def _policy(self, kappa: float = 0.0):
        return LookaheadMPCPolicy(
            power_mw=1.0,
            energy_mwh=2.0,
            eta=1.0,
            init_soc_frac=0.5,
            kappa=kappa,
            soc_min=0.0,
            soc_max=2.0,
            e_cyc=2.0,
        )

    def test_joint_path_charges_low_then_discharges_high(self):
        action = self._policy()(torch.tensor([[10.0, 30.0]]))
        self.assertEqual(float(action[0, 0]), -1.0)

    def test_da_plan_does_not_change_rt_optimum_without_penalty(self):
        policy = self._policy(kappa=1.0)
        price = torch.tensor([[10.0, 30.0]])
        without_da = policy(price)
        with_da = policy(price, da_plan=torch.tensor([[1.0, -1.0]]))
        torch.testing.assert_close(with_da, without_da)

    def test_da_plan_can_change_rt_optimum_when_penalty_is_enabled(self):
        policy = self._policy()
        price = torch.tensor([[1.0]])
        independent = policy(price)
        tracked = policy(
            price,
            da_plan=torch.tensor([[-1.0]]),
            use_deviation_penalty=True,
        )
        self.assertEqual(float(independent[0, 0]), 0.0)
        self.assertEqual(float(tracked[0, 0]), -1.0)


class BESSSimulatorAccountingTests(unittest.TestCase):
    """用能手算的小数字固定动作符号、单位、SOC裁剪和结算口径。"""

    @staticmethod
    def _simulator(kappa: float = 0.0) -> BESSSimulator:
        return BESSSimulator(
            power_mw=1.0,
            energy_mwh=2.0,
            eta=1.0,
            init_soc_frac=0.5,
            dt=1.0,
            kappa=kappa,
            soc_min=0.0,
            soc_max=2.0,
            e_cyc=2.0,
        )

    def test_charge_at_10_discharge_at_30_earns_20(self):
        simulator = self._simulator()
        action = torch.tensor([[-1.0, 1.0]])
        price = torch.tensor([[10.0, 30.0]])

        revenue = simulator(action, price)

        torch.testing.assert_close(revenue, torch.tensor([20.0]))

    def test_do_nothing_earns_zero(self):
        simulator = self._simulator()
        action = torch.zeros((1, 2))
        price = torch.tensor([[10.0, 30.0]])

        revenue = simulator(action, price)

        torch.testing.assert_close(revenue, torch.tensor([0.0]))

    def test_soc_clip_prevents_discharge_beyond_available_energy(self):
        simulator = self._simulator()
        # 初始只有 1 MWh；连续要求放 2 MWh 时，第二小时必须被裁掉。
        action = torch.tensor([[1.0, 1.0]])
        price = torch.tensor([[10.0, 10.0]])

        revenue = simulator(action, price)

        torch.testing.assert_close(revenue, torch.tensor([10.0]))

    def test_dual_settlement_matches_da_when_actual_follows_plan(self):
        simulator = self._simulator()
        plan = torch.tensor([[-1.0, 1.0]])
        actual = plan.clone()
        price_da = torch.tensor([[10.0, 30.0]])
        price_rt = torch.tensor([[100.0, -50.0]])

        revenue = simulator.forward_dual(plan, actual, price_da, price_rt)

        # 实际动作等于计划时，RT偏差为0，只剩DA低买高卖的20。
        torch.testing.assert_close(revenue, torch.tensor([20.0]))

    def test_cancelled_da_plan_is_offset_in_real_time(self):
        simulator = self._simulator()
        plan = torch.tensor([[-1.0, 1.0]])
        actual = torch.zeros_like(plan)
        price_da = torch.tensor([[10.0, 30.0]])
        price_rt = price_da.clone()

        revenue = simulator.forward_dual(plan, actual, price_da, price_rt)

        # DA腿赚20，但同价RT偏差腿正好抵消，最终为0。
        torch.testing.assert_close(revenue, torch.tensor([0.0]))

    def test_throughput_cost_is_charged_once_per_actual_mwh(self):
        simulator = self._simulator(kappa=2.0)
        action = torch.tensor([[-1.0, 1.0]])
        price = torch.tensor([[10.0, 30.0]])

        revenue = simulator(action, price)

        # 毛收益20，充1 MWh和放1 MWh各扣2，共扣4。
        torch.testing.assert_close(revenue, torch.tensor([16.0]))

    def test_plan_projection_has_separate_soc_and_clips_infeasible_action(self):
        simulator = self._simulator()
        # 初始SOC只有1 MWh；第二次连续放电必须被裁为0。
        projection = simulator.project_actions(torch.tensor([[1.0, 1.0]]))

        torch.testing.assert_close(projection.action, torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(
            projection.soc_path_mwh, torch.tensor([[1.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(projection.clipped_energy_mwh, torch.tensor([1.0]))

    def test_unimplemented_plan_track_cannot_fail_silently(self):
        simulator = self._simulator()
        zeros = torch.zeros((1, 2))
        with self.assertRaisesRegex(NotImplementedError, "plan_track"):
            simulator.forward_dual(zeros, zeros, zeros, zeros, plan_track=True)

    def test_plan_projection_tolerates_only_tiny_boundary_roundoff(self):
        simulator = self._simulator()
        projection = simulator.project_actions(
            torch.zeros((1, 1)), initial_soc_mwh=2.0 + 1e-7
        )
        torch.testing.assert_close(projection.soc_path_mwh[0, 0], torch.tensor(2.0))

        with self.assertRaisesRegex(ValueError, "SOC"):
            simulator.project_actions(
                torch.zeros((1, 1)), initial_soc_mwh=2.01
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
