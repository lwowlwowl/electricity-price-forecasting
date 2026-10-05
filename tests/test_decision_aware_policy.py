from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.policy import (  # noqa: E402
    BESSSimulator,
    LPOracleSolveError,
    MILPOracleSolveError,
    MILPSolveStatus,
    LookaheadMPCPolicy,
    STEPolicy,
    compute_regret,
    greedy_hindsight_revenue,
    lp_oracle_revenue,
    lp_oracle_revenue_dual,
    lp_oracle_revenue_dual_plan_tracking_restricted,
    milp_oracle_revenue_dual,
    plan_track_override,
    solve_dual_penalty_milp_day,
)


class PlanTrackPriorityTests(unittest.TestCase):
    def test_uses_rt_topk_when_da_plan_is_zero(self):
        actual = plan_track_override(
            torch.tensor([[0.0, 0.0]]), torch.tensor([[-1.0, 1.0]])
        )
        torch.testing.assert_close(actual, torch.tensor([[-1.0, 1.0]]))

    def test_keeps_da_when_rt_candidate_is_opposite(self):
        actual = plan_track_override(
            torch.tensor([[0.5, -0.5]]), torch.tensor([[-1.0, 1.0]])
        )
        torch.testing.assert_close(actual, torch.tensor([[0.5, -0.5]]))

    def test_same_direction_rt_uses_remaining_power(self):
        actual = plan_track_override(
            torch.tensor([[0.4, -0.25]]), torch.tensor([[1.0, -1.0]])
        )
        torch.testing.assert_close(actual, torch.tensor([[1.0, -1.0]]))


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


class DualPenaltyMILPOracleTests(unittest.TestCase):
    @staticmethod
    def _simulator(kappa: float = 0.0, e_cyc: float = 2.0):
        return BESSSimulator(
            power_mw=1.0,
            energy_mwh=2.0,
            eta=1.0,
            init_soc_frac=0.5,
            kappa=kappa,
            soc_min=0.0,
            soc_max=2.0,
            e_cyc=e_cyc,
        )

    def test_three_percent_band_is_strictly_better_than_old_restricted_lp(self):
        simulator = self._simulator()
        exact = solve_dual_penalty_milp_day([10.0], [100.0], simulator)
        with self.assertWarns(DeprecationWarning):
            restricted = lp_oracle_revenue_dual_plan_tracking_restricted(
                torch.tensor([[10.0]]), torch.tensor([[100.0]]), simulator
            )

        # Commit 100/103 MWh DA and deliver 1 MWh RT.  The 3/103 MWh
        # deviation is exactly the 3% tolerance, so the global optimum earns
        # 1300/103 > the historical uRT=uDA restricted value of 10.
        self.assertAlmostEqual(exact.revenue, 1300.0 / 103.0, places=7)
        self.assertAlmostEqual(float(restricted[0]), 10.0, places=6)
        self.assertGreater(exact.revenue, float(restricted[0]))
        np.testing.assert_allclose(exact.dispatch.excess_mwh, 0.0, atol=1e-9)
        np.testing.assert_allclose(
            exact.dispatch.deviation_mwh,
            exact.dispatch.tolerance_mwh,
            atol=1e-9,
        )

    def test_dispatch_obeys_mode_power_soc_and_both_cycle_constraints(self):
        simulator = self._simulator(e_cyc=0.4)
        solved = solve_dual_penalty_milp_day(
            [-50.0, 80.0, 20.0],
            [-40.0, 100.0, 10.0],
            simulator,
        )
        dispatch = solved.dispatch

        for discharge, charge, soc in (
            (
                dispatch.da_discharge_mwh,
                dispatch.da_charge_mwh,
                dispatch.da_soc_path_mwh,
            ),
            (
                dispatch.rt_discharge_mwh,
                dispatch.rt_charge_mwh,
                dispatch.rt_soc_path_mwh,
            ),
        ):
            self.assertTrue(np.all(discharge <= 1.0 + 1e-8))
            self.assertTrue(np.all(charge <= 1.0 + 1e-8))
            self.assertTrue(np.all(discharge * charge <= 1e-10))
            self.assertLessEqual(float(discharge.sum()), 0.4 + 1e-8)
            self.assertTrue(np.all(soc >= -1e-8))
            self.assertTrue(np.all(soc <= 2.0 + 1e-8))

    def test_degradation_cost_can_make_the_optimal_dispatch_idle(self):
        solved = solve_dual_penalty_milp_day(
            [10.0], [100.0], self._simulator(kappa=20.0)
        )

        self.assertAlmostEqual(solved.revenue, 0.0, places=8)
        np.testing.assert_allclose(solved.dispatch.da_discharge_mwh, 0.0)
        np.testing.assert_allclose(solved.dispatch.rt_discharge_mwh, 0.0)
        self.assertAlmostEqual(solved.dispatch.degradation_cost, 0.0, places=8)

    def test_batch_and_legacy_entrypoint_return_certified_milp_statuses(self):
        simulator = self._simulator()
        price_da = torch.tensor([[10.0], [20.0]], dtype=torch.float64)
        price_rt = torch.tensor([[100.0], [80.0]], dtype=torch.float64)

        direct, statuses = milp_oracle_revenue_dual(
            price_da, price_rt, simulator, return_status=True
        )
        compatible = lp_oracle_revenue_dual(
            price_da,
            price_rt,
            simulator,
            use_deviation_penalty=True,
        )

        self.assertEqual(direct.shape, (2,))
        self.assertEqual(direct.dtype, torch.float64)
        torch.testing.assert_close(compatible, direct)
        self.assertEqual([status.sample_index for status in statuses], [0, 1])
        self.assertTrue(all(isinstance(status, MILPSolveStatus) for status in statuses))
        self.assertTrue(all(status.success for status in statuses))
        self.assertTrue(all(status.status_code == 0 for status in statuses))

    def test_nonoptimal_milp_status_raises_instead_of_returning_an_incumbent(self):
        failed = SimpleNamespace(
            success=False,
            status=1,
            message="time limit reached in test",
            fun=-99.0,
            mip_gap=0.2,
            mip_node_count=3,
            x=np.zeros(7),
        )
        with mock.patch("scipy.optimize.milp", return_value=failed):
            with self.assertRaises(MILPOracleSolveError) as ctx:
                solve_dual_penalty_milp_day(
                    [10.0], [100.0], self._simulator()
                )

        self.assertFalse(ctx.exception.solve_status.success)
        self.assertEqual(ctx.exception.solve_status.status_code, 1)
        self.assertIn("time limit reached", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
