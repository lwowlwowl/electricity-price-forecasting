from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.backtest_dual import locked_dual_backtest  # noqa: E402
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.joint_system import (  # noqa: E402
    estimate_system_zo_gradient,
    joint_proxy_loss,
    revenue_utility,
    settle_joint_episode,
)
from scripts.decision_aware.train_joint_decision_aware import (  # noqa: E402
    _validation_selection_score,
)


def _cfg() -> PilotConfig:
    return PilotConfig(
        bess_power_mw=1.0,
        bess_energy_mwh=2.0,
        bess_eta=1.0,
        bess_init_soc_frac=0.5,
        bess_kappa=1.0,
        bess_soc_min=0.0,
        bess_soc_max=2.0,
        bess_e_cyc=2.0,
        topk_k_charge=1,
        topk_k_discharge=1,
        topk_spread_threshold=0.0,
        use_deviation_penalty=False,
    )


def _inputs(days: int = 2):
    p_da = torch.linspace(0.0, 23.0, 24).repeat(days, 1)
    p_rt_da = torch.zeros(days, 24)
    base = torch.tensor([0.0, 10.0, 4.0, 4.0])
    p_rt = base.repeat(days, 24, 1)
    true_da = torch.linspace(10.0, 40.0, 24).repeat(days, 1)
    true_rt = torch.linspace(-5.0, 50.0, 24).repeat(days, 1)
    return p_da, p_rt_da, p_rt, true_da, true_rt


def test_v2_checkpoint_selection_uses_mean_full_dual_revenue():
    cfg = PilotConfig.from_yaml(
        ROOT / "configs" / "decision_aware" / "joint_decision_aware_v2.yaml"
    )
    validation = {
        "full_dual": {
            "mean_daily_revenue": 123.25,
            "lower_tail_10pct_mean": -80.0,
        }
    }
    assert cfg.joint_selection_tail_weight == 0.0
    assert _validation_selection_score(validation, cfg) == pytest.approx(123.25)


def test_training_kernel_matches_locked_dual_backtest_exactly():
    cfg = _cfg()
    p_da, p_rt_da, p_rt, true_da, true_rt = _inputs(days=2)
    episode = settle_joint_episode(
        p_da, p_rt_da, p_rt, true_da, true_rt,
        torch.tensor([True, False]), cfg,
    )
    timestamps = pd.date_range(
        "2025-01-02 00:00", periods=48, freq="h", tz="America/Chicago"
    )
    dates = sorted(set(timestamps.date))
    report = locked_dual_backtest(
        {dates[i]: p_da[i] for i in range(2)},
        p_rt.reshape(48, 4),
        true_da.reshape(-1),
        true_rt.reshape(-1),
        timestamps,
        cfg,
        da_k_charge=1,
        da_k_discharge=1,
        rt_k_charge=1,
        rt_k_discharge=1,
        rt_at_da_price_forecasts={dates[i]: p_rt_da[i] for i in range(2)},
    )

    expected = torch.tensor([
        report["daily"][str(day)]["net_revenue"] for day in dates
    ])
    assert episode["daily_revenue"].cpu() == pytest.approx(expected)
    assert episode["hourly_revenue"].sum(dim=1).cpu() == pytest.approx(
        episode["daily_revenue"].cpu()
    )
    assert float(episode["final_state"].actual_soc_mwh) == pytest.approx(
        report["actual_state"]["final_soc_mwh"]
    )
    assert float(episode["final_state"].plan_soc_mwh) == pytest.approx(
        report["plan_feasibility"]["final_soc_mwh"]
    )


def test_segment_reset_matches_two_independent_soc_segments():
    cfg = _cfg()
    p_da, p_rt_da, p_rt, true_da, true_rt = _inputs(days=2)
    result = settle_joint_episode(
        p_da, p_rt_da, p_rt, true_da, true_rt,
        torch.tensor([True, True]), cfg,
        initial_plan_soc_mwh=0.0,
        initial_actual_soc_mwh=0.0,
    )
    # starts_new_segment必须覆盖传入的旧状态，两个数据段都从合同初始1 MWh开始。
    one_day = settle_joint_episode(
        p_da[:1], p_rt_da[:1], p_rt[:1], true_da[:1], true_rt[:1],
        torch.tensor([True]), cfg,
    )
    assert result["actual_soc_end_mwh"][0] == pytest.approx(
        one_day["actual_soc_end_mwh"][0]
    )
    assert result["actual_soc_end_mwh"][1] == pytest.approx(
        one_day["actual_soc_end_mwh"][0]
    )


def test_three_system_level_zo_signals_are_finite_and_backpropagate():
    torch.manual_seed(7)
    cfg = _cfg()
    p_da, p_rt_da, p_rt, true_da, true_rt = _inputs(days=2)
    reset = torch.tensor([True, False])

    def revenue(da, rt_da, rt):
        return settle_joint_episode(
            da, rt_da, rt, true_da, true_rt, reset, cfg
        )["daily_revenue"]

    gradients = []
    diagnostics = []
    for prediction, objective in (
        (p_da, lambda value: revenue(value, p_rt_da, p_rt)),
        (p_rt_da, lambda value: revenue(p_da, value, p_rt)),
        (p_rt, lambda value: revenue(p_da, p_rt_da, value)),
    ):
        gradient, diagnostic = estimate_system_zo_gradient(
            prediction,
            objective,
            epsilon=10.0,
            directions=8,
            pred_clamp=(-500.0, 10000.0),
        )
        gradients.append(gradient)
        diagnostics.append(diagnostic)

    leaves = [value.clone().requires_grad_(True) for value in (p_da, p_rt_da, p_rt)]
    loss = sum(
        joint_proxy_loss(value, gradient, scale=1.0)
        for value, gradient in zip(leaves, gradients)
    )
    loss.backward()
    for leaf, gradient, diagnostic in zip(leaves, gradients, diagnostics):
        assert diagnostic["finite"] is True
        assert diagnostic["gradient_norm"] > 0.0
        assert torch.isfinite(gradient).all()
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()
        assert torch.linalg.vector_norm(leaf.grad) > 0.0


def test_per_day_orthogonal_zo_and_sample_sum_proxy_are_finite():
    torch.manual_seed(11)
    cfg = _cfg()
    p_da, p_rt_da, p_rt, true_da, true_rt = _inputs(days=3)
    spread = p_da - p_rt_da

    gradient, diagnostic = estimate_system_zo_gradient(
        spread,
        lambda value: settle_joint_episode(
            value, torch.zeros_like(value), p_rt,
            true_da, true_rt, torch.tensor([True, False, False]), cfg,
        )["daily_revenue"],
        epsilon=5.0,
        directions=4,
        pred_clamp=(-500.0, 10000.0),
        direction_mode="orthogonal",
        feedback_mode="per_day",
        tail_weight=0.2,
    )
    leaf_da = p_da.clone().requires_grad_(True)
    leaf_rt_da = p_rt_da.clone().requires_grad_(True)
    loss = joint_proxy_loss(
        leaf_da - leaf_rt_da,
        gradient,
        scale=10.0,
        reduction="sample_sum_mean",
    )
    loss.backward()

    assert diagnostic["finite"] is True
    assert diagnostic["feedback_mode"] == "per_day"
    assert len(diagnostic["per_day_directional_loss_differences"]) == 4
    assert torch.isfinite(leaf_da.grad).all()
    # 直接优化spread=p_DA-p_RT|DA时，两个独立模型必须收到
    # 大小相同、方向相反的输出层信号。
    assert leaf_da.grad == pytest.approx(-leaf_rt_da.grad)


def test_revenue_utility_penalizes_lower_tail_and_validates_contract():
    revenue = torch.tensor([10.0, 20.0, -30.0, 40.0])
    mean_only = revenue_utility(revenue)
    risk_adjusted = revenue_utility(
        revenue, tail_weight=0.25, tail_fraction=0.25
    )
    assert mean_only == pytest.approx(10.0)
    assert risk_adjusted == pytest.approx(0.0)
    with pytest.raises(ValueError):
        revenue_utility(revenue, tail_weight=1.1)


def test_per_group_feedback_matches_rolling_rt_window_shape():
    torch.manual_seed(19)
    cfg = _cfg()
    p_da, p_rt_da, p_rt, true_da, true_rt = _inputs(days=2)
    gradient, diagnostic = estimate_system_zo_gradient(
        p_rt,
        lambda value: settle_joint_episode(
            p_da, p_rt_da, value, true_da, true_rt,
            torch.tensor([True, False]), cfg,
        )["hourly_revenue"],
        epsilon=5.0,
        directions=4,
        pred_clamp=(-500.0, 10000.0),
        direction_mode="orthogonal",
        feedback_mode="per_group",
        tail_weight=0.1,
    )
    assert gradient.shape == p_rt.shape
    assert diagnostic["feedback_mode"] == "per_group"
    assert diagnostic["finite"] is True
    assert diagnostic["gradient_norm"] > 0.0
