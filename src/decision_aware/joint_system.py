"""三模型联合训练使用的完整日双结算内核和系统级零阶估计。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .policy import BESSSimulator, HardTopKPolicy, plan_track_override


@dataclass(frozen=True)
class JointEpisodeState:
    plan_soc_mwh: torch.Tensor
    actual_soc_mwh: torch.Tensor


def revenue_utility(
    daily_revenue: torch.Tensor,
    *,
    tail_weight: float = 0.0,
    tail_fraction: float = 0.10,
) -> torch.Tensor:
    """把逐日现金收益汇总为训练/选模效用。

    ``tail_weight=0``严格复现v1的日均收益。大于0时使用均值与最差尾部
    均值的凸组合，防止少数尖峰日的灾难性偏差被大量普通日掩盖。
    """
    if daily_revenue.ndim != 1 or daily_revenue.numel() == 0:
        raise ValueError("daily_revenue必须是一维非空张量")
    weight = float(tail_weight)
    fraction = float(tail_fraction)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("tail_weight必须位于[0,1]")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail_fraction必须位于(0,1]")
    mean = daily_revenue.mean()
    if weight == 0.0:
        return mean
    tail_count = max(1, int(torch.ceil(
        daily_revenue.new_tensor(fraction * daily_revenue.numel())
    ).item()))
    lower_tail = torch.topk(
        daily_revenue, tail_count, largest=False
    ).values.mean()
    return (1.0 - weight) * mean + weight * lower_tail


def _simulator(cfg) -> BESSSimulator:
    return BESSSimulator(
        cfg.bess_power_mw,
        cfg.bess_energy_mwh,
        cfg.bess_eta,
        cfg.bess_init_soc_frac,
        kappa=cfg.bess_kappa,
        soc_min=cfg.bess_soc_min,
        soc_max=cfg.bess_soc_max,
        e_cyc=cfg.bess_e_cyc,
    )


def _scalar_state(value, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        value, dtype=reference.dtype, device=reference.device
    ).reshape(())


@torch.no_grad()
def settle_joint_episode(
    p_da: torch.Tensor,
    p_rt_at_da: torch.Tensor,
    p_rt_windows: torch.Tensor,
    true_da: torch.Tensor,
    true_rt: torch.Tensor,
    starts_new_segment: torch.Tensor,
    cfg,
    *,
    initial_plan_soc_mwh: float | torch.Tensor | None = None,
    initial_actual_soc_mwh: float | torch.Tensor | None = None,
    coordination_mode: str | None = None,
) -> dict:
    """按日期顺序执行完整系统，返回逐日现金收益和连续SOC。

    输入第一维是按时间排序的交付日。滚动RT形状为[D,24,H]；每个小时
    对H点曲线做1充1放HardTopK，但只执行第一步。期末SOC不货币化。
    """
    if p_da.ndim != 2 or p_rt_at_da.shape != p_da.shape:
        raise ValueError("p_da和p_rt_at_da必须是相同的[D,24]")
    if p_da.shape[1] != 24:
        raise ValueError("联合结算要求每个交付日24小时")
    days = p_da.shape[0]
    if p_rt_windows.ndim != 3 or p_rt_windows.shape[:2] != (days, 24):
        raise ValueError("p_rt_windows必须是[D,24,H_rt]")
    if true_da.shape != p_da.shape or true_rt.shape != p_da.shape:
        raise ValueError("真实DA/RT价格必须是[D,24]")
    reset = torch.as_tensor(
        starts_new_segment, dtype=torch.bool, device=p_da.device
    ).reshape(-1)
    if reset.numel() != days:
        raise ValueError("starts_new_segment长度必须等于交付日数")

    initial_soc = float(cfg.bess_energy_mwh * cfg.bess_init_soc_frac)
    plan_soc = _scalar_state(
        initial_soc if initial_plan_soc_mwh is None else initial_plan_soc_mwh,
        p_da,
    )
    actual_soc = _scalar_state(
        initial_soc if initial_actual_soc_mwh is None else initial_actual_soc_mwh,
        p_da,
    )
    simulator = _simulator(cfg)
    da_policy = HardTopKPolicy(
        cfg.topk_k_charge,
        cfg.topk_k_discharge,
        spread_threshold=cfg.resolved_spread_threshold,
    )
    rt_policy = HardTopKPolicy(
        cfg.rt_topk_k_charge,
        cfg.rt_topk_k_discharge,
        spread_threshold=cfg.resolved_spread_threshold,
    )
    rt_candidates = rt_policy(
        p_rt_windows.reshape(days * 24, p_rt_windows.shape[-1])
    )[:, 0].reshape(days, 24)
    mode = coordination_mode or getattr(cfg, "joint_coordination_mode", "rt_only")
    if mode not in {"rt_only", "follow_da", "plan_track_topk"}:
        raise ValueError(f"联合结算不支持的RT协调方式: {mode}")
    da_signal_mode = getattr(cfg, "joint_da_signal_mode", "spread")
    if da_signal_mode not in {"spread", "da_only"}:
        raise ValueError(f"未知DA决策信号: {da_signal_mode}")

    daily_revenue = []
    da_legs = []
    rt_legs = []
    degradation_costs = []
    deviation_penalties = []
    plan_actions = []
    actual_actions = []
    plan_soc_end = []
    actual_soc_end = []
    plan_clipped = []
    hourly_revenues = []

    for day in range(days):
        if bool(reset[day]):
            plan_soc = p_da.new_tensor(initial_soc)
            actual_soc = p_da.new_tensor(initial_soc)

        da_signal = (
            p_da[day] - p_rt_at_da[day]
            if da_signal_mode == "spread" else p_da[day]
        )
        intended_plan = da_policy(da_signal.reshape(1, 24))
        plan_projection = simulator.project_actions(intended_plan, plan_soc)
        plan = plan_projection.action[0]
        plan_soc = plan_projection.soc_path_mwh[0, -1]

        if mode == "follow_da":
            rt_intended = plan
        elif mode == "plan_track_topk":
            rt_intended = plan_track_override(
                plan.reshape(1, 24), rt_candidates[day].reshape(1, 24)
            )[0]
        else:
            rt_intended = rt_candidates[day]
        actual_projection = simulator.project_actions(
            rt_intended.reshape(1, 24), actual_soc
        )
        actual_net = actual_projection.net_energy_mwh[0]
        actual_action = actual_projection.action[0]
        actual_soc = actual_projection.soc_path_mwh[0, -1]

        da_net = plan * float(cfg.bess_power_mw)
        hourly_da_leg = da_net * true_da[day]
        hourly_rt_leg = (actual_net - da_net) * true_rt[day]
        hourly_degradation = float(cfg.bess_kappa) * (
            actual_projection.discharge_mwh[0]
            + actual_projection.charge_mwh[0]
        )
        hourly_penalty = p_da.new_zeros(24)
        if bool(cfg.use_deviation_penalty):
            tolerance = 0.03 * torch.abs(da_net)
            excess = torch.clamp(
                torch.abs(actual_net - da_net) - tolerance, min=0.0
            )
            hourly_penalty = 2.0 * torch.abs(true_rt[day]) * excess
        hourly_revenue = (
            hourly_da_leg + hourly_rt_leg
            - hourly_degradation - hourly_penalty
        )
        da_leg = torch.sum(hourly_da_leg)
        rt_leg = torch.sum(hourly_rt_leg)
        degradation = torch.sum(hourly_degradation)
        penalty = torch.sum(hourly_penalty)
        revenue = torch.sum(hourly_revenue)

        daily_revenue.append(revenue)
        da_legs.append(da_leg)
        rt_legs.append(rt_leg)
        degradation_costs.append(degradation)
        deviation_penalties.append(penalty)
        plan_actions.append(plan)
        actual_actions.append(actual_action)
        plan_soc_end.append(plan_soc)
        actual_soc_end.append(actual_soc)
        plan_clipped.append(plan_projection.clipped_energy_mwh[0])
        hourly_revenues.append(hourly_revenue)

    return {
        "coordination_mode": mode,
        "da_signal_mode": da_signal_mode,
        "daily_revenue": torch.stack(daily_revenue),
        "hourly_revenue": torch.stack(hourly_revenues),
        "da_leg": torch.stack(da_legs),
        "rt_deviation_leg": torch.stack(rt_legs),
        "degradation_cost": torch.stack(degradation_costs),
        "deviation_penalty": torch.stack(deviation_penalties),
        "plan_actions": torch.stack(plan_actions),
        "actual_actions": torch.stack(actual_actions),
        "plan_soc_end_mwh": torch.stack(plan_soc_end),
        "actual_soc_end_mwh": torch.stack(actual_soc_end),
        "plan_clipped_mwh": torch.stack(plan_clipped),
        "final_state": JointEpisodeState(
            plan_soc_mwh=plan_soc.detach(),
            actual_soc_mwh=actual_soc.detach(),
        ),
    }


@torch.no_grad()
def estimate_system_zo_gradient(
    prediction: torch.Tensor,
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    epsilon: float,
    directions: int,
    pred_clamp: tuple[float, float],
    direction_mode: str = "gaussian",
    feedback_mode: str = "episode_scalar",
    tail_weight: float = 0.0,
    tail_fraction: float = 0.10,
) -> tuple[torch.Tensor, dict]:
    """对完整episode收益做成对零阶估计。

    ``episode_scalar``严格复现v1：所有天共用一个标量效用。
    ``per_day``利用已有的逐日收益反馈，对batch内每天分别估计
    其输出梯度。``per_group``要求objective形状等于prediction去掉
    最后一维，用于将滚动RT每小时收益分配给对应的H维窗口。
    """
    if epsilon <= 0 or directions < 1:
        raise ValueError("epsilon和directions必须为正")
    detached = prediction.detach().float()
    gradient = torch.zeros_like(detached)
    differences = []
    changed = 0
    changed_day_fraction = 0.0
    lo, hi = (float(pred_clamp[0]), float(pred_clamp[1]))
    if feedback_mode not in {"episode_scalar", "per_day", "per_group"}:
        raise ValueError(f"未知零阶反馈模式: {feedback_mode}")
    if feedback_mode == "per_day" and detached.ndim < 2:
        raise ValueError("per_day模式要求prediction第一维是交付日")
    if feedback_mode == "per_group" and detached.ndim < 2:
        raise ValueError("per_group模式要求prediction至少为二维")

    if direction_mode == "gaussian":
        sampled_directions = [
            torch.randn_like(detached) for _ in range(directions)
        ]
    elif direction_mode == "orthogonal":
        if feedback_mode == "per_day":
            dimension = detached[0].numel()
            raw_shape = (detached.shape[0], dimension, directions)
        elif feedback_mode == "per_group":
            dimension = detached.shape[-1]
            raw_shape = (*detached.shape[:-1], dimension, directions)
        else:
            dimension = detached.numel()
            raw_shape = (dimension, directions)
        if directions > dimension:
            raise ValueError("正交方向数不能超过prediction元素数")
        raw = torch.randn(
            *raw_shape, dtype=detached.dtype, device=detached.device
        )
        basis, _ = torch.linalg.qr(raw, mode="reduced")
        # 单位球方向乘sqrt(d)，使每个坐标的二阶矩与N(0,I)一致。
        basis = basis * (dimension ** 0.5)
        if feedback_mode in {"per_day", "per_group"}:
            sampled_directions = [
                basis[..., index].reshape_as(detached)
                for index in range(directions)
            ]
        else:
            sampled_directions = [
                basis[:, index].reshape_as(detached)
                for index in range(directions)
            ]
    else:
        raise ValueError(f"未知零阶方向模式: {direction_mode}")

    utilities_plus = []
    utilities_minus = []
    per_day_differences = []
    for direction in sampled_directions:
        plus = (detached + epsilon * direction).clamp(lo, hi)
        minus = (detached - epsilon * direction).clamp(lo, hi)
        # objective_fn返回逐日或逐组收益；业务损失F=-utility(R)。
        revenue_plus = objective_fn(plus)
        revenue_minus = objective_fn(minus)
        if revenue_minus.shape != revenue_plus.shape:
            raise ValueError("objective_fn正负扰动必须返回相同形状")
        if feedback_mode == "per_group":
            expected_shape = detached.shape[:-1]
            if revenue_plus.shape != expected_shape:
                raise ValueError(
                    "per_group的objective_fn形状必须等于"
                    "prediction.shape[:-1]"
                )
            daily_plus = revenue_plus.reshape(revenue_plus.shape[0], -1).sum(1)
            daily_minus = revenue_minus.reshape(
                revenue_minus.shape[0], -1
            ).sum(1)
        else:
            if revenue_plus.ndim != 1:
                raise ValueError(
                    "episode_scalar/per_day的objective_fn必须返回逐日收益"
                )
            daily_plus = revenue_plus
            daily_minus = revenue_minus
        utility_plus = revenue_utility(
            daily_plus, tail_weight=tail_weight,
            tail_fraction=tail_fraction,
        )
        utility_minus = revenue_utility(
            daily_minus, tail_weight=tail_weight,
            tail_fraction=tail_fraction,
        )
        if feedback_mode == "episode_scalar":
            difference = (-utility_plus + utility_minus) / (2.0 * epsilon)
            gradient = gradient + difference * direction
            differences.append(float(difference.cpu()))
        else:
            difference_by_group = (
                -revenue_plus + revenue_minus
            ) / (2.0 * epsilon)
            if tail_weight > 0:
                day_count = daily_plus.numel()
                tail_count = max(1, int(torch.ceil(
                    daily_plus.new_tensor(tail_fraction * day_count)
                ).item()))
                midpoint_revenue = 0.5 * (daily_plus + daily_minus)
                tail_indices = torch.topk(
                    midpoint_revenue, tail_count, largest=False
                ).indices
                day_weights = torch.full_like(
                    daily_plus, 1.0 - float(tail_weight)
                )
                day_weights[tail_indices] += (
                    float(tail_weight) * day_count / tail_count
                )
                day_weight_shape = (day_count,) + (
                    (1,) * (difference_by_group.ndim - 1)
                )
                difference_by_group = difference_by_group * day_weights.reshape(
                    day_weight_shape
                )
            if feedback_mode == "per_day":
                expand_shape = (difference_by_group.shape[0],) + (
                    (1,) * (detached.ndim - 1)
                )
                scaled_difference = difference_by_group.reshape(expand_shape)
            else:
                scaled_difference = difference_by_group.unsqueeze(-1)
            gradient = gradient + scaled_difference * direction
            differences.append(float(difference_by_group.mean().cpu()))
            per_day_differences.append(
                difference_by_group.detach().cpu().tolist()
            )
        utilities_plus.append(float(utility_plus.cpu()))
        utilities_minus.append(float(utility_minus.cpu()))
        changed += int(not torch.equal(revenue_plus, revenue_minus))
        changed_values = revenue_plus != revenue_minus
        if feedback_mode == "per_group":
            changed_values = changed_values.reshape(
                changed_values.shape[0], -1
            ).any(dim=1)
        changed_day_fraction += float(changed_values.float().mean().cpu())
    gradient = (gradient / directions).detach()
    diagnostics = {
        "gradient_norm": float(torch.linalg.vector_norm(gradient).cpu()),
        "gradient_rms": float(torch.mean(gradient.square()).sqrt().cpu()),
        "nonzero_fraction": float(torch.mean((gradient != 0).float()).cpu()),
        "changed_direction_fraction": changed / directions,
        "changed_day_fraction": changed_day_fraction / directions,
        "finite": bool(torch.isfinite(gradient).all()),
        "direction_mode": direction_mode,
        "feedback_mode": feedback_mode,
        "tail_weight": float(tail_weight),
        "tail_fraction": float(tail_fraction),
        "directional_loss_differences": differences,
        "utilities_plus": utilities_plus,
        "utilities_minus": utilities_minus,
        "per_day_directional_loss_differences": per_day_differences,
    }
    return gradient, diagnostics


def joint_proxy_loss(
    prediction: torch.Tensor,
    gradient: torch.Tensor,
    scale: float,
    reduction: str = "element_mean",
) -> torch.Tensor:
    """将零阶输出梯度注入可微模型。

    ``element_mean``保留v1的额外按元素缩小。
    ``sample_sum_mean``先对每天所有决策维求和，再对天数
    取均值，与``per_day``估计的逐日损失梯度含义一致。
    ``global_sum``直接做全episode内积，适用于``episode_scalar``已经
    估计完整episode目标梯度的情形，避免再次除以天数。
    """
    if scale <= 0:
        raise ValueError("joint_proxy_scale必须为正")
    if prediction.shape != gradient.shape:
        raise ValueError("prediction和gradient形状必须一致")
    product = prediction * gradient.detach()
    if reduction == "element_mean":
        reduced = product.mean()
    elif reduction == "sample_sum_mean":
        if prediction.ndim < 2:
            raise ValueError("sample_sum_mean要求第一维是交付日")
        reduced = product.reshape(prediction.shape[0], -1).sum(dim=1).mean()
    elif reduction == "global_sum":
        reduced = product.sum()
    else:
        raise ValueError(f"未知零阶代理reduction: {reduction}")
    return reduced / float(scale)
