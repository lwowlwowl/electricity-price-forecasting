"""独立 DA 与独立 RT 模型的锁定计划双结算回测。

口径：
1. 每个交付日的 24 小时 DA 预测先变成动作，并在交付日前锁定；
2. DA 计划用一条独立的计划 SOC 做可行性投影；
3. RT 每小时重新预测未来 H 小时，但只执行第一个动作；
4. 只有 RT 实际动作更新真实 SOC；
5. 收益按 DA 计划量和 RT 实际偏差量进行双结算。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .policy import BESSSimulator, HardTopKPolicy, LookaheadMPCPolicy


def _as_2d_float(value, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 2:
        raise ValueError(f"{name}应为二维张量，实际为{tuple(tensor.shape)}")
    return tensor


def _as_1d_float(value, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 1:
        raise ValueError(f"{name}应为一维张量，实际为{tuple(tensor.shape)}")
    return tensor


def _digest(values: Sequence[float]) -> str:
    data = np.asarray(values, dtype="<f8")
    return hashlib.sha256(data.tobytes()).hexdigest()


def _normalise_daily_mapping(
    forecasts: Mapping[date | str, Sequence[float] | torch.Tensor],
    label: str,
) -> dict[date, torch.Tensor]:
    result: dict[date, torch.Tensor] = {}
    for key, value in forecasts.items():
        delivery_date = pd.Timestamp(key).date()
        prediction = torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(-1)
        if prediction.numel() != 24:
            raise ValueError(
                f"{delivery_date}的{label}应有24个小时，实际为{prediction.numel()}"
            )
        result[delivery_date] = prediction
    return result


def _make_simulator(cfg) -> BESSSimulator:
    return BESSSimulator(
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


@torch.no_grad()
def locked_dual_backtest(
    da_price_forecasts: Mapping[date | str, Sequence[float] | torch.Tensor],
    rt_price_forecasts,
    realized_da_price,
    realized_rt_price,
    local_timestamps: Sequence,
    cfg,
    *,
    da_k_charge: int = 4,
    da_k_discharge: int = 4,
    rt_k_charge: int = 1,
    rt_k_discharge: int = 1,
    use_deviation_penalty: bool | None = None,
    coordination_mode: str = "rt_only",
    rt_at_da_price_forecasts: (
        Mapping[date | str, Sequence[float] | torch.Tensor] | None
    ) = None,
) -> dict:
    """回测锁定的日计划和逐小时 RT 动作，返回可审计的收益分解。

    只使用同时满足以下条件的交付日：有 DA 计划、RT 数据含完整当地 24
    小时、这些小时在 UTC 上连续。夏令时的 23/25 小时日不伪装成 24 小时
    日。若中间出现缺日，则在下一连续段开始时重置两条 SOC，并明确报告。
    """
    da_forecasts = _normalise_daily_mapping(da_price_forecasts, "DA预测")
    rt_at_da_forecasts = (
        _normalise_daily_mapping(rt_at_da_price_forecasts, "RT-at-DA预测")
        if rt_at_da_price_forecasts is not None else None
    )
    rt_forecasts = _as_2d_float(rt_price_forecasts, "rt_price_forecasts")
    price_da = _as_1d_float(realized_da_price, "realized_da_price")
    price_rt = _as_1d_float(realized_rt_price, "realized_rt_price")
    if not (
        len(rt_forecasts) == len(price_da) == len(price_rt) == len(local_timestamps)
    ):
        raise ValueError("RT预测、DA/RT真值和时间戳长度必须一致")
    if len(rt_forecasts) == 0:
        raise ValueError("回测输入不能为空")

    timestamps = [pd.Timestamp(value) for value in local_timestamps]
    if any(timestamp.tzinfo is None for timestamp in timestamps):
        raise ValueError("local_timestamps必须带时区，避免夏令时歧义")
    utc_timestamps = [timestamp.tz_convert("UTC") for timestamp in timestamps]
    if any(right <= left for left, right in zip(utc_timestamps, utc_timestamps[1:])):
        raise ValueError("local_timestamps必须严格递增")

    positions_by_date: dict[date, list[int]] = defaultdict(list)
    for position, timestamp in enumerate(timestamps):
        positions_by_date[timestamp.date()].append(position)
    one_hour = pd.Timedelta(hours=1)
    eligible: dict[date, list[int]] = {}
    excluded_incomplete = []
    excluded_without_plan = []
    for delivery_date, positions in positions_by_date.items():
        local_hours = [timestamps[position].hour for position in positions]
        utc_hours = [utc_timestamps[position] for position in positions]
        is_contiguous = all(
            right - left == one_hour for left, right in zip(utc_hours, utc_hours[1:])
        )
        if not (
            len(positions) == 24
            and sorted(local_hours) == list(range(24))
            and is_contiguous
        ):
            excluded_incomplete.append(str(delivery_date))
            continue
        if delivery_date not in da_forecasts:
            excluded_without_plan.append(str(delivery_date))
            continue
        if (
            rt_at_da_forecasts is not None
            and delivery_date not in rt_at_da_forecasts
        ):
            excluded_without_plan.append(str(delivery_date))
            continue
        eligible[delivery_date] = positions
    if not eligible:
        raise ValueError("没有同时具备完整24小时RT数据和DA计划的交付日")

    threshold = float(cfg.resolved_spread_threshold)
    da_policy = HardTopKPolicy(
        da_k_charge, da_k_discharge, spread_threshold=threshold
    )
    rt_policy = HardTopKPolicy(
        rt_k_charge, rt_k_discharge, spread_threshold=threshold
    )
    simulator = _make_simulator(cfg)
    mpc_policy = LookaheadMPCPolicy(
        cfg.bess_power_mw,
        cfg.bess_energy_mwh,
        cfg.bess_eta,
        cfg.bess_init_soc_frac,
        kappa=cfg.bess_kappa,
        soc_min=cfg.bess_soc_min,
        soc_max=cfg.bess_soc_max,
        e_cyc=cfg.bess_e_cyc,
    )
    if coordination_mode not in {"rt_only", "follow_da", "lookahead_mpc"}:
        raise ValueError(f"未知RT协调方式: {coordination_mode}")
    rt_candidates_all = rt_policy(rt_forecasts)[:, 0]

    ordered_dates = sorted(eligible)
    initial_soc = cfg.bess_energy_mwh * cfg.bess_init_soc_frac
    plan_soc = float(initial_soc)
    actual_soc = float(initial_soc)
    previous_last_utc = None
    segment_count = 0
    segment_start_dates = []
    segment_final_actual_soc = []
    segment_final_plan_soc = []
    da_actions: dict[date, torch.Tensor] = {}
    plan_clipped_by_date: dict[date, float] = {}
    plan_soc_end_by_date: dict[date, float] = {}

    # 先按日期锁定并投影 DA 计划。计划 SOC 跨连续日传递；遇到数据缺口才重置。
    for delivery_date in ordered_dates:
        positions = eligible[delivery_date]
        first_utc = utc_timestamps[positions[0]]
        starts_segment = (
            previous_last_utc is None or first_utc - previous_last_utc != one_hour
        )
        if starts_segment:
            if segment_count:
                segment_final_plan_soc.append(plan_soc)
            segment_count += 1
            segment_start_dates.append(str(delivery_date))
            plan_soc = float(initial_soc)
        da_signal = da_forecasts[delivery_date]
        if rt_at_da_forecasts is not None:
            da_signal = da_signal - rt_at_da_forecasts[delivery_date]
        intended_plan = da_policy(da_signal.reshape(1, 24))
        projection = simulator.project_actions(intended_plan, plan_soc)
        feasible_plan = projection.action[0]
        da_actions[delivery_date] = feasible_plan
        plan_soc = float(projection.soc_path_mwh[0, -1])
        plan_soc_end_by_date[delivery_date] = plan_soc
        plan_clipped_by_date[delivery_date] = float(projection.clipped_energy_mwh[0])
        previous_last_utc = utc_timestamps[positions[-1]]
    segment_final_plan_soc.append(plan_soc)

    penalty_enabled = (
        bool(cfg.use_deviation_penalty)
        if use_deviation_penalty is None
        else bool(use_deviation_penalty)
    )
    daily = {}
    all_actual_actions = []
    all_da_actions = []
    changed_from_da_hours = 0
    previous_last_utc = None
    actual_segment_index = 0

    for delivery_date in ordered_dates:
        positions = eligible[delivery_date]
        first_utc = utc_timestamps[positions[0]]
        starts_segment = (
            previous_last_utc is None or first_utc - previous_last_utc != one_hour
        )
        if starts_segment:
            if actual_segment_index:
                segment_final_actual_soc.append(actual_soc)
            actual_segment_index += 1
            actual_soc = float(initial_soc)
        discharged_today = 0.0
        da_leg_total = 0.0
        rt_leg_total = 0.0
        degradation_total = 0.0
        penalty_total = 0.0
        actual_throughput = 0.0
        plan = da_actions[delivery_date]

        for hour, position in enumerate(positions):
            da_action = float(plan[hour])
            rt_candidate = float(rt_candidates_all[position])
            if coordination_mode == "lookahead_mpc":
                horizon = int(rt_forecasts.shape[1])
                future_da = torch.zeros((1, horizon), dtype=torch.float32)
                cycle_reset = torch.zeros((1, horizon), dtype=torch.bool)
                for step in range(horizon):
                    future_position = position + step
                    if future_position >= len(timestamps):
                        break
                    if step > 0:
                        expected_utc = utc_timestamps[position] + step * one_hour
                        if utc_timestamps[future_position] != expected_utc:
                            break
                        if timestamps[future_position].date() != timestamps[future_position - 1].date():
                            cycle_reset[0, step] = True
                    future_date = timestamps[future_position].date()
                    if future_date in da_actions:
                        future_da[0, step] = da_actions[future_date][
                            timestamps[future_position].hour
                        ]
                rt_intended = float(mpc_policy(
                    rt_forecasts[position].reshape(1, -1),
                    initial_soc_mwh=actual_soc,
                    initial_discharged_mwh=discharged_today,
                    da_plan=future_da,
                    use_deviation_penalty=penalty_enabled,
                    cycle_reset_mask=cycle_reset,
                )[0, 0])
            elif coordination_mode == "follow_da":
                rt_intended = da_action
            else:
                rt_intended = rt_candidate
            changed_from_da_hours += int(abs(rt_intended - da_action) > 1e-8)
            requested_discharge = max(rt_intended, 0.0) * cfg.bess_power_mw
            requested_charge = max(-rt_intended, 0.0) * cfg.bess_power_mw
            actual_discharge = min(
                requested_discharge,
                max(0.0, (actual_soc - cfg.bess_soc_min) * cfg.bess_eta),
                max(0.0, cfg.bess_e_cyc - discharged_today),
            )
            actual_charge = min(
                requested_charge,
                max(0.0, (cfg.bess_soc_max - actual_soc) / cfg.bess_eta),
            )
            actual_net = actual_discharge - actual_charge
            da_net = da_action * cfg.bess_power_mw
            p_da = float(price_da[position])
            p_rt = float(price_rt[position])
            da_leg = da_net * p_da
            rt_leg = (actual_net - da_net) * p_rt
            degradation = cfg.bess_kappa * (actual_discharge + actual_charge)
            penalty = 0.0
            if penalty_enabled:
                tolerance = 0.03 * abs(da_net)
                excess = max(0.0, abs(actual_net - da_net) - tolerance)
                penalty = 2.0 * abs(p_rt) * excess

            actual_soc = (
                actual_soc
                - actual_discharge / cfg.bess_eta
                + actual_charge * cfg.bess_eta
            )
            discharged_today += actual_discharge
            da_leg_total += da_leg
            rt_leg_total += rt_leg
            degradation_total += degradation
            penalty_total += penalty
            actual_throughput += actual_discharge + actual_charge
            all_actual_actions.append(actual_net / cfg.bess_power_mw)
            all_da_actions.append(da_action)

        net = da_leg_total + rt_leg_total - degradation_total - penalty_total
        daily[str(delivery_date)] = {
            "net_revenue": net,
            "da_leg": da_leg_total,
            "rt_deviation_leg": rt_leg_total,
            "degradation_cost": degradation_total,
            "deviation_penalty": penalty_total,
            "actual_throughput_mwh": actual_throughput,
            "actual_soc_end_mwh": actual_soc,
            "plan_soc_end_mwh": plan_soc_end_by_date[delivery_date],
            "plan_clipped_mwh": plan_clipped_by_date[delivery_date],
        }
        previous_last_utc = utc_timestamps[positions[-1]]
    segment_final_actual_soc.append(actual_soc)

    daily_values = list(daily.values())
    total_da = sum(row["da_leg"] for row in daily_values)
    total_rt = sum(row["rt_deviation_leg"] for row in daily_values)
    total_cost = sum(row["degradation_cost"] for row in daily_values)
    total_penalty = sum(row["deviation_penalty"] for row in daily_values)
    total_revenue = total_da + total_rt - total_cost - total_penalty
    return {
        "contract": "locked 24h DA plan + hourly rolling RT first action + dual settlement",
        "da_decision_signal": (
            "predicted_DA_minus_predicted_RT_at_DA"
            if rt_at_da_forecasts is not None else "legacy_predicted_DA_only"
        ),
        "coordination": {
            "mode": coordination_mode,
            "changed_from_da_hours": changed_from_da_hours,
            "changed_from_da_rate": changed_from_da_hours / (24 * len(daily_values)),
        },
        "days": len(daily_values),
        "hours": 24 * len(daily_values),
        "segments": segment_count,
        "segment_start_dates": segment_start_dates,
        "state_resets_due_to_gaps": max(0, segment_count - 1),
        "excluded_incomplete_or_dst_dates": excluded_incomplete,
        "excluded_complete_dates_without_da_plan": excluded_without_plan,
        "total_revenue": total_revenue,
        "mean_daily_revenue": total_revenue / len(daily_values),
        "positive_day_rate": sum(row["net_revenue"] > 0 for row in daily_values)
        / len(daily_values),
        "revenue_components": {
            "da_leg": total_da,
            "rt_deviation_leg": total_rt,
            "degradation_cost": total_cost,
            "deviation_penalty": total_penalty,
        },
        "plan_feasibility": {
            "total_clipped_mwh": sum(plan_clipped_by_date.values()),
            "clipped_days": sum(value > 1e-8 for value in plan_clipped_by_date.values()),
            "segment_final_soc_mwh": segment_final_plan_soc,
        },
        "actual_state": {
            "segment_final_soc_mwh": segment_final_actual_soc,
            "action_digest": _digest(all_actual_actions),
        },
        "da_plan_action_digest": _digest(all_da_actions),
        "terminal_inventory_note": (
            "未做人为期末估值；跨DA模型比较时固定同一RT动作，因此实际SOC与期末库存完全相同，"
            "该项会相互抵消。比较不同RT模型前必须另行统一期末库存规则。"
        ),
        "daily": daily,
    }
