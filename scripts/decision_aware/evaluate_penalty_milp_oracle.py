"""Evaluate the exact deviation-penalty Oracle on the unified validation set.

This script is intentionally evaluation-only: it never loads or updates model
weights.  It first benchmarks one and seven continuous days, then (unless
``--benchmark-only`` is supplied) solves every uninterrupted validation
segment with continuous planned/actual SOC.  Existing system comparison JSON
can be augmented with total Regret and PCR when it contains date-keyed daily
net revenue.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_joint import build_joint_datasets  # noqa: E402
from decision_aware.loader_v2 import load_market_unified, market_timezone  # noqa: E402
from decision_aware.oracle_continuous import (  # noqa: E402
    solve_dual_penalty_milp_segment,
)
from decision_aware.policy import BESSSimulator  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/decision_aware/joint_decision_aware_v3_penalty.yaml",
    )
    parser.add_argument(
        "--comparison-json",
        default=(
            "data/results/"
            "joint_system_baseline_comparison_v5_penalty_kappa57_3seeds.json"
        ),
        help="可选：含逐日收益的已有系统比较JSON",
    )
    parser.add_argument(
        "--output",
        default="data/results/penalty_milp_oracle_validation_v1.json",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="只跑1日/7日benchmark，不解完整验证集",
    )
    parser.add_argument(
        "--max-estimated-full-seconds",
        type=float,
        default=1800.0,
        help="7日线性外推超过此值时安全跳过全量；<=0表示不设限",
    )
    parser.add_argument(
        "--mip-rel-gap",
        type=float,
        default=1e-6,
        help="HiGHS最优性相对间隙；状态中会保存实际mip_gap",
    )
    return parser.parse_args()


def _truth_contract(joint_dataset) -> dict:
    windows = joint_dataset.windows
    da = np.stack(
        [
            joint_dataset.da_dataset[window.da_index]["price_da_tgt"].numpy()
            for window in windows
        ]
    ).astype(np.float64)
    rt = np.stack(
        [
            joint_dataset.rt_at_da_dataset[window.rt_at_da_index][
                "price_rt_at_da_tgt"
            ].numpy()
            for window in windows
        ]
    ).astype(np.float64)
    return {
        "dates": [str(window.delivery_date) for window in windows],
        "starts_new_segment": np.asarray(
            [window.starts_new_segment for window in windows], dtype=bool
        ),
        "price_da": da,
        "price_rt": rt,
    }


def _truth_contract_from_dates(cfg: PilotConfig, dates: list[str]) -> dict:
    """Load only price truth while preserving the joint dataset's date contract.

    The comparison artifact already records the exact 182 accepted delivery
    dates.  Reusing that date list avoids constructing all train/validation/test
    feature tensors merely to read two price columns.
    """
    import pandas as pd

    wide = load_market_unified(
        market=cfg.market,
        node=cfg.node,
        start="2020-01-01",
        end="2026-06-02",
    )
    timezone = market_timezone(cfg.market)
    local_dates = np.asarray(wide.index.tz_convert(timezone).date, dtype=object)
    all_da = wide["price_da"].to_numpy(dtype=np.float64)
    all_rt = wide["price_rt"].to_numpy(dtype=np.float64)
    da_rows = []
    rt_rows = []
    starts_new_segment = []
    previous_last_utc = None
    one_hour = pd.Timedelta(hours=1)
    for date_text in dates:
        delivery_date = pd.Timestamp(date_text).date()
        positions = np.flatnonzero(local_dates == delivery_date)
        if positions.size != 24:
            raise ValueError(
                f"{date_text}在统一市场表中不是24个完整小时"
            )
        timestamps = wide.index[positions]
        if np.any(np.diff(timestamps.as_unit("ns").asi8) != one_hour.value):
            raise ValueError(f"{date_text}的UTC小时不连续")
        starts_new_segment.append(
            previous_last_utc is None
            or timestamps[0] - previous_last_utc != one_hour
        )
        previous_last_utc = timestamps[-1]
        da_rows.append(all_da[positions])
        rt_rows.append(all_rt[positions])
    return {
        "dates": list(dates),
        "starts_new_segment": np.asarray(starts_new_segment, dtype=bool),
        "price_da": np.stack(da_rows),
        "price_rt": np.stack(rt_rows),
    }


def _simulator(cfg: PilotConfig) -> BESSSimulator:
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


def _solve_timed(
    da,
    rt,
    simulator,
    segment_index: int,
    mip_rel_gap: float,
) -> tuple[object, float]:
    start = time.perf_counter()
    result = solve_dual_penalty_milp_segment(
        da,
        rt,
        simulator,
        segment_index=segment_index,
        mip_rel_gap=mip_rel_gap,
    )
    return result, time.perf_counter() - start


def _status_record(result, runtime_seconds: float) -> dict:
    return {
        **asdict(result.status),
        "runtime_seconds": float(runtime_seconds),
    }


def _segment_slices(starts_new_segment: np.ndarray) -> list[slice]:
    reset = np.asarray(starts_new_segment, dtype=bool).reshape(-1)
    if reset.size == 0:
        raise ValueError("验证集不能为空")
    if not reset[0]:
        raise ValueError("第一个验证日必须标记为新连续段")
    boundaries = np.flatnonzero(reset).tolist() + [int(reset.size)]
    return [
        slice(int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]


def _comparison_contract_errors(comparison: dict, contract: dict, cfg) -> list[str]:
    source = comparison.get("contract", {})
    errors = []
    expected_dates = contract["dates"]
    if source.get("validation_dates") != expected_dates:
        errors.append("比较JSON的验证日期与Oracle数据不同")
    if source.get("market") != cfg.market or source.get("node") != cfg.node:
        errors.append("比较JSON的市场/节点与Oracle配置不同")
    if source.get("deviation_penalty_enabled") is not True:
        errors.append("比较JSON没有开启偏差罚金")
    bess = source.get("bess", {})
    numeric_contract = {
        "power_mw": cfg.bess_power_mw,
        "energy_mwh": cfg.bess_energy_mwh,
        "eta": cfg.bess_eta,
        "soc_min_mwh": cfg.bess_soc_min,
        "soc_max_mwh": cfg.bess_soc_max,
        "degradation_usd_per_mwh": cfg.bess_kappa,
        "daily_discharge_limit_mwh": cfg.bess_e_cyc,
    }
    for key, expected in numeric_contract.items():
        actual = bess.get(key)
        if actual is None or not np.isclose(float(actual), float(expected)):
            errors.append(f"比较JSON的bess.{key}与Oracle配置不同")
    return errors


def _score_existing_systems(
    comparison_path: Path,
    dates: list[str],
    oracle_daily: np.ndarray,
    cfg: PilotConfig,
    contract: dict,
) -> dict:
    if not comparison_path.exists():
        return {
            "status": "comparison_json_not_found",
            "path": str(comparison_path),
            "systems": {},
        }
    with comparison_path.open("r", encoding="utf-8") as file:
        comparison = json.load(file)
    errors = _comparison_contract_errors(comparison, contract, cfg)
    if errors:
        return {
            "status": "contract_mismatch",
            "path": str(comparison_path),
            "errors": errors,
            "systems": {},
        }

    oracle_total = float(np.sum(oracle_daily))
    systems = {}
    for name, metrics in comparison.get("systems", {}).items():
        daily_mapping = metrics.get("daily_net_revenue")
        if not isinstance(daily_mapping, dict):
            systems[name] = {
                "status": "daily_net_revenue_missing",
                "regret": None,
                "pcr_percent": None,
            }
            continue
        missing = [date for date in dates if date not in daily_mapping]
        extra = [date for date in daily_mapping if date not in set(dates)]
        if missing or extra:
            systems[name] = {
                "status": "daily_dates_mismatch",
                "missing_dates": missing,
                "extra_dates": extra,
                "regret": None,
                "pcr_percent": None,
            }
            continue
        model_daily = np.asarray(
            [float(daily_mapping[date]) for date in dates], dtype=np.float64
        )
        model_total = float(np.sum(model_daily))
        total_regret = oracle_total - model_total
        systems[name] = {
            "status": "ok",
            "days": len(dates),
            "model_total_net_revenue_usd": model_total,
            "model_mean_daily_net_revenue_usd": float(np.mean(model_daily)),
            "oracle_total_net_revenue_usd": oracle_total,
            "total_regret_usd": total_regret,
            "mean_daily_regret_usd": total_regret / len(dates),
            "pcr_percent": (
                100.0 * model_total / oracle_total if oracle_total > 0.0 else None
            ),
            "oracle_upper_bound_check": bool(model_total <= oracle_total + 1e-5),
        }
    return {
        "status": "ok",
        "path": str(comparison_path),
        "metric_note": (
            "连续SOC使每日的Oracle收益受前后日联动决策影响；"
            "Regret和PCR以完整期间总收益计算，不宣称逐日regret必然非负。"
        ),
        "systems": systems,
    }


def main() -> None:
    args = _parse_args()
    cfg = PilotConfig.from_yaml(args.config)
    if not cfg.use_deviation_penalty:
        raise ValueError("Oracle评估配置必须开启use_deviation_penalty")

    comparison_path = Path(args.comparison_json)
    comparison_dates = None
    if comparison_path.exists():
        with comparison_path.open("r", encoding="utf-8") as file:
            comparison_header = json.load(file)
        candidate_dates = comparison_header.get("contract", {}).get(
            "validation_dates"
        )
        if isinstance(candidate_dates, list) and candidate_dates:
            comparison_dates = [str(value) for value in candidate_dates]
    if comparison_dates is not None:
        contract = _truth_contract_from_dates(cfg, comparison_dates)
        truth_source = "comparison_validation_dates_plus_unified_market_prices"
    else:
        _, validation, _, _ = build_joint_datasets(cfg)
        contract = _truth_contract(validation)
        truth_source = "rebuilt_joint_validation_dataset"
    print(
        json.dumps(
            {
                "event": "oracle_truth_loaded",
                "days": len(contract["dates"]),
                "truth_source": truth_source,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    dates = contract["dates"]
    da = contract["price_da"]
    rt = contract["price_rt"]
    simulator = _simulator(cfg)

    one_day, one_seconds = _solve_timed(
        da[:1], rt[:1], simulator, 0, args.mip_rel_gap
    )
    print(
        json.dumps(
            {"event": "oracle_benchmark_1d", "seconds": one_seconds},
            ensure_ascii=False,
        ),
        flush=True,
    )
    seven_count = min(7, len(dates))
    seven_day, seven_seconds = _solve_timed(
        da[:seven_count], rt[:seven_count], simulator, 0, args.mip_rel_gap
    )
    print(
        json.dumps(
            {"event": "oracle_benchmark_7d", "seconds": seven_seconds},
            ensure_ascii=False,
        ),
        flush=True,
    )
    estimated_full_seconds = seven_seconds * len(dates) / seven_count
    benchmark = {
        "one_day": {
            "days": 1,
            "total_revenue_usd": one_day.revenue,
            "status": _status_record(one_day, one_seconds),
        },
        "seven_day": {
            "days": seven_count,
            "total_revenue_usd": seven_day.revenue,
            "status": _status_record(seven_day, seven_seconds),
        },
        "linear_runtime_estimate_full_seconds": float(estimated_full_seconds),
        "estimate_warning": (
            "MILP复杂度不保证线性，这只是启动全量求解前的粗略尺度。"
        ),
    }

    output = {
        "experiment": "exact_penalty_milp_oracle_continuous_validation_v1",
        "status": "benchmark_only",
        "config_path": str(Path(args.config)),
        "truth_source": truth_source,
        "contract": {
            "market": cfg.market,
            "node": cfg.node,
            "validation_days": len(dates),
            "validation_dates": dates,
            "hours_per_day": int(da.shape[1]),
            "continuous_segments": len(_segment_slices(contract["starts_new_segment"])),
            "initial_soc_mwh_each_segment": (
                cfg.bess_energy_mwh * cfg.bess_init_soc_frac
            ),
            "soc_rule": (
                "每个连续数据段内跨日继承计划SOC和实际SOC；"
                "只在starts_new_segment=true时重置"
            ),
            "daily_discharge_limit_rule": "DA和RT轨迹每个交付日各自重置",
            "terminal_soc_rule": "只报告MWh，不强制回到初值，不货币化",
            "oracle_information": "每个连续段全时域完美预见",
            "solver_mip_relative_gap_tolerance": float(args.mip_rel_gap),
            "settlement_formula": (
                "DA*u_DA + RT*(u_RT-u_DA) - kappa*abs(u_RT) - "
                "2*abs(RT)*max(abs(u_RT-u_DA)-0.03*abs(u_DA),0)"
            ),
            "bess": {
                "power_mw": cfg.bess_power_mw,
                "energy_mwh": cfg.bess_energy_mwh,
                "eta": cfg.bess_eta,
                "soc_min_mwh": cfg.bess_soc_min,
                "soc_max_mwh": cfg.bess_soc_max,
                "degradation_usd_per_mwh": cfg.bess_kappa,
                "daily_discharge_limit_mwh": cfg.bess_e_cyc,
            },
        },
        "benchmark": benchmark,
        "full_validation": None,
        "existing_system_regret_pcr": {
            "status": "requires_full_oracle",
            "systems": {},
        },
    }

    limit = float(args.max_estimated_full_seconds)
    should_run_full = not args.benchmark_only and (
        limit <= 0.0 or estimated_full_seconds <= limit
    )
    if should_run_full:
        segment_records = []
        daily_parts = []
        da_leg_parts = []
        rt_leg_parts = []
        degradation_parts = []
        penalty_parts = []
        total_start = time.perf_counter()
        for segment_index, segment in enumerate(
            _segment_slices(contract["starts_new_segment"])
        ):
            print(
                json.dumps(
                    {
                        "event": "oracle_segment_started",
                        "segment_index": segment_index,
                        "days": segment.stop - segment.start,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            solved, elapsed = _solve_timed(
                da[segment],
                rt[segment],
                simulator,
                segment_index,
                args.mip_rel_gap,
            )
            print(
                json.dumps(
                    {
                        "event": "oracle_segment_complete",
                        "segment_index": segment_index,
                        "seconds": elapsed,
                        "revenue": solved.revenue,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            dispatch = solved.dispatch
            daily_parts.append(dispatch.daily_revenue)
            da_leg_parts.append(dispatch.daily_da_leg)
            rt_leg_parts.append(dispatch.daily_rt_deviation_leg)
            degradation_parts.append(dispatch.daily_degradation_cost)
            penalty_parts.append(dispatch.daily_deviation_penalty)
            segment_records.append(
                {
                    "segment_index": segment_index,
                    "start_day_index": segment.start,
                    "stop_day_index_exclusive": segment.stop,
                    "start_date": dates[segment.start],
                    "end_date": dates[segment.stop - 1],
                    "days": segment.stop - segment.start,
                    "initial_plan_soc_mwh": float(
                        dispatch.da_soc_path_mwh[0]
                    ),
                    "terminal_plan_soc_mwh": float(
                        dispatch.da_soc_path_mwh[-1]
                    ),
                    "initial_actual_soc_mwh": float(
                        dispatch.rt_soc_path_mwh[0]
                    ),
                    "terminal_actual_soc_mwh": float(
                        dispatch.rt_soc_path_mwh[-1]
                    ),
                    "total_revenue_usd": solved.revenue,
                    "status": _status_record(solved, elapsed),
                }
            )
        total_runtime = time.perf_counter() - total_start
        oracle_daily = np.concatenate(daily_parts)
        daily_da_leg = np.concatenate(da_leg_parts)
        daily_rt_leg = np.concatenate(rt_leg_parts)
        daily_degradation = np.concatenate(degradation_parts)
        daily_penalty = np.concatenate(penalty_parts)
        if oracle_daily.size != len(dates):
            raise RuntimeError("Oracle逐日输出长度与验证日期不等")

        output["status"] = "complete"
        output["full_validation"] = {
            "solver_status_optimal_all_segments": True,
            "certified_optimal_within_configured_mip_gap": True,
            "mathematical_zero_gap_claim": bool(args.mip_rel_gap == 0.0),
            "optimality_note": (
                "HiGHS在配置的MIP相对间隙内返回Optimal；"
                "mip_rel_gap>0时不声称数学零间隙。"
            ),
            "days": len(dates),
            "segments": segment_records,
            "runtime_seconds": float(total_runtime),
            "total_net_revenue_usd": float(np.sum(oracle_daily)),
            "mean_daily_net_revenue_usd": float(np.mean(oracle_daily)),
            "components_total_usd": {
                "da_leg": float(np.sum(daily_da_leg)),
                "rt_deviation_leg": float(np.sum(daily_rt_leg)),
                "degradation_cost": float(np.sum(daily_degradation)),
                "deviation_penalty": float(np.sum(daily_penalty)),
            },
            "daily_net_revenue_usd": {
                date: float(value) for date, value in zip(dates, oracle_daily)
            },
            "daily_allocation_note": (
                "这是全连续段联合最优解在各日的现金收益分解，"
                "不是把每日SOC重置后单独求解的Oracle。"
            ),
        }
        output["existing_system_regret_pcr"] = _score_existing_systems(
            comparison_path, dates, oracle_daily, cfg, contract
        )
    elif not args.benchmark_only:
        output["status"] = "full_skipped_by_runtime_guard"
        output["full_validation"] = {
            "solver_status_optimal_all_segments": False,
            "certified_optimal_within_configured_mip_gap": False,
            "mathematical_zero_gap_claim": False,
            "skip_reason": (
                f"线性外推{estimated_full_seconds:.3f}s超过安全阈值{limit:.3f}s"
            ),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"为避免覆盖旧结果，输出文件必须不存在: {output_path}")
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "event": "penalty_milp_oracle_complete",
                "status": output["status"],
                "output": str(output_path),
                "benchmark": benchmark,
                "full_validation": output["full_validation"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
