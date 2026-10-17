#!/usr/bin/env python
"""生成负价/尖峰/极端价差及ERCOT跨年盈利失稳诊断。

本脚本只读取已经冻结的预测和历史训练段真值，不训练模型。2025始终不读。
价格百分位阈值来自对应市场的预评价历史段；它们只定义报告切片，不进入模型。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.continuous_mpc import locked_continuous_mpc_backtest  # noqa: E402
from decision_aware.contracts import FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_evaluation import align_node_task_predictions  # noqa: E402
from decision_aware.loader_v2 import market_timezone, resolve_market_unified_path  # noqa: E402
from decision_aware.tail_diagnostics import (  # noqa: E402
    TailThresholds,
    fit_tail_thresholds,
    price_slice_report,
    settlement_slice_report,
    spread_slice_report,
)
from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _node_baseline_archives,
    _process_safe,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
    _node_record,
)


TASKS = ("da", "rt_at_da", "rt")
MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")


def _load_simple(path: Path, task: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        result = {
            "path": str(path),
            "nodes": archive["nodes"].astype(str),
            "dates": archive["dates"].astype(str),
            "prediction": archive["prediction"].astype(np.float32),
            "target": archive["target"].astype(np.float32),
        }
        if "target_da" in archive.files:
            result["target_da"] = archive["target_da"].astype(np.float32)
        if "metadata" in archive.files:
            result["metadata"] = json.loads(str(archive["metadata"].item()))
    if task == "da":
        result["target_da"] = result["target"]
    else:
        result["target_rt"] = result["target"]
    return result


def _simple_triplet(market: str, phase: str) -> dict[str, dict]:
    suffix = "_outer_frozen_v1" if phase == "outer" else "_v1"
    result = {}
    for task in TASKS:
        path = (
            ROOT / "data" / "results" / f"universal_lomo_{task}{suffix}"
            / f"{market.lower()}_{task}_v1.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        result[task] = _load_simple(path, task)
    return result


def _pooled_xgboost_triplet(market: str) -> dict[str, dict]:
    """读取只由另外三个源市场训练的公平 pooled LOMO XGBoost。"""
    result = {}
    for task in TASKS:
        path = (
            ROOT / "data" / "results" / "universal_lomo_xgboost_v1"
            / f"holdout_{market.lower()}_{task}_inner_seed0.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        item = _load_simple(path, task)
        metadata = item.get("metadata", {})
        if metadata.get("protocol") != "pooled_lomo_xgboost_equal_training_years":
            raise ValueError(f"{path}不是公平pooled LOMO XGBoost产物")
        if metadata.get("uses_target_market_training_labels") is not False:
            raise ValueError(f"{path}未明确禁止目标市场训练标签")
        if metadata.get("uses_node_id_for_prediction") is not False:
            raise ValueError(f"{path}未明确禁止node_id预测输入")
        if metadata.get("uses_market_id_for_prediction") is not False:
            raise ValueError(f"{path}未明确禁止market_id预测输入")
        result[task] = item
    return result


def _tsfm_triplet(market: str, system: str) -> dict[str, dict]:
    """读取已冻结的TimesFM四折预测。

    TimesFM-3是官方frozen基线；TimesFM-2.5是只用三个源市场
    监督数据微调、在留出市场零样本评价的LoRA基线。两者都只有
    2023 inner协议，不允许冒充2024 outer结果。
    """
    if system == "timesfm3":
        directory = ROOT / "data/results/universal_lomo_timesfm3_v1"
        filename = lambda task: f"{market.lower()}_{task}_v1.npz"
        expected_protocol = "frozen_foundation_model_same_information_contract"
    elif system == "timesfm25_lora":
        directory = ROOT / "data/results/universal_lomo_timesfm25_lora_v1"
        filename = lambda task: f"holdout_{market.lower()}_{task}_inner_seed0.npz"
        expected_protocol = (
            "timesfm25_source_market_supervised_lora_target_market_zero_shot"
        )
    else:
        raise ValueError(f"未知TSFM系统: {system}")

    result = {}
    for task in TASKS:
        path = directory / filename(task)
        if not path.is_file():
            raise FileNotFoundError(path)
        item = _load_simple(path, task)
        metadata = item.get("metadata", {})
        if metadata.get("protocol") != expected_protocol:
            raise ValueError(f"{path}的TSFM协议不匹配")
        if metadata.get("test_2025") is not False:
            raise ValueError(f"{path}未明确禁止2025测试")
        if metadata.get("uses_node_id_for_prediction", False):
            raise ValueError(f"{path}使用了node_id预测输入")
        if metadata.get("uses_market_id_for_prediction", False):
            raise ValueError(f"{path}使用了market_id预测输入")
        if system == "timesfm25_lora":
            if metadata.get("uses_holdout_labels_for_training") is not False:
                raise ValueError(f"{path}未明确禁止留出市场标签")
            if metadata.get("selection_uses_holdout") is not False:
                raise ValueError(f"{path}未明确禁止留出市场选模")
        result[task] = item
    return result


def _attach_settlement_contract(
    candidate: dict[str, dict], reference: dict[str, dict]
) -> dict[str, dict]:
    """用同一评价样本的只读结算字段补全预测归档。

    pooled LOMO XGBoost 的通用NPZ合同只保存预测任务所需的真值，
    不重复保存DA归档中的真实RT价格，也不保存滚动RT的执行时间戳。
    LP/MPC需要这些字段，因此从当前Transformer使用的同节点、同日期冻结
    评价归档中复制。复制前逐任务核对节点、日期和任务真值，避免把不同
    样本或不同年份静默拼接到一起。候选预测本身不会被改变。
    """
    completed: dict[str, dict] = {}
    for task in TASKS:
        item = dict(candidate[task])
        contract = reference[task]
        if not np.array_equal(item["nodes"], contract["nodes"]):
            raise ValueError(f"{task}候选与结算参考的节点顺序不一致")
        if not np.array_equal(item["dates"], contract["dates"]):
            raise ValueError(f"{task}候选与结算参考的交付日不一致")
        target_key = "target_da" if task == "da" else "target_rt"
        # 两类NPZ由不同导出链路压缩为float32；在尖峰价上可出现约
        # 5e-4 USD/MWh的最后一位舍入差，不代表评价样本不同。
        if not np.allclose(
            item[target_key], contract[target_key], rtol=0, atol=1e-3
        ):
            raise ValueError(f"{task}候选与结算参考的任务真值不一致")
        # 通过一致性校验后，结算统一使用同一份参考真值，避免不同NPZ的
        # float32末位舍入触发三任务严格一致性保护。
        item[target_key] = contract[target_key]
        if task == "da":
            item["target_rt"] = contract["target_rt"]
        if task == "rt":
            item["executed_timestamps"] = contract["executed_timestamps"]
            item["timezone"] = contract["timezone"]
        completed[task] = item
    return completed


def _profit_triplet(directory: Path, family: str, phase: str, seed: int = 0) -> dict:
    return {
        task: _load_archive(directory, family, task, phase)
        for task in TASKS
    }


def _target_history(market: str, nodes, end_year: int) -> tuple[np.ndarray, np.ndarray]:
    columns = [
        "timestamp_utc", "node", "day_ahead_price_usd_mwh",
        "real_time_price_usd_mwh",
    ]
    frame = pd.read_parquet(
        resolve_market_unified_path(market), columns=columns,
        filters=[("node", "in", list(nodes))],
    )
    timestamp = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.tz_convert(
        market_timezone(market)
    )
    mask = (
        (timestamp.dt.date >= pd.Timestamp("2020-01-01").date())
        & (timestamp.dt.date <= pd.Timestamp(f"{end_year}-12-31").date())
    )
    selected = frame.loc[mask]
    return (
        selected["day_ahead_price_usd_mwh"].to_numpy(np.float64),
        selected["real_time_price_usd_mwh"].to_numpy(np.float64),
    )


def _thresholds(market: str, nodes, phase: str) -> TailThresholds:
    end_year = 2022 if phase == "inner" else 2023
    da, rt = _target_history(market, nodes, end_year)
    return fit_tail_thresholds(da, rt)


def _forecast_report(triplet: dict, thresholds: TailThresholds) -> dict:
    reports = {}
    for task in TASKS:
        item = triplet[task]
        # 预测主表沿用任务正式口径：滚动RT同时评价四个horizon；
        # 只有进入LP/MPC结算时才执行每个滚动窗口的第一步。
        prediction = item["prediction"]
        target = item["target"]
        is_da = task == "da"
        reports[task] = price_slice_report(
            prediction,
            target,
            lower_1pct=(
                thresholds.da_lower_1pct if is_da else thresholds.rt_lower_1pct
            ),
            upper_1pct=(
                thresholds.da_upper_1pct if is_da else thresholds.rt_upper_1pct
            ),
        )

    da = triplet["da"]
    rt_da = triplet["rt_at_da"]
    common_dates = sorted(set(da["dates"]) & set(rt_da["dates"]))
    da_index = {date: index for index, date in enumerate(da["dates"])}
    rt_index = {date: index for index, date in enumerate(rt_da["dates"])}
    da_rows = np.asarray([da_index[date] for date in common_dates])
    rt_rows = np.asarray([rt_index[date] for date in common_dates])
    reports["da_rt_at_da_spread"] = spread_slice_report(
        da["prediction"][:, da_rows],
        rt_da["prediction"][:, rt_rows],
        da["target"][:, da_rows],
        rt_da["target"][:, rt_rows],
        thresholds,
    )
    reports["clamp_audit"] = {
        "configured_lower_bound_usd_mwh": -500.0,
        "true_rt_below_lower_bound": int(np.sum(rt_da["target"] < -500.0)),
        "predicted_rt_at_da_at_lower_bound": int(np.sum(rt_da["prediction"] <= -499.999)),
        "predicted_rolling_rt_at_lower_bound": int(
            np.sum(triplet["rt"]["prediction"] <= -499.999)
        ),
    }
    return reports


def _profit_worker(
    config_path: str,
    system: str,
    phase: str,
    node: str,
    node_input: dict,
) -> tuple[str, str, str, dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = PilotConfig.from_yaml(config_path)
    report = locked_continuous_mpc_backtest(
        node_input["da_price_forecasts"],
        node_input["rt_at_da_price_forecasts"],
        node_input["rt_price_forecasts"],
        node_input["realized_da_price"],
        node_input["realized_rt_price"],
        node_input["local_timestamps"],
        cfg,
        include_hourly_details=True,
    )
    rows = report.pop("hourly")
    hourly = {
        field: np.asarray([row[field] for row in rows], dtype=np.float64)
        for field in rows[0]
        if field not in {"timestamp_local", "delivery_date"}
    }
    compact = {
        "days": report["days"],
        "daily": report["daily"],
        "revenue_components": report["revenue_components"],
        "hourly": hourly,
    }
    return system, phase, node, compact


def _node_inputs(triplet: dict) -> dict[str, dict]:
    nodes = triplet["da"]["nodes"]
    for task in TASKS:
        if not np.array_equal(nodes, triplet[task]["nodes"]):
            raise ValueError(f"{task}节点顺序不一致")
    return {
        str(node): align_node_task_predictions(
            _node_record(triplet["da"], index),
            _node_record(triplet["rt_at_da"], index),
            _node_record(triplet["rt"], index),
        )
        for index, node in enumerate(nodes)
    }


def _hourly_arrays(reports: dict[str, dict]) -> dict[str, np.ndarray]:
    fields = tuple(reports[next(iter(reports))]["hourly"])
    return {
        field: np.concatenate([
            reports[node]["hourly"][field] for node in sorted(reports)
        ])
        for field in fields
    }


def _decision_report(hourly: dict[str, np.ndarray]) -> dict:
    true_spread = hourly["realized_da_price"] - hourly["realized_rt_price"]
    predicted_spread = (
        hourly["predicted_da_price"] - hourly["predicted_rt_at_da_price"]
    )
    plan = hourly["da_plan_mwh"]
    nonzero_true = np.abs(true_spread) > 1e-8
    active = np.abs(plan) > 1e-8
    wrong = active & (plan * true_spread < 0.0)
    missed = ~active & nonzero_true
    return {
        "hours": int(true_spread.size),
        "spread_direction_accuracy": float(np.mean(
            np.sign(predicted_spread[nonzero_true]) == np.sign(true_spread[nonzero_true])
        )),
        "active_da_plan_rate": float(np.mean(active)),
        "wrong_da_plan_direction_hours": int(wrong.sum()),
        "wrong_da_plan_direction_rate_among_active": (
            float(wrong.sum() / active.sum()) if active.any() else None
        ),
        "inactive_with_nonzero_true_spread_hours": int(missed.sum()),
        "mean_absolute_true_spread": float(np.mean(np.abs(true_spread))),
        "mean_absolute_predicted_spread": float(np.mean(np.abs(predicted_spread))),
    }


def _aggregate_profit(
    reports: dict[str, dict], thresholds: TailThresholds
) -> dict:
    hourly = _hourly_arrays(reports)
    daily_dates = sorted(next(iter(reports.values()))["daily"])
    macro_daily = np.asarray([
        np.mean([reports[node]["daily"][day]["net_revenue"] for node in reports])
        for day in daily_dates
    ])
    node_daily_mean = {
        node: float(np.mean([
            reports[node]["daily"][day]["net_revenue"] for day in daily_dates
        ]))
        for node in sorted(reports)
    }
    cumulative = np.concatenate(([0.0], np.cumsum(macro_daily)))
    running_peak = np.maximum.accumulate(cumulative)
    max_drawdown = float(np.max(running_peak - cumulative))
    midpoint = len(macro_daily) // 2

    def without_top(count: int) -> float | None:
        if len(macro_daily) <= count:
            return None
        return float(np.mean(np.sort(macro_daily)[:-count]))

    components = {}
    for field in ("da_leg", "rt_deviation_leg", "degradation_cost", "deviation_penalty"):
        components[field] = float(np.mean([
            reports[node]["revenue_components"][field] / reports[node]["days"]
            for node in reports
        ]))
    return {
        "nodes": len(reports),
        "days": len(daily_dates),
        "macro_mean_daily_net_revenue": float(macro_daily.mean()),
        "macro_median_daily_net_revenue": float(np.median(macro_daily)),
        "macro_positive_day_rate": float(np.mean(macro_daily > 0.0)),
        "macro_max_drawdown_usd_per_node": max_drawdown,
        "macro_first_half_mean_daily_net_revenue": float(
            np.mean(macro_daily[:midpoint])
        ),
        "macro_second_half_mean_daily_net_revenue": float(
            np.mean(macro_daily[midpoint:])
        ),
        "macro_mean_daily_net_revenue_without_top_5_days": without_top(5),
        "macro_mean_daily_net_revenue_without_top_10_days": without_top(10),
        "node_mean_daily_net_revenue": node_daily_mean,
        "worst_node_mean_daily_net_revenue": min(node_daily_mean.values()),
        "best_node_mean_daily_net_revenue": max(node_daily_mean.values()),
        # 保留按日、跨节点宏平均序列，供系统间做严格配对的7日块Bootstrap。
        # 这不是额外评价数据；它只是上面均值所对应的可审计明细。
        "macro_daily_net_revenue": {
            day: float(value) for day, value in zip(daily_dates, macro_daily)
        },
        "macro_mean_daily_components": components,
        "decision_diagnostics": _decision_report(hourly),
        "settlement_slices": settlement_slice_report(hourly, thresholds),
    }


def _flatten(prefix: tuple[str, ...], value, rows: list[dict]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten((*prefix, str(key)), item, rows)
    elif isinstance(value, (int, float)) or value is None:
        rows.append({"path": "/".join(prefix), "value": value})


def _write_outputs(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tail_profit_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows: list[dict] = []
    _flatten((), report, rows)
    with (output_dir / "tail_profit_diagnostics_long.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "value"))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 负价、尖峰、极端价差与ERCOT跨年盈利失稳诊断",
        "",
        "> 百分位阈值只由预评价历史段冻结；2025未读取。",
        "",
        "## ERCOT完整系统盈利",
        "",
        "| 系统 | 年份 | 日均净收益 | DA腿 | RT腿 | 退化成本 | 偏差罚金 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system, phases in report["ercot_profit"].items():
        for phase, item in phases.items():
            comp = item["macro_mean_daily_components"]
            lines.append(
                f"| {system} | {'2023' if phase == 'inner' else '2024'} | "
                f"{item['macro_mean_daily_net_revenue']:.2f} | {comp['da_leg']:.2f} | "
                f"{comp['rt_deviation_leg']:.2f} | {comp['degradation_cost']:.2f} | "
                f"{comp['deviation_penalty']:.2f} |"
            )
    lines.extend((
        "", "## 解释边界", "",
        "切片收益用于定位差额出现在哪些真实价格状态，不把相关性写成网络结构因果。",
        "逐小时结算仍使用同一连续DA LP、滚动RT MPC和双结算合同。",
    ))
    (output_dir / "tail_profit_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/global_ercot_huber_v1.yaml"
    )
    parser.add_argument(
        "--output-dir", default="data/results/tail_profit_diagnostics_v1"
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--forecast-only", action="store_true")
    parser.add_argument(
        "--systems", default="transformer,xgboost_local",
        help=(
            "评价系统，可选transformer,xgboost_local,xgboost_pooled,"
            "timesfm3,timesfm25_lora；外部基线只有2023 inner，"
            "不伪造2024结果"
        ),
    )
    args = parser.parse_args()
    systems = tuple(value.strip() for value in args.systems.split(",") if value.strip())
    unknown = set(systems) - {
        "transformer", "xgboost_local", "xgboost_pooled",
        "timesfm3", "timesfm25_lora",
    }
    if unknown:
        raise ValueError(f"未知系统: {sorted(unknown)}")
    cfg = PilotConfig.from_yaml(args.config)
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)
    started = perf_counter()

    forecast = {}
    thresholds_by_market_phase: dict[tuple[str, str], TailThresholds] = {}
    for market in MARKETS:
        triplet = _simple_triplet(market, "inner")
        thresholds = _thresholds(market, triplet["da"]["nodes"], "inner")
        thresholds_by_market_phase[(market, "inner")] = thresholds
        forecast[market] = {
            "thresholds": thresholds.to_dict(),
            "transformer_2023": _forecast_report(triplet, thresholds),
        }
        if "xgboost_pooled" in systems:
            pooled = _pooled_xgboost_triplet(market)
            forecast[market]["xgboost_pooled_2023"] = _forecast_report(
                pooled, thresholds
            )
        for system in ("timesfm3", "timesfm25_lora"):
            if system in systems:
                forecast[market][f"{system}_2023"] = _forecast_report(
                    _tsfm_triplet(market, system), thresholds
                )
    outer_simple = _simple_triplet("ERCOT", "outer")
    outer_thresholds = _thresholds("ERCOT", outer_simple["da"]["nodes"], "outer")
    thresholds_by_market_phase[("ERCOT", "outer")] = outer_thresholds
    forecast["ERCOT"]["outer_thresholds"] = outer_thresholds.to_dict()
    forecast["ERCOT"]["transformer_2024"] = _forecast_report(
        outer_simple, outer_thresholds
    )

    profit_dirs = {
        "inner": ROOT / "data" / "results" / "universal_lomo_ercot_profit_predictions_v1",
        "outer": ROOT / "data" / "results" / "universal_lomo_ercot_outer_frozen_profit_predictions_v1",
    }
    triplets: dict[tuple[str, str], dict] = {}
    if not args.forecast_only:
        settlement_references = {}
        for phase, directory in profit_dirs.items():
            transformer = _profit_triplet(directory, "universal", phase)
            settlement_references[phase] = transformer
            if "transformer" in systems:
                triplets[("transformer", phase)] = transformer
            if "xgboost_local" in systems:
                reference = {("universal", task): transformer[task] for task in TASKS}
                baseline = _node_baseline_archives(
                    reference, transformer["da"]["nodes"], "xgboost_robust",
                    "universal", phase,
                )
                triplets[("xgboost_local", phase)] = {
                    task: baseline[("xgboost_robust", task)] for task in TASKS
                }
        if "xgboost_pooled" in systems:
            # 当前公平基线固定使用2020—2022三个源市场训练、2023留出市场评价。
            # 没有同协议outer模型时绝不把inner预测冒充2024结果。
            triplets[("xgboost_pooled", "inner")] = _attach_settlement_contract(
                _pooled_xgboost_triplet("ERCOT"),
                settlement_references["inner"],
            )
        for system in ("timesfm3", "timesfm25_lora"):
            if system in systems:
                triplets[(system, "inner")] = _attach_settlement_contract(
                    _tsfm_triplet("ERCOT", system),
                    settlement_references["inner"],
                )

    jobs = []
    for (system, phase), triplet in triplets.items():
        for node, node_input in _node_inputs(triplet).items():
            jobs.append((system, phase, node, _process_safe(node_input)))
    completed = {system: {} for system in systems}
    for system, phase in triplets:
        completed.setdefault(system, {}).setdefault(phase, {})
    config_path = str((ROOT / args.config).resolve())
    if args.workers == 1:
        iterator = (
            _profit_worker(config_path, system, phase, node, node_input)
            for system, phase, node, node_input in jobs
        )
        for system, phase, node, result in iterator:
            completed[system][phase][node] = result
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _profit_worker, config_path, system, phase, node, node_input
                )
                for system, phase, node, node_input in jobs
            ]
            for count, future in enumerate(as_completed(futures), start=1):
                system, phase, node, result = future.result()
                completed[system][phase][node] = result
                print(json.dumps({
                    "event": "tail_profit_node_complete", "completed": count,
                    "total": len(futures), "system": system, "phase": phase,
                    "node": node,
                }, ensure_ascii=False), flush=True)

    profit = {}
    for system, phases in completed.items():
        profit[system] = {}
        for phase, node_reports in phases.items():
            if node_reports:
                profit[system][phase] = _aggregate_profit(
                    node_reports, thresholds_by_market_phase[("ERCOT", phase)]
                )
    report = {
        "experiment": "tail_price_spread_and_profit_instability_v1",
        "status": "2023_inner_and_2024_frozen_outer_diagnostic_not_final_test",
        "test_2025_read": False,
        "threshold_contract": (
            "market-specific percentiles frozen from the target market pre-evaluation "
            "history; diagnostic thresholds only, never model inputs"
        ),
        "forecast_slices": forecast,
        "ercot_profit": profit,
        "runtime_seconds": perf_counter() - started,
    }
    _write_outputs(report, ROOT / args.output_dir)
    print(json.dumps({
        "event": "tail_profit_diagnostics_complete",
        "seconds": report["runtime_seconds"],
        "output_dir": args.output_dir,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
