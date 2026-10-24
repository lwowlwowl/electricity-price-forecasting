#!/usr/bin/env python
"""审计2024 ERCOT DA尾部事件识别与幅度上界。

本脚本不训练模型，也不读取2025。它只在已冻结的2024 DA预测中替换
真实尾部小时，并保持RT-at-DA、滚动RT、LP/MPC和结算合同不变。
因此结果是机制诊断的oracle上界，不是可部署模型的成绩。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.tail_diagnostics import (  # noqa: E402
    TailThresholds,
    daily_peak_report,
    event_detection_report,
    price_slice_report,
)
from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    TASKS,
    _aggregate_profit,
    _node_inputs,
    _profit_worker,
)
from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _block_bootstrap_ci,
    _process_safe,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
)


SYSTEMS = (
    "original",
    "positive_event_exact",
    "positive_event_ranked",
    "positive_amplitude_exact",
    "negative_event_exact",
    "negative_amplitude_exact",
)


def _oracle_prediction(
    prediction,
    target,
    *,
    system: str,
    positive_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """构造单一oracle并返回(预测,被替换的小时掩码)。

    `positive_event_ranked`只使用真值判断“该节点-日是否有尖峰”，
    小时位置仍由原模型的当日最大预测决定。
    """
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    if pred.shape != truth.shape:
        raise ValueError("prediction与target必须同形")
    if pred.ndim < 2 or pred.shape[-1] != 24:
        raise ValueError("DA oracle需要最后一维为24小时")
    if system not in SYSTEMS:
        raise ValueError(f"未知oracle系统: {system}")

    result = pred.copy()
    changed = np.zeros(pred.shape, dtype=bool)
    positive = np.isfinite(truth) & (truth >= positive_threshold)
    negative = np.isfinite(truth) & (truth < 0.0)

    if system == "original":
        return result, changed
    if system == "positive_event_exact":
        result[positive] = np.maximum(result[positive], positive_threshold + 1.0)
        changed = positive & (result != pred)
    elif system == "positive_amplitude_exact":
        result[positive] = truth[positive]
        changed = positive & (result != pred)
    elif system == "negative_event_exact":
        result[negative] = np.minimum(result[negative], -1.0)
        changed = negative & (result != pred)
    elif system == "negative_amplitude_exact":
        result[negative] = truth[negative]
        changed = negative & (result != pred)
    elif system == "positive_event_ranked":
        # 前面所有维度都是独立日曲线，最后一维才是小时。
        flat_pred = result.reshape(-1, result.shape[-1])
        flat_truth = truth.reshape(-1, truth.shape[-1])
        flat_changed = changed.reshape(-1, changed.shape[-1])
        spike_days = np.any(flat_truth >= positive_threshold, axis=-1)
        rows = np.flatnonzero(spike_days)
        hours = np.argmax(flat_pred[rows], axis=-1)
        old = flat_pred[rows, hours].copy()
        flat_pred[rows, hours] = np.maximum(old, positive_threshold + 1.0)
        flat_changed[rows, hours] = flat_pred[rows, hours] != old
    return result, changed


def _forecast_report(
    prediction,
    target,
    changed,
    *,
    thresholds: TailThresholds,
) -> dict:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    mask = np.asarray(changed, dtype=bool)
    return {
        "intervention_hours": int(mask.sum()),
        "intervention_share": float(mask.mean()),
        "unchanged_outside_intervention": True,
        "price_slices": price_slice_report(
            pred,
            truth,
            lower_1pct=thresholds.da_lower_1pct,
            upper_1pct=thresholds.da_upper_1pct,
        ),
        "positive_spike_event": event_detection_report(
            pred,
            truth,
            threshold=thresholds.da_upper_1pct,
            direction="above",
            tolerance_hours=1,
        ),
        "negative_price_event": event_detection_report(
            pred,
            truth,
            threshold=0.0,
            direction="below",
            tolerance_hours=1,
        ),
        "positive_spike_daily_peak": daily_peak_report(
            pred,
            truth,
            positive_threshold=thresholds.da_upper_1pct,
        ),
    }


def _paired(candidate: dict, baseline: dict) -> dict:
    left = candidate["macro_daily_net_revenue"]
    right = baseline["macro_daily_net_revenue"]
    days = sorted(right)
    if days != sorted(left):
        raise ValueError("oracle与原系统的结算日不一致")
    difference = np.asarray([left[day] - right[day] for day in days])
    component_difference = {
        key: float(
            candidate["macro_mean_daily_components"][key]
            - baseline["macro_mean_daily_components"][key]
        )
        for key in baseline["macro_mean_daily_components"]
    }
    return {
        "mean_daily_net_revenue_difference": float(np.mean(difference)),
        "median_daily_net_revenue_difference": float(np.median(difference)),
        "positive_difference_day_rate": float(np.mean(difference > 0.0)),
        "seven_day_block_bootstrap_95ci": _block_bootstrap_ci(
            difference, seed=20261019, draws=5000
        ),
        "macro_mean_daily_component_difference": component_difference,
    }


def _write_prediction_archive(
    output_dir: Path,
    system: str,
    da_archive: dict,
    prediction: np.ndarray,
    changed: np.ndarray,
    threshold: float,
) -> None:
    metadata = {
        "experiment": "ercot_2024_tail_event_amplitude_oracle_v1",
        "system": system,
        "phase": "outer",
        "diagnostic_oracle_not_deployable": system != "original",
        "selection_allowed": False,
        "test_2025_read": False,
        "positive_threshold_frozen_from_2020_2022_usd_mwh": threshold,
        "only_da_prediction_changed": True,
        "rt_at_da_and_rolling_rt_unchanged": True,
    }
    path = output_dir / "predictions" / f"{system}_da_outer_seed0.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        nodes=da_archive["nodes"],
        dates=da_archive["dates"],
        prediction=prediction.astype(np.float32),
        target_da=da_archive["target_da"],
        target_rt=da_archive["target_rt"],
        changed_mask=changed,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def _write_markdown(report: dict, output: Path) -> None:
    lines = [
        "# 2024 ERCOT尾部事件与幅度Oracle审计",
        "",
        "> 这是使用冻结2024真值的机制诊断上界，不是可部署模型，也不用于选择2025方案。",
        "",
        "## 预测尾部",
        "",
        "| 系统 | 改动小时 | 正尖峰召回 | 正尖峰幅度保留 | 负价召回 | 负价幅度保留 | 全时段MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        item = report["forecast"][system]
        pos = item["positive_spike_event"]
        neg = item["negative_price_event"]
        mae = item["price_slices"]["all"]["mae"]
        lines.append(
            f"| {system} | {item['intervention_hours']} | "
            f"{100*pos['exact']['recall']:.1f}% | "
            f"{100*pos['magnitude_ratio_at_true_events']:.1f}% | "
            f"{100*neg['exact']['recall']:.1f}% | "
            f"{100*neg['magnitude_ratio_at_true_events']:.1f}% | {mae:.2f} |"
        )
    if report.get("profit"):
        lines.extend((
            "", "## 完整LP/MPC双结算", "",
            "| 系统 | 日均净收益 | 相对原系统 | 7日块Bootstrap 95%CI | DA腿 | RT修正腿 | 退化成本 | 偏差罚金 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ))
        original = report["profit"]["original"]
        for system in SYSTEMS:
            item = report["profit"][system]
            comp = item["macro_mean_daily_components"]
            if system == "original":
                delta = 0.0
                ci = "-"
            else:
                paired = report["paired_vs_original"][system]
                delta = paired["mean_daily_net_revenue_difference"]
                interval = paired["seven_day_block_bootstrap_95ci"]
                ci = f"[{interval[0]:.2f}, {interval[1]:.2f}]"
            lines.append(
                f"| {system} | {item['macro_mean_daily_net_revenue']:.2f} | "
                f"{delta:+.2f} | {ci} | {comp['da_leg']:.2f} | "
                f"{comp['rt_deviation_leg']:.2f} | {comp['degradation_cost']:.2f} | "
                f"{comp['deviation_penalty']:.2f} |"
            )
        lines.extend((
            "", "## 解释规则", "",
            "- 事件Oracle的增益表示“只把发生与否判对”的经济价值。",
            "- 幅度Oracle减事件Oracle表示“在事件已知后继续修复幅度”的追加价值。",
            "- 所有Oracle都使用真值，只能用来定位下一个可训练机制，不能写成模型成绩。",
        ))
        lines.extend((
            "", "## 嵌套Oracle差值", "",
            "| 差值 | 日均净收益增量 | 7日块Bootstrap 95%CI |",
            "|---|---:|---:|",
        ))
        for label, item in report["nested_oracle_differences"].items():
            interval = item["seven_day_block_bootstrap_95ci"]
            lines.append(
                f"| {label} | {item['mean_daily_net_revenue_difference']:+.2f} | "
                f"[{interval[0]:.2f}, {interval[1]:.2f}] |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/global_ercot_huber_v1.yaml"
    )
    parser.add_argument(
        "--prediction-dir",
        default="data/results/universal_lomo_ercot_outer_frozen_profit_predictions_v1",
    )
    parser.add_argument(
        "--threshold-source",
        default=(
            "data/results/"
            "universal_lomo_ercot_tailprice_da_outer_confirmation_summary_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/tail_event_amplitude_oracle_2024_v1",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--forecast-only", action="store_true")
    parser.add_argument(
        "--reuse-profit",
        action="store_true",
        help="复用同一输出目录已完成的结算，只重建汇总与嵌套差值",
    )
    parser.add_argument(
        "--recorded-settlement-runtime-seconds",
        type=float,
        help="仅在--reuse-profit时用于保留首次完整结算的实测耗时",
    )
    args = parser.parse_args()

    started = perf_counter()
    source = json.loads((ROOT / args.threshold_source).read_text(encoding="utf-8"))
    frozen = source["thresholds_frozen_from_2020_2022"]
    thresholds = TailThresholds(**{key: float(value) for key, value in frozen.items()})
    directory = ROOT / args.prediction_dir
    original = {
        task: _load_archive(directory, "universal", task, "outer")
        for task in TASKS
    }
    da = original["da"]
    target = da["target_da"]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    triplets = {}
    forecast = {}
    for system in SYSTEMS:
        prediction, changed = _oracle_prediction(
            da["prediction"],
            target,
            system=system,
            positive_threshold=thresholds.da_upper_1pct,
        )
        if not np.array_equal(
            prediction[~changed], da["prediction"][~changed], equal_nan=True
        ):
            raise AssertionError(f"{system}改动了介入掩码之外的预测")
        forecast[system] = _forecast_report(
            prediction, target, changed, thresholds=thresholds
        )
        if not args.reuse_profit:
            _write_prediction_archive(
                output_dir,
                system,
                da,
                prediction,
                changed,
                thresholds.da_upper_1pct,
            )
        triplets[system] = {
            "da": {**da, "prediction": prediction},
            "rt_at_da": original["rt_at_da"],
            "rt": original["rt"],
        }

    profit = {}
    settlement_runtime_seconds = None
    if args.reuse_profit:
        previous_path = output_dir / "summary.json"
        if not previous_path.is_file():
            raise FileNotFoundError(f"没有可复用的结算汇总: {previous_path}")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        profit = previous.get("profit", {})
        if set(profit) != set(SYSTEMS):
            raise ValueError("已有汇总不包含全部Oracle系统")
        settlement_runtime_seconds = (
            args.recorded_settlement_runtime_seconds
            if args.recorded_settlement_runtime_seconds is not None
            else previous.get("settlement_runtime_seconds")
        )
    elif not args.forecast_only:
        settlement_started = perf_counter()
        jobs = []
        for system, triplet in triplets.items():
            for node, node_input in _node_inputs(triplet).items():
                jobs.append((system, node, _process_safe(node_input)))
        completed = {system: {} for system in SYSTEMS}
        config_path = str((ROOT / args.config).resolve())
        if args.workers == 1:
            iterator = (
                _profit_worker(config_path, system, "outer", node, node_input)
                for system, node, node_input in jobs
            )
            for count, (system, _, node, result) in enumerate(iterator, start=1):
                completed[system][node] = result
                print(json.dumps({
                    "event": "tail_oracle_node_complete",
                    "completed": count,
                    "total": len(jobs),
                    "system": system,
                    "node": node,
                }, ensure_ascii=False), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(
                        _profit_worker,
                        config_path,
                        system,
                        "outer",
                        node,
                        node_input,
                    )
                    for system, node, node_input in jobs
                ]
                for count, future in enumerate(as_completed(futures), start=1):
                    system, _, node, result = future.result()
                    completed[system][node] = result
                    print(json.dumps({
                        "event": "tail_oracle_node_complete",
                        "completed": count,
                        "total": len(futures),
                        "system": system,
                        "node": node,
                    }, ensure_ascii=False), flush=True)
        profit = {
            system: _aggregate_profit(completed[system], thresholds)
            for system in SYSTEMS
        }
        settlement_runtime_seconds = perf_counter() - settlement_started

    paired = (
        {
            system: _paired(profit[system], profit["original"])
            for system in SYSTEMS if system != "original"
        }
        if profit else {}
    )
    nested = (
        {
            "positive_event_exact_minus_original": _paired(
                profit["positive_event_exact"], profit["original"]
            ),
            "positive_amplitude_exact_minus_positive_event_exact": _paired(
                profit["positive_amplitude_exact"], profit["positive_event_exact"]
            ),
            "positive_amplitude_exact_minus_original": _paired(
                profit["positive_amplitude_exact"], profit["original"]
            ),
            "negative_event_exact_minus_original": _paired(
                profit["negative_event_exact"], profit["original"]
            ),
            "negative_amplitude_exact_minus_negative_event_exact": _paired(
                profit["negative_amplitude_exact"], profit["negative_event_exact"]
            ),
            "negative_amplitude_exact_minus_original": _paired(
                profit["negative_amplitude_exact"], profit["original"]
            ),
        }
        if profit else {}
    )
    report = {
        "experiment": "ercot_2024_tail_event_amplitude_oracle_v1",
        "status": "2024_frozen_outer_mechanism_diagnostic_not_for_selection",
        "selection_allowed": False,
        "deployable_model": False,
        "test_2025_read": False,
        "scope": "DA only; RT-at-DA, rolling RT, LP/MPC and settlement unchanged",
        "thresholds_frozen_from_2020_2022": thresholds.to_dict(),
        "systems": list(SYSTEMS),
        "forecast": forecast,
        "profit": profit,
        "paired_vs_original": paired,
        "nested_oracle_differences": nested,
        "runtime_seconds": perf_counter() - started,
        "settlement_runtime_seconds": settlement_runtime_seconds,
        "workers": args.workers,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(report, output_dir / "summary.md")
    print(json.dumps({
        "event": "tail_event_amplitude_oracle_complete",
        "output_dir": output_dir.relative_to(ROOT).as_posix(),
        "runtime_seconds": report["runtime_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
