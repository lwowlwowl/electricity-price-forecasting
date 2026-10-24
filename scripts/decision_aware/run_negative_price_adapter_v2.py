#!/usr/bin/env python
"""运行保守季节性NEG-E与条件幅度NEG-A，并做完整双结算。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.negative_price_adapter import (  # noqa: E402
    FEATURE_NAMES,
    SEASONAL_EVENT_FEATURE_INDICES,
    SEASONAL_EVENT_FEATURE_NAMES,
    apply_negative_price_adapter,
    event_metrics,
    fit_conservative_seasonal_event_model,
    fit_unweighted_amplitude_model,
    select_source_fpr_threshold,
)
from scripts.decision_aware.audit_tail_event_amplitude_oracles import _paired  # noqa: E402
from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    TASKS,
    _aggregate_profit,
    _node_inputs,
    _profit_worker,
)
from scripts.decision_aware.evaluate_global_continuous_mpc import _process_safe  # noqa: E402
from scripts.decision_aware.evaluate_global_f0_f1_triplets import _load_archive  # noqa: E402
from scripts.decision_aware.run_negative_price_adapter_v1 import (  # noqa: E402
    SOURCE_MARKETS,
    _all_source_validation,
    _ercot_feature_cube,
    _forecast_report,
    _load_or_build,
    _load_thresholds,
    _nodes,
    _sample_event_training,
)


SYSTEMS = ("original", "neg_e_v2", "neg_e_neg_a_v2")
STAGES = ("inner", "outer")


def _coefficients(model, step: str, names) -> list[dict]:
    values = np.asarray(model.named_steps[step].coef_).reshape(-1)
    order = np.argsort(-np.abs(values))
    return [
        {"feature": str(names[index]), "standardized_coefficient": float(values[index])}
        for index in order
    ]


def _write_prediction(
    output_dir: Path,
    phase: str,
    system: str,
    original: dict,
    prediction: np.ndarray,
    triggered: np.ndarray,
    threshold: float,
) -> None:
    path = output_dir / "predictions" / f"{system}_da_{phase}_seed0.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "negative_price_adapter_v2",
        "system": system,
        "phase": phase,
        "training_markets": list(SOURCE_MARKETS),
        "training_years": "2020-2022",
        "threshold_selection": "maximum_source_market_2023_nonnegative_q99",
        "event_threshold": threshold,
        "uses_ercot_labels_for_training_or_selection": False,
        "selection_allowed_on_ercot": False,
        "test_2025_read": False,
    }
    np.savez_compressed(
        path,
        nodes=original["nodes"], dates=original["dates"],
        prediction=np.asarray(prediction, dtype=np.float32),
        target_da=original["target_da"], target_rt=original["target_rt"],
        triggered_mask=np.asarray(triggered, dtype=bool),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# 保守负价事件与条件幅度适配器结果", "",
        "> 模型只在CAISO、NYISO、PJM的2020—2022训练，阈值只由三个源市场2023冻结；ERCOT不参与拟合或选模；未读取2025。",
        "", f"冻结阈值：**{report['event_threshold']:.6f}**", "",
        "## 源市场2023门控审计", "",
        "| 市场 | 精确率 | 召回率 | F1 | 假阳性率 | 市场阈值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for market in SOURCE_MARKETS:
        item = report["source_validation"][market]
        lines.append(
            f"| {market} | {item['precision']:.3f} | {item['recall']:.3f} | "
            f"{item['f1']:.3f} | {item['false_positive_rate']:.3%} | "
            f"{report['source_market_thresholds'][market]:.6f} |"
        )
    for phase, year in (("inner", 2023), ("outer", 2024)):
        lines.extend(("", f"## ERCOT {year}", ""))
        lines.extend((
            "| 系统 | 负价精确率 | 负价召回 | F1 | 负价MAE | 普通时段MAE | 总体MAE | 日均净收益 | 相对原系统 | 95%CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ))
        block = report["ercot"][phase]
        for system in SYSTEMS:
            forecast = block["forecast"][system]
            event = forecast["negative_event"]["exact"]
            profit = block["profit"][system]
            if system == "original":
                delta, ci = 0.0, "-"
            else:
                paired = block["paired_vs_original"][system]
                delta = paired["mean_daily_net_revenue_difference"]
                interval = paired["seven_day_block_bootstrap_95ci"]
                ci = f"[{interval[0]:.2f}, {interval[1]:.2f}]"
            lines.append(
                f"| {system} | {event['precision']:.3f} | {event['recall']:.3f} | "
                f"{event['f1']:.3f} | {forecast['negative_mae']:.2f} | "
                f"{forecast['ordinary_nonnegative_mae']:.2f} | "
                f"{forecast['all']['all']['mae']:.2f} | "
                f"{profit['macro_mean_daily_net_revenue']:.2f} | {delta:+.2f} | {ci} |"
            )
        lines.extend(("", "### 收益分解（USD/节点·日）", ""))
        lines.extend((
            "| 系统 | DA计划收益 | RT修正收益 | 退化成本 | 偏差罚金 |",
            "|---|---:|---:|---:|---:|",
        ))
        for system in SYSTEMS:
            component = block["profit"][system]["macro_mean_daily_components"]
            lines.append(
                f"| {system} | {component['da_leg']:.2f} | "
                f"{component['rt_deviation_leg']:.2f} | "
                f"{component['degradation_cost']:.2f} | "
                f"{component['deviation_penalty']:.2f} |"
            )
    lines.extend(("", "## 主要标准化系数", "", "### NEG-E", ""))
    lines.extend(
        f"- `{row['feature']}`: {row['standardized_coefficient']:+.4f}"
        for row in report["event_coefficients"]
    )
    lines.extend(("", "### NEG-A（绝对值前12项）", ""))
    lines.extend(
        f"- `{row['feature']}`: {row['standardized_coefficient']:+.4f}"
        for row in report["amplitude_coefficients"][:12]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default="data/results/negative_price_adapter_v1/feature_cache",
    )
    parser.add_argument(
        "--inner-prediction-dir",
        default="data/results/universal_lomo_ercot_profit_predictions_v1",
    )
    parser.add_argument(
        "--outer-prediction-dir",
        default="data/results/universal_lomo_ercot_outer_frozen_profit_predictions_v1",
    )
    parser.add_argument(
        "--settlement-config", default="configs/decision_aware/global_ercot_huber_v1.yaml"
    )
    parser.add_argument(
        "--output-dir", default="data/results/negative_price_adapter_v2"
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    started = perf_counter()
    base = PilotConfig.from_yaml(str(ROOT / args.base_config))
    nodes = _nodes()
    cache_dir = ROOT / args.feature_cache_dir
    source_train, source_val = {}, {}
    for market in SOURCE_MARKETS:
        source_train[market] = _load_or_build(
            base, market, nodes[market], "train", cache_dir
        )
        source_val[market] = _load_or_build(
            base, market, nodes[market], "val", cache_dir
        )

    event_x, event_y, _, sampling = _sample_event_training(source_train)
    event_model = fit_conservative_seasonal_event_model(event_x, event_y)
    negative_x = np.concatenate([
        source_train[m]["x"][source_train[m]["target"] < 0.0]
        for m in SOURCE_MARKETS
    ])
    negative_y = np.concatenate([
        source_train[m]["target"][source_train[m]["target"] < 0.0]
        for m in SOURCE_MARKETS
    ])
    amplitude_model = fit_unweighted_amplitude_model(negative_x, negative_y)

    val_x, val_y, val_markets = _all_source_validation(source_val)
    val_probability = event_model.predict_proba(
        val_x[:, SEASONAL_EVENT_FEATURE_INDICES]
    )[:, 1]
    threshold, market_thresholds = select_source_fpr_threshold(
        val_probability, val_y, val_markets
    )
    source_validation = {
        market: event_metrics(
            val_probability[val_markets == market], val_y[val_markets == market],
            threshold,
        )
        for market in SOURCE_MARKETS
    }
    print(json.dumps({
        "event": "negative_adapter_v2_frozen", "threshold": threshold,
        "source_validation": source_validation,
    }, ensure_ascii=False), flush=True)

    output_dir = ROOT / args.output_dir
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(event_model, models_dir / "neg_e_v2.joblib")
    joblib.dump(amplitude_model, models_dir / "neg_a_v2.joblib")

    thresholds = _load_thresholds()
    phase_dirs = {
        "inner": ROOT / args.inner_prediction_dir,
        "outer": ROOT / args.outer_prediction_dir,
    }
    ercot = {}
    for phase in STAGES:
        original = {
            task: _load_archive(phase_dirs[phase], "universal", task, phase)
            for task in TASKS
        }
        da = original["da"]
        features = _ercot_feature_cube(
            base, list(da["nodes"].astype(str)), phase, da, cache_dir
        )
        flat = features.reshape(-1, len(FEATURE_NAMES))
        probability = event_model.predict_proba(
            flat[:, SEASONAL_EVENT_FEATURE_INDICES]
        )[:, 1].reshape(da["prediction"].shape)
        amplitude = amplitude_model.predict(flat).reshape(da["prediction"].shape)
        neg_e, neg_full, triggered = apply_negative_price_adapter(
            da["prediction"], probability, amplitude, threshold=threshold
        )
        predictions = {
            "original": da["prediction"], "neg_e_v2": neg_e,
            "neg_e_neg_a_v2": neg_full,
        }
        for system in SYSTEMS[1:]:
            _write_prediction(
                output_dir, phase, system, da, predictions[system], triggered,
                threshold,
            )
        forecast = {
            system: _forecast_report(pred, da["target_da"], thresholds)
            for system, pred in predictions.items()
        }

        jobs = []
        for system, prediction in predictions.items():
            triplet = {
                "da": {**da, "prediction": prediction},
                "rt_at_da": original["rt_at_da"], "rt": original["rt"],
            }
            for node, node_input in _node_inputs(triplet).items():
                jobs.append((system, node, _process_safe(node_input)))
        completed = {system: {} for system in SYSTEMS}
        config_path = str((ROOT / args.settlement_config).resolve())
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _profit_worker, config_path, system, phase, node, node_input
                )
                for system, node, node_input in jobs
            ]
            for count, future in enumerate(as_completed(futures), start=1):
                system, _, node, result = future.result()
                completed[system][node] = result
                print(json.dumps({
                    "event": "negative_adapter_v2_settlement",
                    "phase": phase, "completed": count, "total": len(futures),
                    "system": system, "node": node,
                }, ensure_ascii=False), flush=True)
        profit = {
            system: _aggregate_profit(completed[system], thresholds)
            for system in SYSTEMS
        }
        paired = {
            system: _paired(profit[system], profit["original"])
            for system in SYSTEMS[1:]
        }
        ercot[phase] = {
            "forecast": forecast, "profit": profit,
            "paired_vs_original": paired,
            "triggered_hours": int(triggered.sum()),
            "triggered_share": float(triggered.mean()),
            "event_probability_metrics": event_metrics(
                probability, da["target_da"], threshold
            ),
        }

    report = {
        "experiment": "negative_price_adapter_v2",
        "status": "complete_source_frozen_ercot_mechanism_and_profit_evaluation",
        "training_markets": list(SOURCE_MARKETS),
        "training_years": "2020-2022",
        "threshold_selection_data": "CAISO/NYISO/PJM 2023 only",
        "uses_ercot_labels_for_training_or_selection": False,
        "selection_allowed_on_ercot": False,
        "test_2025_read": False,
        "event_threshold": threshold,
        "source_market_thresholds": market_thresholds,
        "source_validation": source_validation,
        "event_training_sampling": sampling,
        "event_feature_names": list(SEASONAL_EVENT_FEATURE_NAMES),
        "event_coefficients": _coefficients(
            event_model, "event", SEASONAL_EVENT_FEATURE_NAMES
        ),
        "amplitude_coefficients": _coefficients(
            amplitude_model, "amplitude", FEATURE_NAMES
        ),
        "ercot": ercot,
        "runtime_seconds": perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(report, output_dir / "summary.md")
    print(json.dumps({
        "event": "negative_price_adapter_v2_complete",
        "output_dir": output_dir.relative_to(ROOT).as_posix(),
        "runtime_seconds": report["runtime_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

