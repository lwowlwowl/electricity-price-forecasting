#!/usr/bin/env python
"""训练并评价可解释的NEG-E/NEG-A负价适配器。

训练和阈值选择只使用CAISO/NYISO/PJM的2020--2023数据；ERCOT仅在
适配器冻结后用于2023/2024机制检查。本脚本硬拒绝2025。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.dataset_da import DecisionAwareDADataset  # noqa: E402
from decision_aware.loader_v2 import (  # noqa: E402
    load_market_unified_nodes,
    market_timezone,
)
from decision_aware.negative_price_adapter import (  # noqa: E402
    FEATURE_NAMES,
    apply_negative_price_adapter,
    event_metrics,
    extract_da_adapter_features,
    fit_amplitude_model,
    fit_event_model,
    select_market_macro_threshold,
)
from decision_aware.tail_diagnostics import (  # noqa: E402
    TailThresholds,
    event_detection_report,
    price_slice_report,
)
from scripts.decision_aware.audit_tail_event_amplitude_oracles import (  # noqa: E402
    _paired,
)
from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    TASKS,
    _aggregate_profit,
    _node_inputs,
    _profit_worker,
)
from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _process_safe,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
)


SOURCE_MARKETS = ("CAISO", "NYISO", "PJM")
SYSTEMS = ("original", "neg_e", "neg_e_neg_a")
STAGES = ("inner", "outer")
PHASE_TO_ANNUAL_STAGE = {
    "train": AnnualStage.INNER_TRAIN,
    "val": AnnualStage.INNER_VALIDATION,
    "outer": AnnualStage.OUTER_VALIDATION,
}


def _nodes() -> dict[str, list[str]]:
    raw = yaml.safe_load((ROOT / "configs/nodes.yaml").read_text(encoding="utf-8"))
    return {market: [str(node) for node in raw[market]["all"]] for market in raw}


def _dataset_config(base: PilotConfig, market: str, node: str) -> PilotConfig:
    split = FORMAL_EXPERIMENT_CONTRACT.annual_split
    return replace(
        base,
        market=market,
        node=node,
        da_issue_hour_local=base.resolved_da_issue_hour_local(market),
        split_train_start=split.inner_train.start,
        split_train_end=f"{split.inner_train.end} 23:59:59",
        split_val_start=split.inner_validation.start,
        split_val_end=f"{split.inner_validation.end} 23:59:59",
        split_test_start=split.outer_validation.start,
        split_test_end=f"{split.outer_validation.end} 23:59:59",
    )


def _build_market_bundle(
    base: PilotConfig,
    market: str,
    nodes: list[str],
    split: str,
) -> dict:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"未知split: {split}")
    # 明确截到2024末；本实验不会让加载器看到2025标签。
    local_2025_start_utc = pd.Timestamp(
        "2025-01-01", tz=market_timezone(market)
    ).tz_convert("UTC").isoformat()
    tables = load_market_unified_nodes(
        market,
        nodes,
        start="2020-01-01",
        end=local_2025_start_utc,
        read_end_utc=local_2025_start_utc,
    )
    xs, ys, node_ids, dates, hours = [], [], [], [], []
    for node_index, node in enumerate(nodes):
        cfg = _dataset_config(base, market, node)
        dataset = DecisionAwareDADataset(tables[node], cfg, split)
        bundle = extract_da_adapter_features(dataset)
        xs.append(bundle.x)
        ys.append(bundle.target)
        node_ids.append(np.full(len(bundle.target), node_index, dtype=np.int16))
        dates.append(bundle.delivery_dates)
        hours.append(bundle.hours)
    return {
        "x": np.concatenate(xs),
        "target": np.concatenate(ys),
        "node_id": np.concatenate(node_ids),
        "dates": np.concatenate(dates),
        "hours": np.concatenate(hours),
        "nodes": np.asarray(nodes),
        "market": market,
        "split": split,
    }


def _cache_path(cache_dir: Path, market: str, split: str) -> Path:
    return cache_dir / f"{market.lower()}_{split}_features_v2.npz"


def _load_or_build(
    base: PilotConfig,
    market: str,
    nodes: list[str],
    split: str,
    cache_dir: Path,
) -> dict:
    path = _cache_path(cache_dir, market, split)
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            result = {key: archive[key] for key in archive.files}
        result["market"] = str(result["market"].item())
        result["split"] = str(result["split"].item())
        return result
    result = _build_market_bundle(base, market, nodes, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result


def _sample_event_training(
    bundles: dict[str, dict], *, seed: int = 20261019
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    selected = []
    audit = {}
    for market in SOURCE_MARKETS:
        target = bundles[market]["target"]
        negative = np.flatnonzero(target < 0.0)
        nonnegative = np.flatnonzero(target >= 0.0)
        keep_nonnegative = min(len(nonnegative), 10 * len(negative), 100_000)
        if keep_nonnegative == 0:
            raise ValueError(f"{market}没有可用负价样本")
        chosen_nonnegative = rng.choice(
            nonnegative, size=keep_nonnegative, replace=False
        )
        chosen = np.sort(np.concatenate((negative, chosen_nonnegative)))
        selected.append((market, chosen))
        audit[market] = {
            "all_hours": int(len(target)),
            "negative_hours": int(len(negative)),
            "sampled_nonnegative_hours": int(keep_nonnegative),
        }
    x = np.concatenate([bundles[m]["x"][idx] for m, idx in selected])
    y = np.concatenate([bundles[m]["target"][idx] for m, idx in selected])
    markets = np.concatenate([
        np.full(len(idx), market) for market, idx in selected
    ])
    return x, y, markets, audit


def _all_source_validation(bundles: dict[str, dict]):
    return (
        np.concatenate([bundles[m]["x"] for m in SOURCE_MARKETS]),
        np.concatenate([bundles[m]["target"] for m in SOURCE_MARKETS]),
        np.concatenate([
            np.full(len(bundles[m]["target"]), m) for m in SOURCE_MARKETS
        ]),
    )


def _coefficient_report(model, step: str) -> list[dict]:
    coefficients = np.asarray(model.named_steps[step].coef_).reshape(-1)
    order = np.argsort(-np.abs(coefficients))
    return [
        {"feature": FEATURE_NAMES[index], "standardized_coefficient": float(coefficients[index])}
        for index in order
    ]


def _ercot_feature_cube(
    base: PilotConfig,
    nodes: list[str],
    phase: str,
    archive: dict,
    cache_dir: Path,
) -> np.ndarray:
    split = "val" if phase == "inner" else "test"
    bundle = _load_or_build(base, "ERCOT", nodes, split, cache_dir)
    cubes = []
    for node_index, node in enumerate(nodes):
        mask = bundle["node_id"] == node_index
        x = bundle["x"][mask]
        dates = bundle["dates"][mask][::24].astype(str)
        if not np.array_equal(dates, archive["dates"].astype(str)):
            raise ValueError(f"{phase} {node}适配器特征与预测档案日期不一致")
        if x.shape != (len(dates) * 24, len(FEATURE_NAMES)):
            raise ValueError(f"{phase} {node}特征形状异常: {x.shape}")
        cubes.append(x.reshape(len(dates), 24, len(FEATURE_NAMES)))
    return np.stack(cubes)


def _forecast_report(prediction, target, thresholds: TailThresholds) -> dict:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    negative = truth < 0.0
    ordinary = ~negative
    return {
        "all": price_slice_report(
            pred, truth,
            lower_1pct=thresholds.da_lower_1pct,
            upper_1pct=thresholds.da_upper_1pct,
        ),
        "negative_event": event_detection_report(
            pred, truth, threshold=0.0, direction="below", tolerance_hours=1
        ),
        "negative_mae": float(np.mean(np.abs(pred[negative] - truth[negative]))),
        "ordinary_nonnegative_mae": float(np.mean(np.abs(
            pred[ordinary] - truth[ordinary]
        ))),
        "negative_hours": int(negative.sum()),
    }


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
        "experiment": "negative_price_adapter_v1",
        "system": system,
        "phase": phase,
        "training_markets": list(SOURCE_MARKETS),
        "training_years": "2020-2022",
        "threshold_selection": "source_markets_2023_market_macro_f1",
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


def _load_thresholds() -> TailThresholds:
    path = ROOT / "data/results/universal_lomo_ercot_tailprice_da_outer_confirmation_summary_v1.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    return TailThresholds(**{
        key: float(value)
        for key, value in report["thresholds_frozen_from_2020_2022"].items()
    })


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# 负价事件与条件幅度适配器结果", "",
        "> NEG-E/NEG-A只在CAISO、NYISO、PJM训练并用源市场2023选择阈值；ERCOT不参与拟合或选模；未读取2025。",
        "", f"冻结事件阈值：**{report['event_threshold']:.2f}**", "",
        "## 源市场2023负价事件验证", "",
        "| 市场 | 精确率 | 召回率 | F1 | 假阳性率 |", "|---|---:|---:|---:|---:|",
    ]
    selected = min(
        report["source_validation_thresholds"],
        key=lambda row: abs(row["threshold"] - report["event_threshold"]),
    )
    for market, item in selected["by_market"].items():
        lines.append(
            f"| {market} | {item['precision']:.3f} | {item['recall']:.3f} | "
            f"{item['f1']:.3f} | {item['false_positive_rate']:.3f} |"
        )
    for phase in STAGES:
        lines.extend(("", f"## ERCOT {2023 if phase == 'inner' else 2024}", ""))
        lines.extend((
            "| 系统 | 负价召回 | 负价幅度保留 | 负价MAE | 普通时段MAE | 总体MAE | 日均净收益 | 相对原系统 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ))
        for system in SYSTEMS:
            item = report["ercot"][phase]["forecast"][system]
            event = item["negative_event"]
            profit = report["ercot"][phase].get("profit", {}).get(system)
            revenue = profit["macro_mean_daily_net_revenue"] if profit else None
            delta = (
                report["ercot"][phase]["paired_vs_original"][system][
                    "mean_daily_net_revenue_difference"
                ] if system != "original" and profit else 0.0 if profit else None
            )
            lines.append(
                f"| {system} | {event['exact']['recall']:.3f} | "
                f"{event['magnitude_ratio_at_true_events']:.3f} | "
                f"{item['negative_mae']:.2f} | {item['ordinary_nonnegative_mae']:.2f} | "
                f"{item['all']['all']['mae']:.2f} | "
                f"{revenue:.2f} | {delta:+.2f} |" if profit else
                f"| {system} | {event['exact']['recall']:.3f} | "
                f"{event['magnitude_ratio_at_true_events']:.3f} | "
                f"{item['negative_mae']:.2f} | {item['ordinary_nonnegative_mae']:.2f} | "
                f"{item['all']['all']['mae']:.2f} | - | - |"
            )
    lines.extend(("", "## 最重要的可解释系数", "", "### NEG-E", ""))
    lines.extend(
        f"- `{row['feature']}`: {row['standardized_coefficient']:+.4f}"
        for row in report["event_coefficients"][:12]
    )
    lines.extend(("", "### NEG-A", ""))
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
        "--output-dir", default="data/results/negative_price_adapter_v1"
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--forecast-only", action="store_true")
    args = parser.parse_args()

    started = perf_counter()
    output_dir = ROOT / args.output_dir
    cache_dir = output_dir / "feature_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = PilotConfig.from_yaml(str(ROOT / args.base_config))
    all_nodes = _nodes()

    source_train = {}
    source_val = {}
    for market in SOURCE_MARKETS:
        source_train[market] = _load_or_build(
            base, market, all_nodes[market], "train", cache_dir
        )
        source_val[market] = _load_or_build(
            base, market, all_nodes[market], "val", cache_dir
        )
        print(json.dumps({
            "event": "source_features_ready", "market": market,
            "train_hours": len(source_train[market]["target"]),
            "validation_hours": len(source_val[market]["target"]),
        }, ensure_ascii=False), flush=True)

    event_x, event_y, event_markets, sampling = _sample_event_training(source_train)
    # NEG-A需要全部负价；传入完整源训练集，模型内部只保留负价。
    all_train_x = np.concatenate([source_train[m]["x"] for m in SOURCE_MARKETS])
    all_train_y = np.concatenate([source_train[m]["target"] for m in SOURCE_MARKETS])
    all_train_markets = np.concatenate([
        np.full(len(source_train[m]["target"]), m) for m in SOURCE_MARKETS
    ])
    event_model = fit_event_model(event_x, event_y, event_markets)
    amplitude_model = fit_amplitude_model(
        all_train_x, all_train_y, all_train_markets
    )
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(event_model, models_dir / "neg_e.joblib")
    joblib.dump(amplitude_model, models_dir / "neg_a.joblib")

    val_x, val_y, val_markets = _all_source_validation(source_val)
    val_probability = event_model.predict_proba(val_x)[:, 1]
    threshold, threshold_rows = select_market_macro_threshold(
        val_probability, val_y, val_markets
    )
    print(json.dumps({
        "event": "adapter_frozen", "event_threshold": threshold,
        "source_validation_hours": len(val_y),
    }, ensure_ascii=False), flush=True)

    thresholds = _load_thresholds()
    ercot_report = {}
    phase_directories = {
        "inner": ROOT / args.inner_prediction_dir,
        "outer": ROOT / args.outer_prediction_dir,
    }
    for phase in STAGES:
        original = {
            task: _load_archive(phase_directories[phase], "universal", task, phase)
            for task in TASKS
        }
        da = original["da"]
        features = _ercot_feature_cube(
            base, list(da["nodes"].astype(str)), phase, da, cache_dir
        )
        flat = features.reshape(-1, len(FEATURE_NAMES))
        probability = event_model.predict_proba(flat)[:, 1].reshape(da["prediction"].shape)
        amplitude = amplitude_model.predict(flat).reshape(da["prediction"].shape)
        neg_e, neg_full, triggered = apply_negative_price_adapter(
            da["prediction"], probability, amplitude, threshold=threshold
        )
        predictions = {
            "original": da["prediction"], "neg_e": neg_e,
            "neg_e_neg_a": neg_full,
        }
        forecast = {
            system: _forecast_report(prediction, da["target_da"], thresholds)
            for system, prediction in predictions.items()
        }
        for system in ("neg_e", "neg_e_neg_a"):
            _write_prediction(
                output_dir, phase, system, da, predictions[system], triggered,
                threshold,
            )

        profit = {}
        paired = {}
        if not args.forecast_only:
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
                        "event": "adapter_settlement_node_complete",
                        "phase": phase, "completed": count,
                        "total": len(futures), "system": system, "node": node,
                    }, ensure_ascii=False), flush=True)
            profit = {
                system: _aggregate_profit(completed[system], thresholds)
                for system in SYSTEMS
            }
            paired = {
                system: _paired(profit[system], profit["original"])
                for system in SYSTEMS if system != "original"
            }
        ercot_report[phase] = {
            "forecast": forecast, "profit": profit,
            "paired_vs_original": paired,
            "triggered_hours": int(triggered.sum()),
            "triggered_share": float(triggered.mean()),
            "event_probability_metrics": event_metrics(
                probability, da["target_da"], threshold
            ),
        }

    report = {
        "experiment": "negative_price_adapter_v1",
        "status": "source_trained_target_free_ercot_mechanism_evaluation",
        "training_markets": list(SOURCE_MARKETS),
        "training_years": "2020-2022",
        "threshold_selection_data": "CAISO/NYISO/PJM 2023 only",
        "uses_ercot_labels_for_training_or_selection": False,
        "selection_allowed_on_ercot": False,
        "test_2025_read": False,
        "event_threshold": threshold,
        "event_training_sampling": sampling,
        "source_validation_thresholds": threshold_rows,
        "event_coefficients": _coefficient_report(event_model, "event"),
        "amplitude_coefficients": _coefficient_report(amplitude_model, "amplitude"),
        "ercot": ercot_report,
        "runtime_seconds": perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(report, output_dir / "summary.md")
    print(json.dumps({
        "event": "negative_price_adapter_complete",
        "output_dir": output_dir.relative_to(ROOT).as_posix(),
        "runtime_seconds": report["runtime_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

