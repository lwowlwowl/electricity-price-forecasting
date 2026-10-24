#!/usr/bin/env python
"""DA尾部分布v1.1：q99对照、日期聚类bootstrap与Beta单调审计。

只读取CAISO/NYISO/PJM 2020--2024；禁止ERCOT、2025与LP/MPC。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from scripts.decision_aware.calibrate_da_tail_probabilities_lomo import (  # noqa: E402
    MARKETS,
    _features,
    _raw_probability,
)
from scripts.decision_aware.run_da_tail_distribution_posttraining import (  # noqa: E402
    _config_from_checkpoint,
    _json_ready,
    _node_thresholds,
)


BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261020


def _pinball(truth: np.ndarray, prediction: np.ndarray, level: float = 0.99) -> float:
    error = truth - prediction
    return float(np.mean(np.maximum(level * error, (level - 1.0) * error)))


def _training_profiles(train_dataset) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    seasonal: dict[int, np.ndarray] = {}
    historical_tail_mean: dict[int, float] = {}
    for (market, node), dataset in zip(train_dataset.entities, train_dataset.datasets):
        _, node_id = train_dataset.mappings.identity(market, node)
        positions = np.concatenate([window.target_positions for window in dataset.windows])
        values = dataset.price_da[positions].astype(np.float64)
        local = dataset.local_index[positions]
        buckets = np.asarray(local.dayofweek * 24 + local.hour, dtype=np.int64)
        fixed = float(np.quantile(values, 0.99))
        table = np.full(168, fixed, dtype=np.float64)
        for bucket in range(168):
            mask = buckets == bucket
            if mask.any():
                table[bucket] = float(np.quantile(values[mask], 0.99))
        seasonal[node_id] = table
        tail = values[values > fixed]
        historical_tail_mean[node_id] = float(tail.mean())
    return seasonal, historical_tail_mean


def _aligned_metadata(validation_dataset, seasonal_tables: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    collected = {
        "date": [], "fixed_q99": [], "seasonal_q99": [], "rolling_q99": [],
        "truth": [], "node_id": [], "market_id": [],
    }
    for (market, node), dataset in zip(
        validation_dataset.entities, validation_dataset.datasets
    ):
        market_id, node_id = validation_dataset.mappings.identity(market, node)
        table = seasonal_tables[node_id]
        for window in dataset.windows:
            target = window.target_positions
            local = dataset.local_index[target]
            buckets = np.asarray(local.dayofweek * 24 + local.hour, dtype=np.int64)
            context = dataset.price_da[window.context_start:window.context_end]
            date_value = np.datetime64(window.delivery_date, "D").astype(np.int64)
            collected["date"].append(np.full(24, date_value, dtype=np.int64))
            # 训练期固定q99随后由冻结threshold表按node_id注入。
            collected["fixed_q99"].append(np.zeros(24, dtype=np.float64))
            collected["seasonal_q99"].append(table[buckets])
            collected["rolling_q99"].append(np.full(
                24, float(np.quantile(context, 0.99)), dtype=np.float64
            ))
            collected["truth"].append(dataset.price_da[target].astype(np.float64))
            collected["node_id"].append(np.full(24, node_id, dtype=np.int64))
            collected["market_id"].append(np.full(24, market_id, dtype=np.int64))
    return {key: np.concatenate(value) for key, value in collected.items()}


def _inject_fixed_q99(metadata: dict[str, np.ndarray], thresholds: dict[int, dict[str, float]]) -> None:
    metadata["fixed_q99"] = np.asarray([
        thresholds[int(node)]["q99"] for node in metadata["node_id"]
    ], dtype=np.float64)


def _q99_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    fixed_threshold: np.ndarray,
) -> dict:
    event = truth > fixed_threshold
    return {
        "pinball_q99": _pinball(truth, prediction),
        "coverage": float(np.mean(truth <= prediction)),
        "spike_hours": int(event.sum()),
        "spike_mae": float(np.mean(np.abs(prediction[event] - truth[event])))
        if event.any() else float("nan"),
        "mean_q99": float(np.mean(prediction)),
    }


def _aggregate_by_date(
    dates: np.ndarray,
    event: np.ndarray,
    learned_abs_error: np.ndarray,
    comparator_errors: dict[str, np.ndarray],
    calibrated_probability: np.ndarray,
    constant_probability: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    unique, inverse = np.unique(dates, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    event_count = np.bincount(inverse, weights=event.astype(float))
    fields = {
        "count": count,
        "event_count": event_count,
        "learned_error": np.bincount(
            inverse, weights=np.where(event, learned_abs_error, 0.0)
        ),
        "model_brier": np.bincount(
            inverse, weights=(calibrated_probability - event.astype(float)) ** 2
        ),
        "constant_brier": np.bincount(
            inverse, weights=(constant_probability - event.astype(float)) ** 2
        ),
    }
    for name, error in comparator_errors.items():
        fields[f"{name}_error"] = np.bincount(
            inverse, weights=np.where(event, error, 0.0)
        )
    return unique, fields


def _cluster_bootstrap(
    dates: np.ndarray,
    event: np.ndarray,
    learned_abs_error: np.ndarray,
    comparator_errors: dict[str, np.ndarray],
    calibrated_probability: np.ndarray,
    constant_probability: float,
) -> dict:
    unique_dates, fields = _aggregate_by_date(
        dates, event, learned_abs_error, comparator_errors,
        calibrated_probability, constant_probability,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    improvements = {name: [] for name in comparator_errors}
    skills = []
    invalid_spike_draws = 0
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.integers(0, len(unique_dates), size=len(unique_dates))
        event_total = float(fields["event_count"][sampled].sum())
        if event_total <= 0:
            invalid_spike_draws += 1
        else:
            learned = float(fields["learned_error"][sampled].sum()) / event_total
            for name in comparator_errors:
                comparator = float(fields[f"{name}_error"][sampled].sum()) / event_total
                improvements[name].append(comparator - learned)
        constant = float(fields["constant_brier"][sampled].sum())
        model = float(fields["model_brier"][sampled].sum())
        if constant > 0:
            skills.append(1.0 - model / constant)

    total_events = float(fields["event_count"].sum())
    point_learned = float(fields["learned_error"].sum()) / total_events
    point_improvements = {
        name: float(fields[f"{name}_error"].sum()) / total_events - point_learned
        for name in comparator_errors
    }
    point_constant = float(fields["constant_brier"].sum())
    point_model = float(fields["model_brier"].sum())
    point_skill = 1.0 - point_model / point_constant

    def interval(values: list[float], point_estimate: float) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {
            "estimate": float(point_estimate),
            "bootstrap_mean": float(np.mean(array)),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
            "valid_repetitions": int(len(array)),
        }

    return {
        "cluster_unit": "delivery_date",
        "clusters": int(len(unique_dates)),
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "invalid_spike_repetitions": int(invalid_spike_draws),
        "q99_spike_mae_improvement": {
            name: interval(values, point_improvements[name])
            for name, values in improvements.items()
        },
        "brier_skill_vs_constant": interval(skills, point_skill),
    }


def _beta_probability(raw: np.ndarray, coefficients: list[float], intercept: float) -> np.ndarray:
    z = _features(raw) @ np.asarray(coefficients, dtype=np.float64) + float(intercept)
    return 1.0 / (1.0 + np.exp(-z))


def _monotonicity_audit(
    raw: np.ndarray,
    calibrated: np.ndarray,
    event: np.ndarray,
    coefficients: list[float],
) -> dict:
    a, b = (float(value) for value in coefficients)
    lo = float(raw.min())
    hi = float(raw.max())
    grid = np.linspace(lo, hi, 100001)
    derivative_logit = a / grid - b / (1.0 - grid)
    strict = bool(np.all(derivative_logit > 0.0))
    raw_rank = rankdata(raw, method="average")
    calibrated_rank = rankdata(calibrated, method="average")
    rank_correlation = float(np.corrcoef(raw_rank, calibrated_rank)[0, 1])
    raw_pr = float(average_precision_score(event, raw))
    calibrated_pr = float(average_precision_score(event, calibrated))
    return {
        "actual_probability_range": [lo, hi],
        "min_logit_derivative_on_dense_grid": float(derivative_logit.min()),
        "strictly_increasing_on_actual_range": strict,
        "spearman_rank_correlation": rank_correlation,
        "raw_unique_values": int(np.unique(raw).size),
        "calibrated_unique_values": int(np.unique(calibrated).size),
        "raw_pr_auc": raw_pr,
        "calibrated_pr_auc": calibrated_pr,
        "absolute_pr_auc_difference": abs(raw_pr - calibrated_pr),
        "ranking_and_pr_auc_preserved": bool(
            strict
            and abs(rank_correlation - 1.0) <= 1.0e-12
            and abs(raw_pr - calibrated_pr) <= 1.0e-12
        ),
    }


def main() -> None:
    checkpoint_path = ROOT / (
        "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/"
        "da_inner_seed0/best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: PilotConfig = _config_from_checkpoint(checkpoint)
    if tuple(cfg.global_training_markets) != MARKETS:
        raise ValueError("v1.1只允许CAISO/NYISO/PJM三个源市场")
    mappings = load_formal_entity_mappings(ROOT / "configs/nodes.yaml").subset(MARKETS)
    entities = [(mappings.node_to_market[node], node) for node in mappings.node_to_id]
    train_dataset, _ = build_global_task_datasets(
        cfg, "da", entities, mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
    )
    _, validation_2024 = build_global_task_datasets(
        cfg, "da", entities, mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.OUTER_VALIDATION,
    )
    thresholds = _node_thresholds(train_dataset)
    seasonal_tables, historical_tail_mean = _training_profiles(train_dataset)
    metadata = _aligned_metadata(validation_2024, seasonal_tables)
    _inject_fixed_q99(metadata, thresholds)

    inputs = ROOT / "data/results/da_tail_distribution_calibration_inputs_v1"
    hourly = dict(np.load(inputs / "hourly_2024.npz"))
    for key in ("truth", "node_id", "market_id"):
        if not np.allclose(metadata[key], hourly[key]):
            raise RuntimeError(f"保存数组与Dataset顺序不一致: {key}")
    calibration = json.loads((
        ROOT / "data/results/da_tail_probability_calibration_lomo_v1/summary.json"
    ).read_text(encoding="utf-8"))
    market_names = [
        market for market, _ in sorted(
            mappings.market_to_id.items(), key=lambda item: item[1]
        )
    ]
    learned_q99 = hourly["quantiles"][:, -1].astype(np.float64)
    raw_probability = _raw_probability(hourly["quantiles"], hourly["node_id"], thresholds)
    folds = {}
    for market in MARKETS:
        market_id = market_names.index(market)
        mask = hourly["market_id"] == market_id
        truth = hourly["truth"][mask].astype(np.float64)
        fixed = metadata["fixed_q99"][mask]
        event = truth > fixed
        learned = learned_q99[mask]
        comparators = {
            "fixed_historical_q99": fixed,
            "seasonal_hour_of_week_q99": metadata["seasonal_q99"][mask],
            "rolling_168h_q99": metadata["rolling_q99"][mask],
        }
        fold = calibration["folds"][market]
        raw = raw_probability[mask]
        calibrated = _beta_probability(raw, fold["coefficients"], fold["intercept"])
        constant_probability = float(event.mean())
        bootstrap = _cluster_bootstrap(
            metadata["date"][mask],
            event,
            np.abs(learned - truth),
            {name: np.abs(value - truth) for name, value in comparators.items()},
            calibrated,
            constant_probability,
        )
        monotonicity = _monotonicity_audit(
            raw, calibrated, event, fold["coefficients"]
        )
        positive_comparator_intervals = sum(
            item["ci95"][0] > 0.0
            for item in bootstrap["q99_spike_mae_improvement"].values()
        )
        passes = bool(
            positive_comparator_intervals >= 2
            and bootstrap["brier_skill_vs_constant"]["ci95"][0] > 0.0
            and monotonicity["strictly_increasing_on_actual_range"]
            and monotonicity["ranking_and_pr_auc_preserved"]
        )
        folds[market] = {
            "samples": int(mask.sum()),
            "events": int(event.sum()),
            "event_rate": constant_probability,
            "learned_q99": _q99_metrics(truth, learned, fixed),
            "comparators": {
                name: _q99_metrics(truth, prediction, fixed)
                for name, prediction in comparators.items()
            },
            "bootstrap": bootstrap,
            "beta_monotonicity": monotonicity,
            "positive_comparator_intervals": int(positive_comparator_intervals),
            "passes": passes,
        }

    passed = sum(values["passes"] for values in folds.values())
    summary = _json_ready({
        "experiment": "da_tail_distribution_v11_source_audit",
        "status": "complete",
        "uses_ercot": False,
        "runs_lp_mpc": False,
        "test_2025_read": False,
        "scope_note": (
            "Beta calibrator is leave-one-market-out; the frozen Transformer "
            "and quantile head were trained jointly on all three source markets."
        ),
        "estimand_note": (
            "q99 is a conditional quantile, not a conditional tail mean or "
            "risk-neutral expectation."
        ),
        "historical_tail_means_for_separate_gpd_audit": historical_tail_mean,
        "folds": folds,
        "gate": {
            "passed_markets": int(passed),
            "required_markets": 2,
            "passes": bool(passed >= 2),
            "requires_source_gpd_validation": True,
            "allows_ercot_evaluation": False,
        },
    })
    output = ROOT / "data/results/da_tail_distribution_v11_source_audit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "aligned_2024.npz",
        **metadata,
        learned_q99=learned_q99,
        raw_probability=raw_probability,
    )
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
