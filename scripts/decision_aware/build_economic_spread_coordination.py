#!/usr/bin/env python
"""Fit a frozen, interpretable DA/RT-at-DA economic spread coordinator.

The coordinator is fitted once on the inner (2023) archives and then applied
unchanged to both inner and outer (2024) predictions.  It never changes the DA
or rolling-RT forecasts; only RT-at-DA is reconstructed from the frozen DA
forecast and the coordinated DA-minus-RT spread.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]
FEATURE_NAMES = (
    "predicted_spread",
    "absolute_predicted_spread",
    "da_relative_to_daily_mean",
    "rt_at_da_relative_to_daily_mean",
    "hour_sin",
    "hour_cos",
)


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_archive(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def validate_triplet(tasks: dict[str, dict[str, np.ndarray]]) -> None:
    reference = tasks["da"]
    for task, archive in tasks.items():
        if not np.array_equal(reference["nodes"], archive["nodes"]):
            raise ValueError(f"node mismatch for {task}")
        if task != "rt" and not np.array_equal(reference["dates"], archive["dates"]):
            raise ValueError(f"date mismatch for {task}")
    if reference["prediction"].shape != tasks["rt_at_da"]["prediction"].shape:
        raise ValueError("DA and RT-at-DA prediction shapes differ")


def raw_features(da: np.ndarray, rt_at_da: np.ndarray) -> np.ndarray:
    if da.shape != rt_at_da.shape or da.ndim != 3 or da.shape[-1] != 24:
        raise ValueError("expected DA and RT-at-DA arrays shaped (node, day, 24)")
    spread = da - rt_at_da
    hours = np.arange(24, dtype=np.float64)
    hour_sin = np.broadcast_to(np.sin(2.0 * np.pi * hours / 24.0), da.shape)
    hour_cos = np.broadcast_to(np.cos(2.0 * np.pi * hours / 24.0), da.shape)
    values = np.stack(
        [
            spread,
            np.abs(spread),
            da - da.mean(axis=-1, keepdims=True),
            rt_at_da - rt_at_da.mean(axis=-1, keepdims=True),
            hour_sin,
            hour_cos,
        ],
        axis=-1,
    )
    return values.reshape(-1, len(FEATURE_NAMES))


def economic_state(realized_spread: np.ndarray, threshold: float) -> np.ndarray:
    state = np.zeros(realized_spread.shape, dtype=np.int8)
    state[realized_spread > threshold] = 1
    state[realized_spread < -threshold] = -1
    return state


def magnitude_design(
    predicted_spread: np.ndarray, threshold: float, scale: float | None = None
) -> np.ndarray:
    absolute = np.abs(predicted_spread).reshape(-1).astype(np.float64)
    scale = max(float(np.std(absolute)) if scale is None else float(scale), 1.0)
    return np.column_stack(
        [
            np.ones_like(absolute),
            absolute / scale,
            np.maximum(absolute - threshold, 0.0) / scale,
        ]
    )


def fit_coordinator(
    da_prediction: np.ndarray,
    rt_prediction: np.ndarray,
    realized_spread: np.ndarray,
    threshold: float,
) -> dict:
    features = raw_features(da_prediction, rt_prediction)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1.0e-8] = 1.0
    standardized = (features - mean) / std
    target_state = economic_state(realized_spread, threshold).reshape(-1)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    classifier.fit(standardized, target_state)

    magnitude_scale = max(
        float(np.std(np.abs((da_prediction - rt_prediction).reshape(-1)))), 1.0
    )
    design = magnitude_design(
        da_prediction - rt_prediction, threshold, scale=magnitude_scale
    )
    realized_abs = np.abs(realized_spread).reshape(-1).astype(np.float64)
    magnitude = {}
    for state in (-1, 1):
        mask = target_state == state
        if int(mask.sum()) < design.shape[1]:
            raise ValueError(f"not enough state {state} samples for magnitude fit")
        coefficients, residual_norm = nnls(design[mask], realized_abs[mask])
        magnitude[str(state)] = {
            "coefficients": coefficients.tolist(),
            "residual_norm": float(residual_norm),
            "sample_count": int(mask.sum()),
        }
    return {
        "threshold": float(threshold),
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "classes": classifier.classes_.astype(int).tolist(),
        "classifier_coefficients": classifier.coef_.tolist(),
        "classifier_intercepts": classifier.intercept_.tolist(),
        "magnitude": magnitude,
        "magnitude_scale": magnitude_scale,
        "training_state_counts": {
            str(state): int(np.sum(target_state == state)) for state in (-1, 0, 1)
        },
        "fixed_settings": {
            "classifier_C": 1.0,
            "classifier_class_weight": "balanced",
            "classifier_solver": "lbfgs",
            "magnitude_fit": "nonnegative_least_squares",
            "state_decision": "argmax",
        },
    }


def _classifier_probabilities(features: np.ndarray, model: dict) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    coefficients = np.asarray(model["classifier_coefficients"], dtype=np.float64)
    intercepts = np.asarray(model["classifier_intercepts"], dtype=np.float64)
    logits = ((features - mean) / std) @ coefficients.T + intercepts
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def apply_coordinator(
    da_prediction: np.ndarray, rt_prediction: np.ndarray, model: dict
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    threshold = float(model["threshold"])
    original_spread = da_prediction.astype(np.float64) - rt_prediction.astype(np.float64)
    probabilities = _classifier_probabilities(raw_features(da_prediction, rt_prediction), model)
    classes = np.asarray(model["classes"], dtype=np.int8)
    predicted_state = classes[np.argmax(probabilities, axis=1)]
    design = magnitude_design(
        original_spread, threshold, scale=float(model["magnitude_scale"])
    )
    coordinated = np.clip(original_spread.reshape(-1), -threshold, threshold)
    for state in (-1, 1):
        coefficients = np.asarray(model["magnitude"][str(state)]["coefficients"], dtype=np.float64)
        magnitude = np.maximum(design @ coefficients, threshold + 1.0e-3)
        mask = predicted_state == state
        coordinated[mask] = state * magnitude[mask]
    coordinated = coordinated.reshape(original_spread.shape)
    calibrated_rt = da_prediction.astype(np.float64) - coordinated
    diagnostics = {
        "predicted_state": predicted_state.reshape(original_spread.shape),
        "probabilities": probabilities.reshape(*original_spread.shape, 3),
        "original_spread": original_spread,
        "coordinated_spread": coordinated,
    }
    return calibrated_rt.astype(rt_prediction.dtype), diagnostics


def _load_triplet(directory: Path, family: str, phase: str, seed: int) -> dict:
    suffix = f"{phase}_seed{seed}.npz"
    tasks = {
        task: load_archive(directory / f"{family}_{task}_{suffix}")
        for task in ("da", "rt_at_da", "rt")
    }
    validate_triplet(tasks)
    return tasks


def _state_report(diagnostics: dict, realized_spread: np.ndarray, threshold: float) -> dict:
    truth = economic_state(realized_spread, threshold).reshape(-1)
    prediction = diagnostics["predicted_state"].reshape(-1)
    report = {
        "accuracy": float(np.mean(prediction == truth)),
        "predicted_counts": {str(v): int(np.sum(prediction == v)) for v in (-1, 0, 1)},
        "true_counts": {str(v): int(np.sum(truth == v)) for v in (-1, 0, 1)},
        "per_state": {},
    }
    for state in (-1, 0, 1):
        tp = int(np.sum((prediction == state) & (truth == state)))
        predicted = int(np.sum(prediction == state))
        actual = int(np.sum(truth == state))
        report["per_state"][str(state)] = {
            "precision": float(tp / predicted) if predicted else 0.0,
            "recall": float(tp / actual) if actual else 0.0,
        }
    original = diagnostics["original_spread"].reshape(-1)
    coordinated = diagnostics["coordinated_spread"].reshape(-1)
    realized = realized_spread.reshape(-1)
    for label, values in (("original", original), ("coordinated", coordinated)):
        report[label] = {
            "spread_mae": float(np.mean(np.abs(values - realized))),
            "direction_accuracy": float(np.mean(np.sign(values) == np.sign(realized))),
            "economic_crossing_rate": float(np.mean(np.abs(values) > threshold)),
            "mean_absolute_spread": float(np.mean(np.abs(values))),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-dir", required=True)
    parser.add_argument("--outer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-family", default="universal")
    parser.add_argument("--baseline-family", default="current")
    parser.add_argument("--candidate-family", default="econcoord")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=6.0)
    args = parser.parse_args()

    inner = _load_triplet(ROOT / args.inner_dir, args.source_family, "inner", args.seed)
    outer = _load_triplet(ROOT / args.outer_dir, args.source_family, "outer", args.seed)
    inner_realized = (
        inner["da"]["target_da"].astype(np.float64)
        - inner["rt_at_da"]["target_rt"].astype(np.float64)
    )
    model = fit_coordinator(
        inner["da"]["prediction"].astype(np.float64),
        inner["rt_at_da"]["prediction"].astype(np.float64),
        inner_realized,
        args.threshold,
    )

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for phase, tasks in (("inner", inner), ("outer", outer)):
        calibrated, diagnostics = apply_coordinator(
            tasks["da"]["prediction"], tasks["rt_at_da"]["prediction"], model
        )
        realized = (
            tasks["da"]["target_da"].astype(np.float64)
            - tasks["rt_at_da"]["target_rt"].astype(np.float64)
        )
        reports[phase] = _state_report(diagnostics, realized, args.threshold)
        suffix = f"{phase}_seed{args.seed}.npz"
        for task in ("da", "rt_at_da", "rt"):
            baseline = dict(tasks[task])
            baseline_metadata = {
                "family": args.baseline_family,
                "source_family": args.source_family,
                "task": task,
                "phase": phase,
                "method": "unchanged_frozen_baseline",
            }
            baseline["metadata"] = np.asarray(
                json.dumps(baseline_metadata, ensure_ascii=False)
            )
            save_archive(output_dir / f"{args.baseline_family}_{task}_{suffix}", baseline)
            candidate = dict(tasks[task])
            if task == "rt_at_da":
                candidate["prediction"] = calibrated
            metadata = {
                "family": args.candidate_family,
                "source_family": args.source_family,
                "task": task,
                "phase": phase,
                "method": "frozen_economic_spread_coordination_v1",
                "fit_phase": "inner_2023",
                "application_phase": phase,
                "threshold_usd_per_mwh": args.threshold,
                "da_and_rt_backbones_retrained": False,
                "realized_target_used_at_inference": False,
                "selection_allowed": phase == "inner",
            }
            candidate["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
            save_archive(output_dir / f"{args.candidate_family}_{task}_{suffix}", candidate)

    artifact = {
        "experiment": "frozen_economic_spread_coordination_v1",
        "status": "fitted_on_2023_applied_unchanged_to_2024",
        "selection_allowed": {"inner_2023": True, "outer_2024": False},
        "source_family": args.source_family,
        "seed": args.seed,
        "model": model,
        "diagnostics": reports,
        "prediction_dir": args.output_dir,
    }
    path = output_dir / "economic_spread_coordination_model_and_diagnostics.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "artifact": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
