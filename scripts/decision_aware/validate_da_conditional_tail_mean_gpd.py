#!/usr/bin/env python
"""源市场留一市场POT-GPD条件超阈值均值验证。

只使用CAISO/NYISO/PJM 2020--2024，禁止ERCOT、2025和LP/MPC。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.tail_gpd import (  # noqa: E402
    conditional_scale,
    conditional_tail_mean,
    fit_gpd_regression,
)
from scripts.decision_aware.calibrate_da_tail_probabilities_lomo import MARKETS  # noqa: E402
from scripts.decision_aware.run_da_tail_distribution_posttraining import (  # noqa: E402
    _config_from_checkpoint,
    _json_ready,
)


def _thresholds(train_dataset) -> dict[int, dict[str, float]]:
    result = {}
    for (market, node), dataset in zip(train_dataset.entities, train_dataset.datasets):
        _, node_id = train_dataset.mappings.identity(market, node)
        values = np.concatenate([
            dataset.price_da[window.target_positions]
            for window in dataset.windows
        ]).astype(np.float64)
        q95 = float(np.quantile(values, 0.95))
        q99 = float(np.quantile(values, 0.99))
        tail = values[values > q99]
        result[node_id] = {
            "q95": q95,
            "q99": q99,
            "historical_tail_mean": float(tail.mean()),
        }
    return result


def _vectors(node_ids: np.ndarray, thresholds: dict[int, dict[str, float]]):
    u95 = np.asarray([thresholds[int(node)]["q95"] for node in node_ids])
    u99 = np.asarray([thresholds[int(node)]["q99"] for node in node_ids])
    historical = np.asarray([
        thresholds[int(node)]["historical_tail_mean"] for node in node_ids
    ])
    return u95, u99, historical


def _raw_features(quantiles: np.ndarray) -> np.ndarray:
    q50 = quantiles[:, 4].astype(np.float64)
    q95 = quantiles[:, 7].astype(np.float64)
    q99 = quantiles[:, 8].astype(np.float64)
    return np.column_stack([
        np.log(np.maximum(q99 - q95, 1.0e-3)),
        np.log(np.maximum(q95 - q50, 1.0e-3)),
    ])


def _standardize_fit_apply(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    return (train - mean) / std, (test - mean) / std, mean, std


def _market_equal_weights(market_ids: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(market_ids), dtype=np.float64)
    unique = np.unique(market_ids)
    for market_id in unique:
        mask = market_ids == market_id
        weights[mask] = 1.0 / mask.sum()
    weights *= len(weights) / weights.sum()
    return weights


def _evaluate_prediction(truth: np.ndarray, prediction: np.ndarray) -> dict:
    finite = np.isfinite(prediction)
    error = prediction[finite] - truth[finite]
    return {
        "samples": int(len(truth)),
        "finite_predictions": int(finite.sum()),
        "nonfinite_predictions": int((~finite).sum()),
        "mae": float(np.mean(np.abs(error))) if finite.any() else float("nan"),
        "bias": float(np.mean(error)) if finite.any() else float("nan"),
    }


def main() -> None:
    checkpoint = torch.load(
        ROOT / "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed0/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg: PilotConfig = _config_from_checkpoint(checkpoint)
    if tuple(cfg.global_training_markets) != MARKETS:
        raise ValueError("GPD验证只允许三个源市场")
    mappings = load_formal_entity_mappings(ROOT / "configs/nodes.yaml").subset(MARKETS)
    entities = [(mappings.node_to_market[node], node) for node in mappings.node_to_id]
    train_dataset, _ = build_global_task_datasets(
        cfg, "da", entities, mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
    )
    thresholds = _thresholds(train_dataset)
    input_dir = ROOT / "data/results/da_tail_distribution_calibration_inputs_v1"
    arrays = {
        year: dict(np.load(input_dir / f"hourly_{year}.npz"))
        for year in ("2023", "2024")
    }
    market_names = [
        market for market, _ in sorted(
            mappings.market_to_id.items(), key=lambda item: item[1]
        )
    ]
    folds = {}
    for holdout in MARKETS:
        holdout_id = market_names.index(holdout)
        train_mask_market = arrays["2023"]["market_id"] != holdout_id
        test_mask_market = arrays["2024"]["market_id"] == holdout_id
        train_nodes = arrays["2023"]["node_id"]
        test_nodes = arrays["2024"]["node_id"]
        train_u95, _, _ = _vectors(train_nodes, thresholds)
        test_u95, test_u99, test_historical = _vectors(test_nodes, thresholds)
        train_exceed = train_mask_market & (arrays["2023"]["truth"] > train_u95)
        test_tail = test_mask_market & (arrays["2024"]["truth"] > test_u99)

        raw_train = _raw_features(arrays["2023"]["quantiles"])[train_exceed]
        raw_test = _raw_features(arrays["2024"]["quantiles"])[test_tail]
        x_train, x_test, feature_mean, feature_std = _standardize_fit_apply(
            raw_train, raw_test
        )
        z_train = (
            arrays["2023"]["truth"][train_exceed] - train_u95[train_exceed]
        ).astype(np.float64)
        train_market_ids = arrays["2023"]["market_id"][train_exceed]
        weights = _market_equal_weights(train_market_ids)

        conditional_fit = fit_gpd_regression(z_train, x_train, weights)
        static_fit = fit_gpd_regression(
            z_train, np.empty((len(z_train), 0), dtype=np.float64), weights
        )
        eval_u95 = test_u95[test_tail]
        eval_u99 = test_u99[test_tail]
        truth = arrays["2024"]["truth"][test_tail].astype(np.float64)
        sigma_conditional = conditional_scale(
            x_test, conditional_fit["coefficients"]
        )
        sigma_static = conditional_scale(
            np.empty((len(truth), 0), dtype=np.float64),
            static_fit["coefficients"],
        )
        prediction_conditional, valid_conditional = conditional_tail_mean(
            eval_u95, eval_u99, sigma_conditional, conditional_fit["xi"]
        )
        prediction_static, valid_static = conditional_tail_mean(
            eval_u95, eval_u99, sigma_static, static_fit["xi"]
        )
        historical = test_historical[test_tail]
        conditional_metrics = _evaluate_prediction(truth, prediction_conditional)
        static_metrics = _evaluate_prediction(truth, prediction_static)
        historical_metrics = _evaluate_prediction(truth, historical)
        support_violations = int((~valid_conditional).sum())
        passes = bool(
            conditional_fit["xi"] < 1.0
            and support_violations == 0
            and conditional_metrics["nonfinite_predictions"] == 0
            and conditional_metrics["mae"] < static_metrics["mae"]
            and conditional_metrics["mae"] < historical_metrics["mae"]
        )
        folds[holdout] = {
            "fit_markets": [market for market in MARKETS if market != holdout],
            "fit_year": 2023,
            "evaluation_year": 2024,
            "fit_exceedances_above_q95": int(train_exceed.sum()),
            "evaluation_exceedances_above_q99": int(test_tail.sum()),
            "feature_definition": ["log(q99-q95)", "log(q95-q50)"],
            "feature_standardization_mean": feature_mean.tolist(),
            "feature_standardization_std": feature_std.tolist(),
            "conditional_fit": {
                **conditional_fit,
                "coefficients": conditional_fit["coefficients"].tolist(),
            },
            "static_fit": {
                **static_fit,
                "coefficients": static_fit["coefficients"].tolist(),
            },
            "conditional_gpd_tail_mean": conditional_metrics,
            "static_gpd_tail_mean": static_metrics,
            "node_historical_tail_mean": historical_metrics,
            "conditional_support_violations": support_violations,
            "static_support_violations": int((~valid_static).sum()),
            "passes": passes,
        }

    passed = sum(fold["passes"] for fold in folds.values())
    v11 = json.loads((
        ROOT / "data/results/da_tail_distribution_v11_source_audit/summary.json"
    ).read_text(encoding="utf-8"))
    summary = _json_ready({
        "experiment": "da_conditional_tail_mean_gpd_lomo_source_v1",
        "status": "complete",
        "uses_ercot": False,
        "runs_lp_mpc": False,
        "test_2025_read": False,
        "estimand": "E[Y | Y > node historical q99, x] under POT-GPD",
        "scope_note": (
            "The GPD head is leave-one-market-out. The frozen Transformer and "
            "quantile head are not leave-one-market-out."
        ),
        "folds": folds,
        "gpd_gate": {
            "passed_markets": int(passed),
            "required_markets": 2,
            "passes": bool(passed >= 2),
        },
        "combined_gate": {
            "v11_q99_beta_passes": bool(v11["gate"]["passes"]),
            "gpd_passes": bool(passed >= 2),
            "allows_ercot_evaluation": bool(
                v11["gate"]["passes"] and passed >= 2
            ),
        },
    })
    output = ROOT / "data/results/da_conditional_tail_mean_gpd_source_v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "gpd_gate": summary["gpd_gate"],
        "combined_gate": summary["combined_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
