"""正尖峰概率门控(P-E)三折可用性审计的纯函数。

本模块只实现概率校准、冻结阈值、总体/节点等权指标和硬门判定。
它不包含条件幅度(P-A)、ERCOT或2025入口。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from decision_aware.positive_spike_day_audit import (
    binary_metrics,
    market_equal_weights,
    quantile_calibration,
)


THRESHOLD_GRID = tuple(float(value) for value in np.arange(0.01, 1.00, 0.01))


@dataclass(frozen=True)
class PlattCalibrator:
    slope: float
    intercept: float

    def predict(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=np.float64)
        return expit(self.slope * score + self.intercept)


def fit_platt_calibrator(
    score: np.ndarray,
    truth: np.ndarray,
    markets: Sequence[str],
) -> PlattCalibrator:
    """用源市场等权样本拟合共享二参数Platt校准。"""
    score = np.asarray(score, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    markets = np.asarray(markets).astype(str)
    if score.ndim != 1 or len(score) != len(truth) or len(score) != len(markets):
        raise ValueError("Platt输入长度不一致")
    if not np.isfinite(score).all() or not np.isfinite(truth).all():
        raise ValueError("Platt输入包含非有限值")
    if not truth.any() or truth.all():
        raise ValueError("Platt校准集必须同时包含事件与非事件")
    weights = market_equal_weights(markets)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = parameters
        linear = slope * score + intercept
        probability = expit(linear)
        eps = 1e-12
        loss = -np.average(
            truth * np.log(np.clip(probability, eps, 1.0))
            + (1.0 - truth) * np.log(np.clip(1.0 - probability, eps, 1.0)),
            weights=weights,
        )
        residual = probability - truth
        gradient = np.asarray([
            np.average(residual * score, weights=weights),
            np.average(residual, weights=weights),
        ])
        return float(loss), gradient

    fitted = minimize(
        lambda value: objective(value)[0],
        x0=np.asarray([1.0, 0.0]),
        jac=lambda value: objective(value)[1],
        method="L-BFGS-B",
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Platt校准失败: {fitted.message}")
    return PlattCalibrator(float(fitted.x[0]), float(fitted.x[1]))


def select_fpr_budget_threshold(
    probability: np.ndarray,
    truth: np.ndarray,
    markets: Sequence[str],
    *,
    candidates: Sequence[float] = THRESHOLD_GRID,
    source_fpr_budget: float = 0.05,
) -> tuple[float | None, list[dict]]:
    """选择每个源市场FPR<=预算且recall>0的最低冻结阈值。"""
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    markets = np.asarray(markets).astype(str)
    rows: list[dict] = []
    for candidate in candidates:
        by_market = {
            market: binary_metrics(
                probability[markets == market], truth[markets == market], float(candidate)
            )
            for market in sorted(set(markets.tolist()))
        }
        feasible = all(
            metric["false_positive_rate"] <= source_fpr_budget
            and metric["recall"] > 0.0
            for metric in by_market.values()
        )
        rows.append({
            "threshold": float(candidate),
            "feasible": bool(feasible),
            "by_market": by_market,
        })
    feasible = [row["threshold"] for row in rows if row["feasible"]]
    return (float(min(feasible)) if feasible else None), rows


def expected_calibration_error(calibration: Sequence[dict]) -> float:
    samples = sum(int(row["samples"]) for row in calibration)
    if not samples:
        return 0.0
    return float(sum(
        int(row["samples"]) / samples
        * abs(float(row["mean_probability"]) - float(row["observed_rate"]))
        for row in calibration
    ))


def calibration_slope_intercept(probability: np.ndarray, truth: np.ndarray) -> dict:
    """仅作留出集诊断：拟合truth ~ logit(probability)的斜率与截距。"""
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    score = logit(np.clip(probability, 1e-6, 1.0 - 1e-6))
    calibrator = fit_platt_calibrator(score, truth, np.full(len(truth), "evaluation"))
    return {"slope": calibrator.slope, "intercept": calibrator.intercept}


def metrics_with_optional_threshold(
    probability: np.ndarray,
    truth: np.ndarray,
    threshold: float | None,
) -> dict:
    """无可行阈值时仍报告概率指标，分类指标明确记为None。"""
    if threshold is not None:
        return binary_metrics(probability, truth, threshold)
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    event_rate = float(np.mean(truth))
    from sklearn.metrics import average_precision_score, brier_score_loss

    pr_auc = float(average_precision_score(truth, probability)) if truth.any() else 0.0
    brier = float(brier_score_loss(truth, probability))
    constant = event_rate * (1.0 - event_rate)
    return {
        "samples": int(len(truth)), "events": int(truth.sum()),
        "event_rate": event_rate, "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / event_rate if event_rate else None,
        "precision": None, "recall": None, "f1": None,
        "false_positive_rate": None,
        "brier": brier, "constant_event_rate_brier": constant,
        "brier_improvement_vs_constant": constant - brier,
        "tp": None, "fp": None, "fn": None, "tn": None,
    }


def node_equal_metrics(
    probability: np.ndarray,
    truth: np.ndarray,
    node_ids: np.ndarray,
    nodes: Sequence[str],
    threshold: float | None,
) -> tuple[dict, list[dict]]:
    """先逐节点计算，再对可定义的节点指标作等权平均。"""
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    nodes = np.asarray(nodes).astype(str)
    rows = []
    for node_id, node in enumerate(nodes):
        selected = node_ids == node_id
        if not selected.any():
            continue
        rows.append({
            "node": str(node),
            **metrics_with_optional_threshold(
                probability[selected], truth[selected], threshold
            ),
        })

    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    return {
        "nodes": len(rows),
        "event_rate": mean("event_rate"),
        "pr_auc": mean("pr_auc"),
        "pr_auc_lift": mean("pr_auc_lift"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "false_positive_rate": mean("false_positive_rate"),
        "brier": mean("brier"),
        "constant_event_rate_brier": mean("constant_event_rate_brier"),
        "brier_improvement_vs_constant": mean("brier_improvement_vs_constant"),
    }, rows


def pe_market_gate(
    overall: dict,
    node_equal: dict,
    bootstrap: dict,
    threshold: float | None,
) -> dict:
    checks = {
        "feasible_threshold": threshold is not None,
        "pr_auc_lift": bool(
            overall["pr_auc_lift"] is not None and overall["pr_auc_lift"] >= 1.25
        ),
        "pr_auc_ci": bool(bootstrap["ci95"][0] > 0.0),
        "overall_brier": bool(overall["brier_improvement_vs_constant"] > 0.0),
        "node_equal_brier": bool(node_equal["brier_improvement_vs_constant"] > 0.0),
        "overall_fpr": bool(
            overall["false_positive_rate"] is not None
            and overall["false_positive_rate"] <= 0.10
        ),
        "node_equal_fpr": bool(
            node_equal["false_positive_rate"] is not None
            and node_equal["false_positive_rate"] <= 0.10
        ),
        "nonzero_recall": bool(overall["recall"] is not None and overall["recall"] > 0.0),
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}
