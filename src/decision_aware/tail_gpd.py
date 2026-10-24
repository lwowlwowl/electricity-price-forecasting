"""可解释的POT-GPD条件尾部均值工具。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


XI_LIMIT = 0.8
EPSILON = 1.0e-8


def gpd_shape(raw_shape: float) -> float:
    """把无约束参数映射到(-0.8, 0.8)，从而保证尾部均值存在。"""
    return float(XI_LIMIT * np.tanh(float(raw_shape)))


def conditional_scale(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    linear = coefficients[0] + features @ coefficients[1:]
    return np.exp(np.clip(linear, -20.0, 20.0))


def gpd_negative_log_likelihood(
    parameters: np.ndarray,
    exceedance: np.ndarray,
    features: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """GPD负对数似然；非法支持域返回有限大惩罚。"""
    beta = np.asarray(parameters[:-1], dtype=np.float64)
    xi = gpd_shape(float(parameters[-1]))
    sigma = conditional_scale(features, beta)
    z = np.asarray(exceedance, dtype=np.float64)
    if abs(xi) < 1.0e-6:
        loss = np.log(sigma) + z / sigma
    else:
        support = 1.0 + xi * z / sigma
        if np.any(support <= 0.0):
            violation = np.maximum(-support, 0.0).sum()
            return float(1.0e12 + 1.0e9 * violation)
        loss = np.log(sigma) + (1.0 / xi + 1.0) * np.log(support)
    if sample_weight is None:
        return float(np.mean(loss))
    weights = np.asarray(sample_weight, dtype=np.float64)
    return float(np.sum(weights * loss) / np.sum(weights))


def fit_gpd_regression(
    exceedance: np.ndarray,
    features: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """用一次L-BFGS-B拟合固定特征的GPD尺度回归与共享形状。"""
    z = np.asarray(exceedance, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    initial = np.zeros(x.shape[1] + 2, dtype=np.float64)
    initial[0] = float(np.log(max(np.median(z), 1.0e-3)))
    # 不从xi=0精确起步：指数极限的数值分支在0点会让有限差分看不到
    # 形状方向。0.1只是稳定的优化初值，不是搜索的超参数。
    initial[-1] = float(np.arctanh(0.1 / XI_LIMIT))
    result = minimize(
        gpd_negative_log_likelihood,
        initial,
        args=(z, x, sample_weight),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "negative_log_likelihood": float(result.fun),
        "coefficients": parameters[:-1],
        "raw_shape": float(parameters[-1]),
        "xi": gpd_shape(float(parameters[-1])),
    }


def conditional_tail_mean(
    u_fit: np.ndarray,
    u_target: np.ndarray,
    sigma_fit: np.ndarray,
    xi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """由GPD阈值稳定性计算E[Y|Y>u_target,x]。

    返回预测与合法掩码。需要xi<1且目标阈值处尺度为正。
    """
    u_fit = np.asarray(u_fit, dtype=np.float64)
    u_target = np.asarray(u_target, dtype=np.float64)
    sigma_target = np.asarray(sigma_fit, dtype=np.float64) + xi * (u_target - u_fit)
    valid = np.isfinite(sigma_target) & (sigma_target > 0.0) & (xi < 1.0)
    prediction = np.full_like(sigma_target, np.nan, dtype=np.float64)
    prediction[valid] = u_target[valid] + sigma_target[valid] / (1.0 - xi)
    return prediction, valid
