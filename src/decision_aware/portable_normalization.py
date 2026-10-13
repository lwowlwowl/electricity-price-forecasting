"""只依赖起报历史窗口的可迁移归一化。

这里故意不读取节点、市场或目标窗口。这样同一组模型权重可以用于训练时
从未出现的实体，也不会因使用未来目标统计量而产生泄漏。
"""
from __future__ import annotations

from typing import Any

import numpy as np


CONTEXT_ROBUST_MODE = "context_robust"
TRAINING_ENTITY_MODE = "training_entity"
NORMALIZATION_MODES = {CONTEXT_ROBUST_MODE, TRAINING_ENTITY_MODE}


def normalization_mode(config: Any) -> str:
    mode = str(
        getattr(config, "global_normalization_mode", TRAINING_ENTITY_MODE)
    ).lower()
    if mode not in NORMALIZATION_MODES:
        raise ValueError(
            "global_normalization_mode必须是training_entity或context_robust"
        )
    return mode


def robust_context_stats(values: np.ndarray) -> dict[str, np.ndarray | float]:
    """沿时间维计算中位数和稳健尺度，只使用传入的历史窗口。"""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim not in {1, 2} or array.shape[0] < 2:
        raise ValueError("稳健上下文归一化要求形如[T]或[T,C]且T>=2")
    center = np.median(array, axis=0)
    mad = np.median(np.abs(array - center), axis=0)
    scale = np.asarray(1.4826 * mad, dtype=np.float32)
    # 常数或长时间平坦序列的MAD可能为0。回退只读取同一历史窗口的标准差，
    # 最后下限为1，避免把微小数值噪声放大成假尖峰。
    fallback = np.asarray(np.std(array, axis=0), dtype=np.float32)
    scale = np.where(scale > 1.0e-6, scale, fallback)
    scale = np.maximum(scale, 1.0).astype(np.float32)
    if array.ndim == 1:
        return {"mean": float(center), "std": float(scale)}
    return {
        "mean": np.asarray(center, dtype=np.float32),
        "std": np.asarray(scale, dtype=np.float32),
    }


def normalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return (
        (np.asarray(values, dtype=np.float32) - stats["mean"]) / stats["std"]
    ).astype(np.float32)


def portable_contract(context_len: int) -> dict[str, Any]:
    """写入checkpoint的无实体scaler合同；不是任何节点的拟合统计量。"""
    return {
        "mode": CONTEXT_ROBUST_MODE,
        "context_len": int(context_len),
        "center": "median",
        "scale": "1.4826_mad_with_context_std_fallback",
        "scale_floor": 1.0,
        "uses_target_window": False,
        "requires_entity_history": False,
    }
