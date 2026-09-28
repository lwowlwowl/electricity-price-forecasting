"""独立DA任务的可复现基线与统一特征构造。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DA_FEATURE_KEYS = (
    "price_da_ctx",
    "price_rt_ctx",
    "load_ctx",
    "system_ctx",
    "cal_ctx",
    "cal_tgt",
)


def flatten_da_sample(sample: dict) -> np.ndarray:
    """把合法可见的历史输入与目标日历展平，供树模型使用。"""
    return np.concatenate(
        [sample[key].detach().cpu().numpy().reshape(-1) for key in DA_FEATURE_KEYS]
    ).astype(np.float32)


def dataset_to_numpy(dataset, indices: Sequence[int] | None = None):
    """返回树模型使用的X，以及真实DA价格Y。"""
    selected: Iterable[int] = range(len(dataset)) if indices is None else indices
    features, targets = [], []
    for index in selected:
        sample = dataset[index]
        features.append(flatten_da_sample(sample))
        targets.append(sample["price_da_tgt"].numpy())
    return np.stack(features), np.stack(targets)


def weekly_seasonal_predictions(dataset, indices: Sequence[int] | None = None):
    """用每个目标时点向前168个实际小时的价格，严格使用起报前数据。

    使用位置差而不是把当地时间减7天，可避免秋季夏令时回拨时01:00出现两次
    所造成的歧义。除切换周外，它就等于“上周同一当地小时”。
    """
    selected = list(range(len(dataset)) if indices is None else indices)
    predictions = []
    for index in selected:
        window = dataset.windows[index]
        reference_positions = window.target_positions - (7 * 24)
        context_positions = set(range(window.context_start, window.context_end))
        if np.any(reference_positions < 0) or not all(
            int(position) in context_positions for position in reference_positions
        ):
            # 极端缺口或夏令时边界下没有完整同相位日时，退化为最近24个已知小时。
            reference_positions = np.arange(window.context_end - 24, window.context_end)
        predictions.append(dataset.price_da[reference_positions].copy())
    return np.stack(predictions).astype(np.float32)


def fit_extra_trees(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """不依赖额外下载的强树基线；原生支持24维多输出。"""
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=240,
        max_features=0.7,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def fit_xgboost(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """可选XGBoost；依赖缺失时明确失败，不静默冒充。"""
    try:
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("当前项目环境未安装xgboost；没有用其他模型冒充XGBoost") from exc

    base = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=180,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=seed,
    )
    # Windows上的loky多进程会复制整套训练数据，并可能在受限环境中出现
    # BrokenProcessPool。24个预测小时顺序拟合更稳定，也保持结果可复现。
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(X, y)
    return model


class DAGRUBaseline(nn.Module):
    """小型RNN基线：历史序列GRU + 目标日历条件，输出24小时DA价格。"""

    def __init__(self, hidden_size: int = 64, calendar_dim: int = 8):
        super().__init__()
        self.gru = nn.GRU(13, hidden_size, batch_first=True)
        self.calendar_proj = nn.Linear(calendar_dim, hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def history_tensor(batch: dict) -> torch.Tensor:
        return torch.cat(
            [
                batch["price_da_ctx"].unsqueeze(-1),
                batch["price_rt_ctx"].unsqueeze(-1),
                batch["load_ctx"],
                batch["system_ctx"],
                batch["cal_ctx"],
            ],
            dim=-1,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        history = self.history_tensor(batch)
        _, hidden = self.gru(history)
        query = hidden[-1].unsqueeze(1) + self.calendar_proj(batch["cal_tgt"])
        return self.head(query).squeeze(-1)

    @staticmethod
    def to_price(normalized: torch.Tensor, batch: dict) -> torch.Tensor:
        return normalized * batch["price_da_std"].unsqueeze(-1) + batch[
            "price_da_mean"
        ].unsqueeze(-1)


@dataclass(frozen=True)
class DAMetrics:
    mae: float
    rmse: float
    mean_revenue: float
    mean_oracle: float
    mean_regret: float
    positive_day_rate: float

