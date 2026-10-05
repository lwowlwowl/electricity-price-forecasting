"""完整三模型系统使用的传统基线。

这里只定义预测器和合法特征，不包含任何电池策略或收益公式。所有系统的
HardTopK、连续SOC和双结算统一交给 :mod:`decision_aware.joint_system`。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


HISTORY_KEYS = (
    "price_da_ctx",
    "price_rt_ctx",
    "load_ctx",
    "system_ctx",
    "cal_ctx",
)


def flatten_baseline_sample(sample: dict, task: str) -> np.ndarray:
    """把某一任务在起报时合法可见的输入展平给树模型。

    DA和RT-at-DA在D-1 10:00只能看到历史流与目标日历；滚动RT还可以看到
    已公布的目标小时DA价。真实目标价格从不进入特征。
    """
    if task not in {"da", "rt_at_da", "rt"}:
        raise ValueError(f"未知任务: {task}")
    keys = list(HISTORY_KEYS)
    if task == "rt":
        keys.append("price_da_tgt_known")
    keys.append("cal_tgt")
    return np.concatenate([
        sample[key].detach().cpu().numpy().reshape(-1) for key in keys
    ]).astype(np.float32)


def baseline_dataset_to_numpy(
    dataset,
    task: str,
    indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回树模型的特征X与相应任务的价格标签Y。"""
    target_key = {
        "da": "price_da_tgt",
        "rt_at_da": "price_rt_at_da_tgt",
        "rt": "price_rt_tgt",
    }.get(task)
    if target_key is None:
        raise ValueError(f"未知任务: {task}")
    selected: Iterable[int] = range(len(dataset)) if indices is None else indices
    features, targets = [], []
    for index in selected:
        sample = dataset[index]
        features.append(flatten_baseline_sample(sample, task))
        targets.append(sample[target_key].detach().cpu().numpy())
    return np.stack(features), np.stack(targets)


def weekly_seasonal_task_predictions(
    dataset,
    task: str,
    indices: Sequence[int] | None = None,
) -> np.ndarray:
    """以目标时点前168小时的同类价格作为预测。

    位置差而非当地时间减7天可以避开夏令时回拨的重复小时。每一个引用位置
    都必须早于起报时点；不满足时立即报错，绝不静默读取未来标签。
    """
    if task == "da":
        values = dataset.price_da
    elif task in {"rt_at_da", "rt"}:
        values = dataset.price_rt
    else:
        raise ValueError(f"未知任务: {task}")
    selected = list(range(len(dataset)) if indices is None else indices)
    predictions = []
    for index in selected:
        window = dataset.windows[index]
        references = np.asarray(window.target_positions) - 7 * 24
        issue_position = (
            window.context_end if hasattr(window, "context_end")
            else window.issue_position
        )
        if np.any(references < window.context_start) or np.any(
            references >= issue_position
        ):
            raise ValueError(
                f"{task} seasonal样本{index}的t-168不在合法历史窗口内"
            )
        predictions.append(values[references].copy())
    return np.stack(predictions).astype(np.float32)


def fit_joint_xgboost(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """按现有DA基线的冻结参数拟合多输出XGBoost。"""
    try:
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("项目环境没有安装xgboost") from exc

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
    # Windows并行复制大矩阵容易耗尽内存；输出维度顺序拟合更稳定且可复现。
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(X, y)
    return model


@dataclass(frozen=True)
class GRUTaskSpec:
    task: str
    target_key: str
    price_prefix: str
    target_context_dim: int


GRU_TASKS = {
    "da": GRUTaskSpec("da", "price_da_tgt", "price_da", 8),
    "rt_at_da": GRUTaskSpec(
        "rt_at_da", "price_rt_at_da_tgt", "price_rt", 8
    ),
    "rt": GRUTaskSpec("rt", "price_rt_tgt", "price_rt", 9),
}


class JointGRUBaseline(nn.Module):
    """同一骨架、三个独立实例的GRU价格基线。"""

    def __init__(self, task: str, hidden_size: int = 64):
        super().__init__()
        if task not in GRU_TASKS:
            raise ValueError(f"未知任务: {task}")
        self.spec = GRU_TASKS[task]
        # 两条价格(2)+负荷(1)+风光(2)+日历(8)=13。
        self.gru = nn.GRU(13, hidden_size, batch_first=True)
        self.target_proj = nn.Linear(self.spec.target_context_dim, hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def history_tensor(batch: dict) -> torch.Tensor:
        return torch.cat([
            batch["price_da_ctx"].unsqueeze(-1),
            batch["price_rt_ctx"].unsqueeze(-1),
            batch["load_ctx"],
            batch["system_ctx"],
            batch["cal_ctx"],
        ], dim=-1)

    def target_context(self, batch: dict) -> torch.Tensor:
        if self.spec.task == "rt":
            return torch.cat([
                batch["price_da_tgt_known"].unsqueeze(-1),
                batch["cal_tgt"],
            ], dim=-1)
        return batch["cal_tgt"]

    def forward(self, batch: dict) -> torch.Tensor:
        _, hidden = self.gru(self.history_tensor(batch))
        query = hidden[-1].unsqueeze(1) + self.target_proj(
            self.target_context(batch)
        )
        return self.head(query).squeeze(-1)

    def normalized_target(self, batch: dict) -> torch.Tensor:
        mean = batch[f"{self.spec.price_prefix}_mean"].unsqueeze(-1)
        std = batch[f"{self.spec.price_prefix}_std"].unsqueeze(-1)
        return (batch[self.spec.target_key] - mean) / std

    def to_price(self, normalized: torch.Tensor, batch: dict) -> torch.Tensor:
        mean = batch[f"{self.spec.price_prefix}_mean"].unsqueeze(-1)
        std = batch[f"{self.spec.price_prefix}_std"].unsqueeze(-1)
        return normalized * std + mean

