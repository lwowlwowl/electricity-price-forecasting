"""新年度协议下的多节点传统基线公共组件。

LEAR部分复现公开算法的Invariant变换、LassoLarsIC选alpha和逐小时Lasso，
但把价格滞后改为严格位于本项目D-1 10:00截止时点之前的四个24小时块。
因此报告名固定为 ``cutoff_safe_lear``，不冒充原始午夜起报实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator
import warnings

import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC
from sklearn.multioutput import MultiOutputRegressor

from .baselines_joint import weekly_seasonal_task_predictions
from .global_data import MultiNodeDataset, RTDeliveryDayDataset


class InvariantMedianScaler:
    """median/MAD标准化后做asinh，与LEAR公开Invariant定义一致。"""

    def fit(self, values: np.ndarray) -> "InvariantMedianScaler":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Invariant scaler只接受二维数组")
        self.median_ = np.median(values, axis=0)
        self.mad_ = median_abs_deviation(
            values, axis=0, scale="normal", nan_policy="raise"
        )
        self.mad_ = np.where(self.mad_ > 1e-12, self.mad_, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "median_"):
            raise ValueError("Invariant scaler尚未fit")
        values = np.asarray(values, dtype=np.float64)
        return np.arcsinh((values - self.median_) / self.mad_)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "median_"):
            raise ValueError("Invariant scaler尚未fit")
        values = np.asarray(values, dtype=np.float64)
        return np.sinh(values) * self.mad_ + self.median_


class CutoffSafeLEAR24:
    """固定截止时点适配的24输出LEAR估计器。"""

    def __init__(self, max_iter: int = 2500):
        self.max_iter = int(max_iter)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "CutoffSafeLEAR24":
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if features.ndim != 2 or targets.ndim != 2 or targets.shape[1] != 24:
            raise ValueError("LEAR要求X为二维、Y为[n,24]")
        if len(features) != len(targets):
            raise ValueError("LEAR的X/Y样本数不一致")
        if features.shape[1] < 8:
            raise ValueError("LEAR特征必须以7个星期哑变量结尾")
        if len(features) <= features.shape[1] + 1:
            raise ValueError("LassoLarsIC要求样本数明显多于特征数")

        self.feature_scaler_ = InvariantMedianScaler()
        self.target_scaler_ = InvariantMedianScaler()
        scaled_x = features.copy()
        scaled_x[:, :-7] = self.feature_scaler_.fit_transform(features[:, :-7])
        scaled_y = self.target_scaler_.fit_transform(targets)
        self.models_: list[Lasso] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for hour in range(24):
                selector = LassoLarsIC(criterion="aic", max_iter=self.max_iter)
                alpha = float(selector.fit(scaled_x, scaled_y[:, hour]).alpha_)
                model = Lasso(alpha=alpha, max_iter=self.max_iter)
                model.fit(scaled_x, scaled_y[:, hour])
                self.models_.append(model)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise ValueError("LEAR尚未fit")
        features = np.asarray(features, dtype=np.float64)
        scaled_x = features.copy()
        scaled_x[:, :-7] = self.feature_scaler_.transform(features[:, :-7])
        scaled_prediction = np.column_stack([
            model.predict(scaled_x) for model in self.models_
        ])
        return self.target_scaler_.inverse_transform(scaled_prediction).astype(
            np.float32
        )


def cutoff_safe_lear_arrays(dataset: Any, task: str) -> tuple[np.ndarray, np.ndarray]:
    """从单节点DA日数据构造无未来泄漏的103维LEAR X/Y。"""
    if task not in {"da", "rt_at_da"}:
        raise ValueError("cutoff_safe_lear只适用于DA和RT-at-DA两个24小时任务")
    prices = dataset.price_da if task == "da" else dataset.price_rt
    target_key = "price_da_tgt" if task == "da" else "price_rt_at_da_tgt"
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, window in enumerate(dataset.windows):
        end = int(window.context_end)
        blocks = (
            (end - 24, end),
            (end - 48, end - 24),
            (end - 72, end - 48),
            (end - 168, end - 144),
        )
        if any(start < window.context_start or stop > end for start, stop in blocks):
            raise ValueError(f"LEAR样本{index}的价格滞后越过合法上下文")
        price_features = np.concatenate([
            prices[start:stop] for start, stop in blocks
        ]).astype(np.float64)
        dow = np.zeros(7, dtype=np.float64)
        dow[window.delivery_date.weekday()] = 1.0
        features.append(np.concatenate([price_features, dow]))
        targets.append(dataset[index][target_key].detach().cpu().numpy())
    return np.stack(features), np.stack(targets).astype(np.float32)


def iter_individual_samples(dataset: Any, task: str) -> Iterator[dict]:
    """把RT日样本展开成24个小时样本；两个日级任务保持一天一个样本。"""
    if task == "rt":
        if not isinstance(dataset, RTDeliveryDayDataset):
            raise TypeError("RT global基线要求RTDeliveryDayDataset")
        for day_index in range(len(dataset)):
            day = dataset[day_index]
            for hour in range(24):
                yield {key: value[hour] for key, value in day.items()}
        return
    if task not in {"da", "rt_at_da"}:
        raise ValueError(f"未知任务: {task}")
    for index in range(len(dataset)):
        yield dataset[index]


def xgboost_feature(sample: dict, task: str, history_steps: int = 24) -> np.ndarray:
    """只展开合法可见字段；目标价格从不进入XGBoost特征。"""
    history_keys = (
        "price_da_ctx", "price_rt_ctx", "load_ctx", "system_ctx", "cal_ctx"
    )
    values = []
    for key in history_keys:
        array = sample[key].detach().cpu().numpy()
        values.append(array[-history_steps:].reshape(-1))
    if task == "rt":
        values.append(sample["price_da_tgt_known"].detach().cpu().numpy().reshape(-1))
    values.append(sample["cal_tgt"].detach().cpu().numpy().reshape(-1))
    return np.concatenate(values).astype(np.float32)


@dataclass(frozen=True)
class NodeMetrics:
    mae: float
    rmse: float
    samples: int


def prediction_metrics(target: np.ndarray, prediction: np.ndarray) -> NodeMetrics:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError(f"预测/真值形状不一致: {prediction.shape} vs {target.shape}")
    error = prediction - target
    return NodeMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        samples=int(target.shape[0]),
    )


def node_datasets(global_dataset: MultiNodeDataset) -> Iterator[tuple[str, str, Any]]:
    for (market, node), dataset in zip(
        global_dataset.entities, global_dataset.datasets
    ):
        yield market, node, dataset


def weekly_seasonal_node_predictions(
    dataset: Any, task: str
) -> tuple[np.ndarray, np.ndarray]:
    """返回一个节点的周季节预测和真值，RT保持[day,24,H]。"""
    if task == "rt":
        if not isinstance(dataset, RTDeliveryDayDataset):
            raise TypeError("RT seasonal要求RTDeliveryDayDataset")
        hourly_indices = [
            index for _, indices in dataset.days for index in indices
        ]
        prediction = weekly_seasonal_task_predictions(
            dataset.hourly_dataset, "rt", hourly_indices
        ).reshape(len(dataset), 24, -1)
        target = np.stack([
            dataset[index]["price_rt_tgt"].detach().cpu().numpy()
            for index in range(len(dataset))
        ])
        return prediction, target
    target_key = "price_da_tgt" if task == "da" else "price_rt_at_da_tgt"
    prediction = weekly_seasonal_task_predictions(dataset, task)
    target = np.stack([
        dataset[index][target_key].detach().cpu().numpy()
        for index in range(len(dataset))
    ])
    return prediction, target


def rolling_cutoff_safe_lear_predictions(
    train_dataset: Any,
    validation_dataset: Any,
    task: str,
    *,
    calibration_window: int = 1092,
    refit_every_days: int = 1,
    max_validation_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按时间顺序重估LEAR；当天真值只能在下一天之后进入历史。"""
    if refit_every_days < 1:
        raise ValueError("refit_every_days必须至少为1")
    train_x, train_y = cutoff_safe_lear_arrays(train_dataset, task)
    validation_x, validation_y = cutoff_safe_lear_arrays(validation_dataset, task)
    if max_validation_days is not None:
        validation_x = validation_x[:max_validation_days]
        validation_y = validation_y[:max_validation_days]
    history_x = [row.copy() for row in train_x]
    history_y = [row.copy() for row in train_y]
    predictions: list[np.ndarray] = []
    model: CutoffSafeLEAR24 | None = None
    for day_index, (features, realized) in enumerate(
        zip(validation_x, validation_y)
    ):
        if model is None or day_index % refit_every_days == 0:
            fit_x = np.stack(history_x[-calibration_window:])
            fit_y = np.stack(history_y[-calibration_window:])
            model = CutoffSafeLEAR24().fit(fit_x, fit_y)
        predictions.append(model.predict(features[None, :])[0])
        # 严格在当天预测完成后，才允许真实价格进入下一天重估样本。
        history_x.append(features.copy())
        history_y.append(realized.copy())
    return np.stack(predictions), validation_y


def xgboost_arrays(dataset: Any, task: str) -> tuple[np.ndarray, np.ndarray]:
    target_key = {
        "da": "price_da_tgt",
        "rt_at_da": "price_rt_at_da_tgt",
        "rt": "price_rt_tgt",
    }.get(task)
    if target_key is None:
        raise ValueError(f"未知任务: {task}")
    features, targets = [], []
    for sample in iter_individual_samples(dataset, task):
        features.append(xgboost_feature(sample, task))
        targets.append(sample[target_key].detach().cpu().numpy())
    return np.stack(features), np.stack(targets).astype(np.float32)


def fit_xgboost_multioutput(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int = 0,
    n_estimators: int = 180,
    objective: str = "reg:squarederror",
):
    """真实XGBoost；缺包时明确失败，绝不回退为其他算法。"""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "当前解释器未安装xgboost；请按requirements.txt准备环境"
        ) from exc
    base = XGBRegressor(
        objective=objective,
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=seed,
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(features, targets)
    return model
