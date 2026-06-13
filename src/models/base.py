"""
Forecaster 统一接口 base.py
============================
把"训练 / 零样本""单 / 多变量""有 / 无协变量"全部藏到同一个协议之后。
所有模型——无论是 Naive 这种零样本基线，还是 TimesFM/Chronos/Toto 这种
基础模型，亦或 RF/LSTM 这种需训练模型——对外都长一个样子：

    forecaster.predict(context_df, future_covariates, horizon) -> Forecast

这样滚动回测引擎(backtest.py)不需要关心模型内部差异，只管在每个起报点
喂历史、收预测。模型之间唯一的差异来自"旋钮"，而非散落各处的硬编码。

设计要点
--------
1. 三个能力开关 needs_training / supports_covariates / supports_multivariate，
   让回测引擎可以做"自动降级"并在结果中显式标记（手册 §3.3 能力诚实原则）。
2. context_df 只包含起报点【之前】的数据 → 从结构上杜绝数据泄露（手册 §1.1）。
3. 输出统一为 Forecast(mean, q10, q50, q90)，让点指标与概率指标都能算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd


# ── 统一预测结果容器 ──────────────────────────────────────────────────────────
@dataclass
class Forecast:
    """
    一次预测的输出。每个字段形状均为 (horizon,) 或 (horizon, n_series)。

    - mean : 点预测（期望）
    - q10/q50/q90 : 分位数预测，用于概率指标与"按 q90 判定尖峰"
      不支持分位数的模型可只填 mean，其余设为 None（指标模块会跳过概率项）。
    - index : 预测对应的时间戳（DatetimeIndex），便于和真值对齐
    - series_names : 多序列时各列对应的节点名；单序列可为 None
    """
    mean: np.ndarray
    q10: Optional[np.ndarray] = None
    q50: Optional[np.ndarray] = None
    q90: Optional[np.ndarray] = None
    index: Optional[pd.DatetimeIndex] = None
    series_names: Optional[List[str]] = None

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float)
        for q in ("q10", "q50", "q90"):
            v = getattr(self, q)
            if v is not None:
                setattr(self, q, np.asarray(v, dtype=float))

    @property
    def horizon(self) -> int:
        return self.mean.shape[0]

    @property
    def has_quantiles(self) -> bool:
        return self.q10 is not None and self.q90 is not None


# ── Forecaster 抽象基类 ───────────────────────────────────────────────────────
class Forecaster:
    """
    所有预测器的统一父类。子类至少要实现 `predict`。

    类属性（能力声明，子类按需覆盖）：
      name                  : 模型名（出现在结果表里）
      needs_training        : True=需要 fit（RF/LSTM…），False=零样本（Naive/TimesFM…）
      supports_covariates   : 是否能吃协变量
      supports_multivariate : 是否原生支持多节点联合建模
    """

    name: str = "Forecaster"
    needs_training: bool = False
    supports_covariates: bool = False
    supports_multivariate: bool = False

    # 目标列前缀（与 loader.py 的 price__<node> 约定一致）
    TARGET_PREFIX = "price__"

    def predict(
        self,
        context_df: pd.DataFrame,
        future_covariates: Optional[pd.DataFrame] = None,
        horizon: int = 24,
    ) -> Forecast:
        """
        参数
        ----
        context_df : 起报点之前的历史，索引为时间戳。
                     包含一个或多个 price__<node> 目标列，可能还含协变量列。
        future_covariates : 预测窗口内的未来协变量（仅协变量列，无目标列）。
                     模型/实验不支持时为 None。
        horizon : 预测步数。

        返回
        ----
        Forecast 对象，mean 等字段长度为 horizon。
        """
        raise NotImplementedError

    # ── 给子类用的小工具 ──────────────────────────────────────────────────────
    def _target_columns(self, df: pd.DataFrame) -> List[str]:
        """从 DataFrame 里挑出目标列（price__ 开头）。"""
        cols = [c for c in df.columns if c.startswith(self.TARGET_PREFIX)]
        if not cols:
            raise ValueError(
                f"context_df 中找不到目标列（应以 {self.TARGET_PREFIX!r} 开头）："
                f"现有列 {df.columns.tolist()}"
            )
        return cols

    def _future_index(self, context_df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
        """根据历史推断未来 horizon 步的时间戳（用历史的频率外推）。"""
        idx = context_df.index
        if len(idx) >= 2:
            step = idx[-1] - idx[-2]
        else:
            step = pd.Timedelta(hours=1)
        start = idx[-1] + step
        return pd.date_range(start=start, periods=horizon, freq=step)

    def __repr__(self):
        flags = []
        if self.needs_training:
            flags.append("train")
        if self.supports_covariates:
            flags.append("cov")
        if self.supports_multivariate:
            flags.append("mv")
        return f"<Forecaster {self.name} [{','.join(flags) or 'zeroshot'}]>"
