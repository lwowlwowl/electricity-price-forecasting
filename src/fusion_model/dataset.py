"""
dataset.py — ElecFM 训练数据集
================================
滑窗 Dataset：把 ERCOT 时序数据切成 (context, target, spike_labels) 三元组。

实际可用数据范围：2025-01-01 ~ 2026-06-02（共 ~17 个月）

完整时间划分（无重叠、无缝隙）：
  训练 Seg1  2025-01-01 ~ 2025-02-20   1224h / 1033 windows/node
  排除 W2+buf 2025-02-21 ~ 2025-03-31  （W2 测试窗口 + 168h context buffer）
  训练 Seg2  2025-04-01 ~ 2025-07-24   2760h / 2569 windows/node
  排除 W1+buf 2025-07-25 ~ 2025-08-31  （W1 测试窗口 + 168h context buffer）
  训练 Seg3  2025-09-01 ~ 2025-11-30   2184h / 1993 windows/node
  验证集     2025-12-01 ~ 2025-12-24   576h  /  385 windows/node
  排除 W3+buf 2025-12-25 ~ 2026-01-31  （W3 测试窗口 + 168h context buffer）
  训练 Seg4  2026-02-01 ~ 2026-06-02   2928h / 2737 windows/node
  ─────────────────────────────────────────────────────
  训练合计：~24,996 样本（3 节点 × 8332）
  验证合计：~1,155  样本（3 节点 × 385）

单变量（Q2 决策）：每个节点独立构成样本，不做节点间联合建模。
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── 路径 ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
sys.path.insert(0, os.path.join(_ROOT, "src", "evaluation"))

# ── 时间常量（基于实际数据范围 2025-01-01 ~ 2026-06-02）────────────────────────

# 测试窗口 + 168h context buffer（任何 context 或 target 与此重叠的样本都排除）
EXCLUDED_RANGES = [
    ("2025-02-21", "2025-03-31"),  # W2 (2025-03) + buffer
    ("2025-07-25", "2025-08-31"),  # W1 (2025-08) + buffer
    ("2025-12-25", "2026-01-31"),  # W3 (2026-01) + buffer
]

# 验证集独立时段（从训练集中独立切出，不参与训练）
VAL_START = "2025-12-01"
VAL_END   = "2025-12-24"   # 留出 25-31 给 W3 buffer


def _is_excluded_from_train(context_start: pd.Timestamp,
                             target_end: pd.Timestamp) -> bool:
    """若样本的 context 或 target 与测试窗口 / 验证集有任何重叠，返回 True。"""
    # 检查与测试窗口重叠
    for excl_start, excl_end in EXCLUDED_RANGES:
        es = pd.Timestamp(excl_start, tz="UTC")
        ee = pd.Timestamp(excl_end + " 23:59", tz="UTC")
        if not (target_end < es or context_start > ee):
            return True
    # 检查与验证集重叠（训练集不能包含 Dec 1-24 的数据）
    vs = pd.Timestamp(VAL_START, tz="UTC")
    ve = pd.Timestamp(VAL_END + " 23:59", tz="UTC")
    if not (target_end < vs or context_start > ve):
        return True
    return False


def _is_in_val(context_start: pd.Timestamp, target_end: pd.Timestamp) -> bool:
    """判断样本是否完全位于验证集时段内（context 和 target 都在 Dec 1-24）。"""
    vs = pd.Timestamp(VAL_START, tz="UTC")
    ve = pd.Timestamp(VAL_END + " 23:59", tz="UTC")
    return context_start >= vs and target_end <= ve


class ElecFMDataset(Dataset):
    """
    单节点滑窗数据集。

    每条样本：
      context  : np.float32 [context_len]  — 起报点前的历史电价
      target   : np.float32 [horizon]      — 起报点后的真实电价
      spike_lb : np.float32 [horizon]      — 尖峰二值标签（target > threshold → 1.0）

    参数
    ----
    price_series    : 单节点电价序列（DatetimeIndex, UTC）
    context_len     : 回看窗口（默认 168h = 1 周）
    horizon         : 预测步长（默认 24h）
    spike_threshold : P95 尖峰阈值；None 时内部自动计算
    split           : "train" | "val"
    stride          : 滑窗步长（训练默认 1，验证默认 1）
    """

    def __init__(
        self,
        price_series: pd.Series,
        context_len: int = 168,
        horizon: int = 24,
        spike_threshold: Optional[float] = None,
        split: str = "train",
        stride: int = 1,
    ):
        assert split in ("train", "val"), f"split 必须是 train 或 val，收到 {split!r}"
        self.context_len = context_len
        self.horizon = horizon
        self.split = split

        # UTC 索引
        series = price_series.dropna().sort_index()
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC")
        else:
            series.index = series.index.tz_convert("UTC")

        self.values = series.to_numpy(dtype=np.float32)
        self.index  = series.index

        # P95 尖峰阈值：用训练段数据计算（不含测试窗口和验证集）
        if spike_threshold is not None:
            self.threshold = float(spike_threshold)
        else:
            train_mask = pd.Series(True, index=series.index)
            for excl_s, excl_e in EXCLUDED_RANGES:
                es = pd.Timestamp(excl_s, tz="UTC")
                ee = pd.Timestamp(excl_e + " 23:59", tz="UTC")
                train_mask &= ~((series.index >= es) & (series.index <= ee))
            train_mask &= ~((series.index >= pd.Timestamp(VAL_START, tz="UTC")) &
                            (series.index <= pd.Timestamp(VAL_END + " 23:59", tz="UTC")))
            self.threshold = float(np.nanquantile(series[train_mask].values, 0.95))

        # 构建合法起点列表
        self.valid_starts: List[int] = []
        n = len(self.values)
        for i in range(0, n - context_len - horizon + 1, stride):
            ctx_start = self.index[i]
            tgt_end   = self.index[min(i + context_len + horizon - 1, n - 1)]
            if split == "train" and _is_excluded_from_train(ctx_start, tgt_end):
                continue
            if split == "val" and not _is_in_val(ctx_start, tgt_end):
                continue
            self.valid_starts.append(i)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i  = self.valid_starts[idx]
        ctx = self.values[i: i + self.context_len]
        tgt = self.values[i + self.context_len: i + self.context_len + self.horizon]
        spike_lb = (tgt > self.threshold).astype(np.float32)
        return (
            torch.from_numpy(ctx),
            torch.from_numpy(tgt),
            torch.from_numpy(spike_lb),
        )

    @property
    def spike_pos_weight(self) -> float:
        """从本 split 数据统计实际正样本率，用于 BCE pos_weight。"""
        all_tgt = []
        for i in self.valid_starts:
            tgt = self.values[i + self.context_len: i + self.context_len + self.horizon]
            all_tgt.append(tgt)
        if not all_tgt:
            return 19.0
        all_tgt = np.concatenate(all_tgt)
        pos = int(np.sum(all_tgt > self.threshold))
        neg = int(np.sum(all_tgt <= self.threshold))
        return float(neg / pos) if pos > 0 else 19.0


def build_datasets(
    market: str,
    nodes: List[str],
    context_len: int = 168,
    horizon: int = 24,
    stride: int = 1,
) -> Tuple["ElecFMDataset", "ElecFMDataset"]:
    """
    便捷函数：加载多个节点数据并合并成训练/验证集。
    每个节点各贡献独立的滑窗样本（单变量设计）。
    共享阈值（用第一个节点计算，保证一致性）。
    """
    import loader  # src/data_processing/loader.py

    train_datasets = []
    val_datasets   = []
    common_threshold = None

    for node in nodes:
        df = loader.load_slice(
            market=market, nodes=[node], freq="1h",
            start="2025-01-01", end="2026-06-05",   # 全量，dataset 内部按 split 过滤
        )
        col    = f"price__{node}"
        series = df[col].dropna()

        tr = ElecFMDataset(series, context_len, horizon,
                           spike_threshold=common_threshold,
                           split="train", stride=stride)
        if common_threshold is None:
            common_threshold = tr.threshold

        va = ElecFMDataset(series, context_len, horizon,
                           spike_threshold=common_threshold,
                           split="val", stride=1)

        train_datasets.append(tr)
        val_datasets.append(va)
        print(f"  {node}: train={len(tr)}  val={len(va)}  "
              f"threshold={common_threshold:.2f}")

    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds   = ConcatDataset(val_datasets)   if len(val_datasets)   > 1 else val_datasets[0]

    return train_ds, val_ds
