"""
dataset.py — ElecFM 训练数据集
================================
滑窗 Dataset：把时序电价数据切成 (context, target, spike_labels) 三元组。

实际可用数据范围：2025-01-01 ~ 2026-06-02（共 ~17 个月）

完整时间划分（无重叠、无缝隙，对应 Lago 2021 checklist #9 #10 #12）：
  train      : 2025-01-01 ~ 2025-05-31（5 个月，spike head 训练期）
  validation : 2025-06-01 ~ 2025-06-30（1 个月，早停 + 超参调优）
  test       : 2025-07-01 ~ 2026-06-01（12 个月连续，评估期）

切割逻辑（以 target 窗口终点为准）：
  train   → target 窗口终点 ≤ TRAIN_END
  val     → VAL_START ≤ target 窗口起点 且 target 窗口终点 ≤ VAL_END
  test    → TEST_START ≤ target 窗口起点 且 target 窗口终点 ≤ TEST_END

context 窗口可向前延伸到数据起点（不受划分约束），从结构上保证：
  ─ train/val/test 无重叠                         (checklist #9 #12)
  ─ test 是数据最后一段，无数据泄露               (checklist #12)
  ─ 不再依赖 "排除范围" 列表，逻辑简洁无歧义     (checklist #10)

单变量设计：每个节点独立构成样本，不做节点间联合建模（基准版本）。
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

# ── 时间划分常量（UTC，含边界）────────────────────────────────────────────────
# 修改这里的日期即可全局调整划分（无需改动其他任何代码）。
TRAIN_START = "2025-01-01"
TRAIN_END   = "2025-05-31 23:00"   # 含 5 月 31 日最后一小时
VAL_START   = "2025-06-01 00:00"
VAL_END     = "2025-06-30 23:00"   # 含 6 月 30 日最后一小时
TEST_START  = "2025-07-01 00:00"
TEST_END    = "2026-06-01 23:00"   # 含 6 月 1 日最后一小时

_SPLIT_BOUNDS = {
    "train": (TRAIN_START, TRAIN_END),
    "val":   (VAL_START,   VAL_END),
    "test":  (TEST_START,  TEST_END),
}


def _ts(s: str) -> pd.Timestamp:
    """将字符串转为 UTC Timestamp。"""
    return pd.Timestamp(s, tz="UTC")


class ElecFMDataset(Dataset):
    """
    单节点滑窗数据集。

    每条样本：
      context  : np.float32 [context_len]  — 起报点前的历史电价
      target   : np.float32 [horizon]      — 起报点后的真实电价
      spike_lb : np.float32 [horizon]      — 尖峰二值标签（target > threshold → 1.0）

    参数
    ----
    price_series    : 单节点电价序列（DatetimeIndex，任意时区均可，内部转 UTC）
    context_len     : 回看窗口（默认 168h = 1 周）
    horizon         : 预测步长（默认 24h = 1 天）
    spike_threshold : P95 尖峰阈值；None 时用 train 分段数据自动计算
    split           : "train" | "val" | "test"
    stride          : 滑窗步长（训练默认 1，评估可设 24 加速）
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
        assert split in _SPLIT_BOUNDS, \
            f"split 必须是 {list(_SPLIT_BOUNDS)!r}，收到 {split!r}"
        self.context_len = context_len
        self.horizon = horizon
        self.split = split

        # ── 统一时区为 UTC ────────────────────────────────────────────────────
        series = price_series.dropna().sort_index()
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC")
        else:
            series.index = series.index.tz_convert("UTC")

        self.values = series.to_numpy(dtype=np.float32)
        self.index  = series.index

        # ── P95 尖峰阈值：仅用 train 分段计算，避免测试期信息泄露 ──────────────
        if spike_threshold is not None:
            self.threshold = float(spike_threshold)
        else:
            train_mask = (series.index <= _ts(TRAIN_END))
            train_vals = series[train_mask].values
            self.threshold = float(np.nanquantile(train_vals, 0.95)) \
                if train_vals.size else 0.0

        # ── 划分边界 ──────────────────────────────────────────────────────────
        lo_str, hi_str = _SPLIT_BOUNDS[split]
        split_lo = _ts(lo_str)
        split_hi = _ts(hi_str)

        # ── 构建合法起点列表（以 target 窗口终点落在 [split_lo, split_hi] 为准）
        self.valid_starts: List[int] = []
        n = len(self.values)
        for i in range(0, n - context_len - horizon + 1, stride):
            tgt_start = self.index[i + context_len]
            tgt_end   = self.index[i + context_len + horizon - 1]
            if tgt_start >= split_lo and tgt_end <= split_hi:
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


# ── 节点配置（保留向后兼容） ─────────────────────────────────────────────────
# 消融实验代表节点（ERCOT 高波动组）
V6_NODES = ["LZ_LCRA", "LZ_WEST", "LZ_RAYBN"]

# ERCOT 15 节点分组（5组×3节点），用于全节点训练
V6_NODE_GROUPS = [
    ["LZ_LCRA",   "LZ_WEST",    "LZ_RAYBN"],   # G1: 高波动（主测试节点）
    ["HB_BUSAVG", "HB_HUBAVG",  "HB_HOUSTON"], # G2: Hub 均价节点
    ["HB_NORTH",  "HB_PAN",     "HB_SOUTH"],   # G3: Hub 地理分区
    ["HB_WEST",   "LZ_AEN",     "LZ_CPS"],     # G4: 西部 Hub + AEN/CPS 负荷区
    ["LZ_HOUSTON","LZ_NORTH",   "LZ_SOUTH"],   # G5: 剩余负荷区
]


# ── 便捷构建函数 ──────────────────────────────────────────────────────────────
class _Dataset3Node(torch.utils.data.Dataset):
    """将 3 个单节点数据集 zip 成联合数据集（内部使用）。"""

    def __init__(self, datasets: List[ElecFMDataset]):
        assert len(datasets) == 3, "需要恰好 3 个节点数据集"
        self.datasets = datasets

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int):
        items = [d[idx] for d in self.datasets]
        ctx      = torch.stack([it[0] for it in items])   # [3, context_len]
        tgt      = torch.stack([it[1] for it in items])   # [3, horizon]
        spike_lb = torch.stack([it[2] for it in items])   # [3, horizon]
        return ctx, tgt, spike_lb

    @property
    def spike_pos_weight(self) -> float:
        return float(sum(d.spike_pos_weight for d in self.datasets) / len(self.datasets))


def build_datasets_v6(
    market: str,
    context_len: int = 168,
    horizon: int = 24,
    stride: int = 1,
) -> Tuple["_Dataset3Node", "_Dataset3Node"]:
    """
    V6 专用：V6_NODES（LZ_LCRA / LZ_WEST / LZ_RAYBN）3 节点联合同步数据集。

    train : target 在 2025-01-01 ~ 2025-05-31（5 个月）
    val   : target 在 2025-06-01 ~ 2025-06-30（1 个月）

    返回 (train_ds, val_ds)，每条样本形状 [3, context_len / horizon]。
    """
    import loader  # noqa: F401

    node_trains, node_vals = [], []
    for node in V6_NODES:
        df = loader.load_slice(
            market=market, nodes=[node], freq="1h",
            start=TRAIN_START, end=TEST_END,
        )
        col    = f"price__{node}"
        series = df[col].dropna()

        tr = ElecFMDataset(series, context_len, horizon, split="train", stride=stride)
        va = ElecFMDataset(series, context_len, horizon,
                           spike_threshold=tr.threshold, split="val", stride=1)

        node_trains.append(tr)
        node_vals.append(va)
        print(f"  {node}: train={len(tr)}  val={len(va)}  threshold={tr.threshold:.2f}")

    assert len(node_trains[0]) == len(node_trains[1]) == len(node_trains[2]), \
        f"三节点训练样本数不一致：{[len(d) for d in node_trains]}"

    return _Dataset3Node(node_trains), _Dataset3Node(node_vals)


def build_datasets_v6_allgroups(
    market: str,
    context_len: int = 168,
    horizon: int = 24,
    stride: int = 1,
) -> Tuple[torch.utils.data.ConcatDataset, "_Dataset3Node"]:
    """
    V6 全节点版：5 组 × 3 节点，训练数据量是单组的 5 倍。

    训练集：5 组数据合并（ConcatDataset）
    验证集：仅用第一组（V6_NODES）
    """
    import loader  # noqa: F401

    all_train_datasets = []
    for g_idx, group in enumerate(V6_NODE_GROUPS):
        node_trains = []
        for node in group:
            df = loader.load_slice(
                market=market, nodes=[node], freq="1h",
                start=TRAIN_START, end=TEST_END,
            )
            col    = f"price__{node}"
            series = df[col].dropna()
            tr = ElecFMDataset(series, context_len, horizon, split="train", stride=stride)
            node_trains.append(tr)

        group_ds = _Dataset3Node(node_trains)
        all_train_datasets.append(group_ds)
        label = "+".join(group)
        print(f"  G{g_idx+1}({label}): train={len(group_ds)}")

    # 验证集只用第一组
    _, val_ds = build_datasets_v6(market, context_len, horizon, stride=1)

    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(all_train_datasets)
    print(f"  总训练样本：{len(train_ds)}（{len(V6_NODE_GROUPS)} 组合并）")
    print(f"  验证样本（G1）：{len(val_ds)}")
    return train_ds, val_ds


def build_datasets(
    market: str,
    nodes: List[str],
    context_len: int = 168,
    horizon: int = 24,
    stride: int = 1,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    便捷函数：加载多个节点并合并成训练 / 验证集（单变量设计）。

    每个节点各贡献独立的滑窗样本；各节点独立计算 P95 阈值。
    """
    import loader  # noqa: F401

    train_datasets: List[ElecFMDataset] = []
    val_datasets:   List[ElecFMDataset] = []

    for node in nodes:
        df = loader.load_slice(
            market=market, nodes=[node], freq="1h",
            start=TRAIN_START, end=TEST_END,
        )
        col    = f"price__{node}"
        series = df[col].dropna()

        tr = ElecFMDataset(series, context_len, horizon, split="train", stride=stride)
        va = ElecFMDataset(series, context_len, horizon,
                           spike_threshold=tr.threshold, split="val", stride=1)

        train_datasets.append(tr)
        val_datasets.append(va)
        print(f"  {node}: train={len(tr)}  val={len(va)}  threshold={tr.threshold:.2f}")

    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds   = ConcatDataset(val_datasets)   if len(val_datasets)   > 1 else val_datasets[0]
    return train_ds, val_ds


# ── 自测 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
    try:
        import loader  # noqa: F401
    except ImportError:
        print("loader 不可用，生成合成数据进行测试")
        loader = None

    print("=" * 60)
    print("dataset.py 自测（合成数据）")
    print("=" * 60)

    # 构造覆盖全部时段的合成序列
    idx = pd.date_range("2025-01-01", "2026-06-02", freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    prices = pd.Series(rng.normal(40, 15, len(idx)), index=idx)

    for split in ("train", "val", "test"):
        ds = ElecFMDataset(prices, context_len=168, horizon=24, split=split, stride=24)
        lo, hi = _SPLIT_BOUNDS[split]
        print(f"  {split:5s}: {len(ds):5d} 样本  "
              f"(target 覆盖 {lo} ~ {hi})")
        if len(ds):
            ctx, tgt, sp = ds[0]
            assert ctx.shape == (168,), f"context shape 错误: {ctx.shape}"
            assert tgt.shape == (24,),  f"target shape 错误: {tgt.shape}"
            assert sp.shape  == (24,),  f"spike_lb shape 错误: {sp.shape}"

    # 验证无重叠
    def _tgt_indices(ds):
        out = set()
        for k in range(len(ds)):
            i = ds.valid_starts[k]
            for h in range(ds.horizon):
                out.add(ds.index[i + ds.context_len + h])
        return out

    tr_ds  = ElecFMDataset(prices, split="train", stride=1)
    val_ds = ElecFMDataset(prices, split="val",   stride=1)
    te_ds  = ElecFMDataset(prices, split="test",  stride=1)

    tr_ts  = _tgt_indices(tr_ds)
    val_ts = _tgt_indices(val_ds)
    te_ts  = _tgt_indices(te_ds)

    assert not (tr_ts & val_ts),  "❌ train/val target 有重叠！"
    assert not (tr_ts & te_ts),   "❌ train/test target 有重叠！"
    assert not (val_ts & te_ts),  "❌ val/test target 有重叠！"

    print(f"\n  threshold(train P95) = {tr_ds.threshold:.2f}")
    print(f"  spike_pos_weight     = {tr_ds.spike_pos_weight:.2f}")
    print("\n✅ dataset.py 工作正常（无重叠，划分正确）")
