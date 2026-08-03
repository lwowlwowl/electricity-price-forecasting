"""dataset_v3.py — v3 数据集（6.5年 ERCOT 统一表，DA+RT 双价，真节假日）.

与 dataset.py 区别：
- 读 ercot_unified_hourly_2020_2026.csv（loader_v2.py）
- 4 流：Price(DA+RT)/Load/System(wind,solar)/Calendar(hour,dow,weekend,holiday)
- 返回 price_da + price_rt 两条目标（真双结算）
- 真节假日 is_holiday（不再用 is_weekend 代理）
- 归一化 stats 按 train 段算
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
sys.path.insert(0, os.path.join(_ROOT, "src", "decision_aware"))

from .config import PilotConfig
from .loader_v2 import load_ercot_unified


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# v3 流分组
STREAMS_V3 = {
    "price": [],     # price_da, price_rt 特殊处理（目标，不进 encoder 而是作为 target）
    "load":   ["load"],
    "system": ["wind", "solar"],
    "cal":    ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"],
}


class DecisionAwareDatasetV3(Dataset):
    """v3 多流滑窗数据集（ERCOT 6.5年 DA+RT 双价）。

    每条样本（dict）：
      price_da_ctx  : [context_len]  归一化 DA 价历史
      price_rt_ctx  : [context_len]  归一化 RT 价历史
      price_da_tgt  : [horizon]      真实 DA 价（结算用，不归一化）
      price_rt_tgt  : [horizon]      真实 RT 价（结算用）
      price_da_mean/std, price_rt_mean/std : float（反归一化用）
      load_ctx      : [context_len, 1]
      system_ctx    : [context_len, 2]
      cal_ctx       : [context_len, 6]
    """

    def __init__(self, wide_df: pd.DataFrame, cfg: PilotConfig, split: str,
                 norm_stats: Optional[Dict] = None, stride: Optional[int] = None):
        assert split in ("train", "val", "test")
        self.cfg = cfg
        self.split = split
        self.context_len = cfg.context_len
        self.horizon = cfg.horizon_da

        df = wide_df.sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        self.price_da = df["price_da"].to_numpy(dtype=np.float32)
        self.price_rt = df["price_rt"].to_numpy(dtype=np.float32)
        self.index = df.index

        # 各流
        self.streams: Dict[str, np.ndarray] = {}
        for stream, cols in STREAMS_V3.items():
            if stream == "price":
                continue
            have = [c for c in cols if c in df.columns]
            if not have:
                raise ValueError(f"流 {stream} 的列 {cols} 在 wide_df 中全部缺失")
            self.streams[stream] = df[have].to_numpy(dtype=np.float32)

        # 归一化 stats（仅 train 段计算）
        tr_lo, tr_hi = cfg.split_bounds("train")
        train_lo, train_hi = _ts(tr_lo), _ts(tr_hi)
        train_mask = (self.index >= train_lo) & (self.index <= train_hi)

        if norm_stats is None:
            assert split == "train"
            norm_stats = {}
            for key in ("price_da", "price_rt"):
                arr = self.price_da if key == "price_da" else self.price_rt
                norm_stats[key] = {"mean": float(arr[train_mask].mean()),
                                   "std": float(arr[train_mask].std() + 1e-8)}
            for name, arr in self.streams.items():
                norm_stats[name] = {"mean": arr[train_mask].mean(axis=0),
                                    "std": arr[train_mask].std(axis=0) + 1e-8}
        self.norm_stats = norm_stats

        # 合法起点
        lo, hi = cfg.split_bounds(split)
        split_lo, split_hi = _ts(lo), _ts(hi)
        n = len(self.price_da)
        stride = stride or (cfg.train_stride if split == "train" else cfg.eval_stride)
        self.valid_starts: List[int] = []
        for i in range(0, n - self.context_len - self.horizon + 1, stride):
            tgt_start = self.index[i + self.context_len]
            tgt_end = self.index[i + self.context_len + self.horizon - 1]
            if tgt_start >= split_lo and tgt_end <= split_hi:
                self.valid_starts.append(i)

    def _norm(self, arr, st):
        return ((arr - st["mean"]) / st["std"]).astype(np.float32)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        i = self.valid_starts[idx]
        c_lo, c_hi = i, i + self.context_len
        t_lo, t_hi = c_hi, c_hi + self.horizon

        da_st = self.norm_stats["price_da"]
        rt_st = self.norm_stats["price_rt"]
        sample = {
            "price_da_ctx": torch.from_numpy(self._norm(self.price_da[c_lo:c_hi], da_st)),
            "price_rt_ctx": torch.from_numpy(self._norm(self.price_rt[c_lo:c_hi], rt_st)),
            "price_da_tgt": torch.from_numpy(self.price_da[t_lo:t_hi].copy()),
            "price_rt_tgt": torch.from_numpy(self.price_rt[t_lo:t_hi].copy()),
            "price_da_mean": torch.tensor(da_st["mean"], dtype=torch.float32),
            "price_da_std": torch.tensor(da_st["std"], dtype=torch.float32),
            "price_rt_mean": torch.tensor(rt_st["mean"], dtype=torch.float32),
            "price_rt_std": torch.tensor(rt_st["std"], dtype=torch.float32),
        }
        for name, arr in self.streams.items():
            st = self.norm_stats[name]
            sample[f"{name}_ctx"] = torch.from_numpy(self._norm(arr[c_lo:c_hi], st))
        return sample


def build_datasets_v3(cfg: PilotConfig):
    """返回 (train_ds, val_ds, test_ds, wide_df)。"""
    wide_df = load_ercot_unified(node=cfg.node, start="2020-01-01", end="2026-06-02")
    print(f"  [data v3] wide_df shape={wide_df.shape}  cols={wide_df.columns.tolist()}")
    print(f"  [data v3] time {wide_df.index.min()} → {wide_df.index.max()}  n={len(wide_df)}")

    train_ds = DecisionAwareDatasetV3(wide_df, cfg, split="train", stride=cfg.train_stride)
    stats = train_ds.norm_stats
    val_ds = DecisionAwareDatasetV3(wide_df, cfg, split="val", norm_stats=stats, stride=cfg.eval_stride)
    test_ds = DecisionAwareDatasetV3(wide_df, cfg, split="test", norm_stats=stats, stride=cfg.eval_stride)
    print(f"  [data v3] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"  [data v3] DA price train mean={stats['price_da']['mean']:.2f} std={stats['price_da']['std']:.2f}")
    print(f"  [data v3] RT price train mean={stats['price_rt']['mean']:.2f} std={stats['price_rt']['std']:.2f}")
    return train_ds, val_ds, test_ds, wide_df


def collate_v3(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
