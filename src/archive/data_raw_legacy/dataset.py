"""dataset.py — 先行版多模态滑窗数据集.

复用 src/data_processing/loader.py::load_slice_model_ready(forecast=True) 取 price +
全协变量（model_ready 已按政策B滞后/日前预报，无泄漏）。比归档 ElecFM 用的
load_slice（实测协变量，forecast 窗口会泄漏）更干净。

6 流分组（与 config.STREAM_COLS 一致）：
  Price (target) / Load / Weather / System / Econ / Calendar(由 timestamp 派生)

划分镜像 src/archive/fusion_model/dataset.py：以 target 窗口终点落在 split 区间为准，
train/val/test 无重叠、test 在数据末段（Lago 2021 checklist #9 #12）。

归一化：price 与各连续流按【train 段】均值/标准差 z-score（stats 由 build_datasets
在 train 上算好后传给 val/test，杜绝泄漏）。Calendar 用 sin/cos + 0/1 不归一化。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 让 `import loader` 可用（与归档 dataset.py 同套路）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))

from .config import PilotConfig, STREAM_COLS_V12 as STREAM_COLS, ALL_COVARIATES_V12 as ALL_COVARIATES  # noqa: E402


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _calendar_features(idx: pd.DatetimeIndex) -> np.ndarray:
    """hour/dow 的 sin-cos + 是否周末 + 是否节假日标志 → [T, 5]。

    节假日用美国联邦节假日粗略估计（无第三方依赖，pandas holiday 逻辑）。
    先行版足够；正式版可换真实节假日表。
    """
    hour = idx.hour.values
    dow = idx.dayofweek.values
    feats = np.stack([
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0),  np.cos(2 * np.pi * dow / 7.0),
        (dow >= 5).astype(np.float32),          # 周末标志（节假日代理）
    ], axis=-1).astype(np.float32)
    return feats


class DecisionAwareDataset(Dataset):
    """多流滑窗数据集。

    每条样本（dict）：
      price_ctx   : [context_len]            归一化历史电价
      price_tgt   : [horizon]                真实电价（regret/loss 用真实尺度）
      load_ctx     : [context_len, 1]
      weather_ctx  : [context_len, 1]
      system_ctx   : [context_len, n_sys]
      econ_ctx     : [context_len, n_econ]
      cal_ctx      : [context_len, 5]
      price_mean   : float  / price_std : float   （de-norm 用，每样本同值）
    """

    def __init__(
        self,
        wide_df: pd.DataFrame,
        cfg: PilotConfig,
        split: str,
        norm_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        stride: Optional[int] = None,
    ):
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

        price_col = cfg.price_col
        assert price_col in df.columns, f"{price_col} 不在 wide_df，列={list(df.columns)}"

        # ── 抽各流（缺失列跳过，与 loader 的 have 过滤一致） ──────────────────
        self.streams: Dict[str, np.ndarray] = {}
        for stream, cols in STREAM_COLS.items():
            have = [c for c in cols if c in df.columns]
            if not have:
                raise ValueError(f"流 {stream} 的列 {cols} 在 wide_df 中全部缺失")
            self.streams[stream] = df[have].to_numpy(dtype=np.float32)  # [T, n_feat]

        self.price = df[price_col].to_numpy(dtype=np.float32)          # [T]
        self.cal = _calendar_features(df.index)                        # [T, 5]
        self.index = df.index

        # ── 归一化 stats（price + 各连续流；仅 train 段计算） ─────────────────
        tr_lo, tr_hi = cfg.split_bounds("train")
        train_lo, train_hi = _ts(tr_lo), _ts(tr_hi)
        train_mask = (self.index >= train_lo) & (self.index <= train_hi)

        if norm_stats is None:
            assert split == "train", "val/test 必须传入 train 的 norm_stats"
            norm_stats = self._compute_stats(train_mask)
        self.norm_stats = norm_stats

        # ── 合法起点（target 窗口终点落在 split 区间） ───────────────────────
        lo, hi = cfg.split_bounds(split)
        split_lo, split_hi = _ts(lo), _ts(hi)
        n = len(self.price)
        stride = stride or (cfg.train_stride if split == "train" else cfg.eval_stride)
        self.valid_starts: List[int] = []
        for i in range(0, n - self.context_len - self.horizon + 1, stride):
            tgt_start = self.index[i + self.context_len]
            tgt_end   = self.index[i + self.context_len + self.horizon - 1]
            if tgt_start >= split_lo and tgt_end <= split_hi:
                self.valid_starts.append(i)

    # ── train 段每流 mean/std（price 单独存标量，其余按列存） ──────────────────
    def _compute_stats(self, train_mask: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        stats: Dict[str, Dict[str, np.ndarray]] = {}
        stats["price"] = {
            "mean": float(self.price[train_mask].mean()),
            "std":  float(self.price[train_mask].std() + 1e-8),
        }
        for name, arr in self.streams.items():
            m = arr[train_mask].mean(axis=0)      # [n_feat]
            s = arr[train_mask].std(axis=0) + 1e-8
            stats[name] = {"mean": m, "std": s}
        return stats

    def _norm(self, arr: np.ndarray, st: Dict[str, np.ndarray]) -> np.ndarray:
        return ((arr - st["mean"]) / st["std"]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.valid_starts[idx]
        c_lo, c_hi = i, i + self.context_len
        t_lo, t_hi = c_hi, c_hi + self.horizon

        pst = self.norm_stats["price"]
        sample = {
            "price_ctx":  torch.from_numpy(self._norm(self.price[c_lo:c_hi], pst)),
            "price_tgt":  torch.from_numpy(self.price[t_lo:t_hi].copy()),   # 真实尺度
            "price_mean":  torch.tensor(pst["mean"], dtype=torch.float32),
            "price_std":   torch.tensor(pst["std"],  dtype=torch.float32),
            "cal_ctx":     torch.from_numpy(self.cal[c_lo:c_hi].copy()),
        }
        for name, arr in self.streams.items():
            st = self.norm_stats[name]
            sample[f"{name}_ctx"] = torch.from_numpy(self._norm(arr[c_lo:c_hi], st))
        return sample


# ── 便捷构建：train/val/test 三 split + 共享 train norm_stats ──────────────────
def build_datasets(cfg: PilotConfig):
    """返回 (train_ds, val_ds, test_ds, raw_wide_df)。"""
    import loader  # noqa: F401  (确保 src/data_processing 在 path)

    df = loader.load_slice_model_ready(
        market=cfg.market, nodes=[cfg.node], freq=cfg.freq,
        covariates=ALL_COVARIATES,
        start="2025-01-01", end="2026-06-02",
        forecast=True, dropna=True,
    )
    print(f"  [data] wide_df shape={df.shape}  cols={df.columns.tolist()}")
    print(f"  [data] time {df.index.min()} → {df.index.max()}  n={len(df)}")

    train_ds = DecisionAwareDataset(df, cfg, split="train", stride=cfg.train_stride)
    stats = train_ds.norm_stats
    val_ds = DecisionAwareDataset(df, cfg, split="val", norm_stats=stats, stride=cfg.eval_stride)
    test_ds = DecisionAwareDataset(df, cfg, split="test", norm_stats=stats, stride=cfg.eval_stride)
    print(f"  [data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
          f"(stride train={cfg.train_stride} eval={cfg.eval_stride})")
    print(f"  [data] price train mean={stats['price']['mean']:.2f} "
          f"std={stats['price']['std']:.2f}")
    return train_ds, val_ds, test_ds, df


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """把 dict 样本列表 stack 成 batched dict。"""
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


if __name__ == "__main__":
    # 冒烟测试：构建三 split，取一条样本，打印形状 + 验证无重叠
    cfg = PilotConfig()
    cfg.node = "LZ_LCRA"
    train_ds, val_ds, test_ds, _ = build_datasets(cfg)
    s = train_ds[0]
    print("\n样本字段形状：")
    for k, v in s.items():
        print(f"  {k:12s} {tuple(v.shape) if hasattr(v,'shape') else v}")
