"""按市场日前截止时点构造独立DA日样本。

ERCOT第一版合同：D-1日10:00（当地CPT）截断输入，预测D日当地完整交付日。
这里只负责模型数据，不修改任何Excel或Parquet。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PilotConfig
from .loader_v2 import load_market_unified, market_timezone


STREAMS_DA = {
    "load": ["load"],
    "system": ["wind", "solar"],
    "cal": [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
    ],
}


def _market_ts(value: str, timezone: str) -> pd.Timestamp:
    """把配置中的分割边界解释为市场当地时间，再转换为UTC索引。"""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class DASampleWindow:
    context_start: int
    context_end: int
    target_positions: np.ndarray
    issue_time_utc: pd.Timestamp
    delivery_date: date


class DecisionAwareDADataset(Dataset):
    """固定市场截止时点的24小时DA-only样本。"""

    def __init__(
        self,
        wide_df: pd.DataFrame,
        cfg: PilotConfig,
        split: str,
        norm_stats: Optional[Dict] = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"未知split: {split}")
        if cfg.horizon_da != 24:
            raise ValueError("DA日样本第一版要求 horizon_da=24")

        self.cfg = cfg
        self.split = split
        self.context_len = cfg.context_len
        self.horizon_tgt = 24
        self.timezone = market_timezone(cfg.market)
        self.issue_hour_local = cfg.da_issue_hour_local

        df = wide_df.sort_index().copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        if not df.index.is_unique:
            raise ValueError("DA数据索引存在重复时间戳")

        self.index = df.index
        self.local_index = self.index.tz_convert(self.timezone)
        self.price_da = df["price_da"].to_numpy(dtype=np.float32)
        self.price_rt = df["price_rt"].to_numpy(dtype=np.float32)
        self.streams: Dict[str, np.ndarray] = {}
        for stream, cols in STREAMS_DA.items():
            have = [col for col in cols if col in df.columns]
            if len(have) != len(cols):
                missing = sorted(set(cols) - set(have))
                raise ValueError(f"流 {stream} 缺列: {missing}")
            self.streams[stream] = df[have].to_numpy(dtype=np.float32)

        train_lo, train_hi = (
            _market_ts(value, self.timezone)
            for value in cfg.split_bounds("train")
        )
        train_mask = (self.index >= train_lo) & (self.index <= train_hi)
        if norm_stats is None:
            if split != "train":
                raise ValueError("验证/测试集必须复用训练集norm_stats")
            if not train_mask.any():
                raise ValueError("数据中没有配置所指定的训练时段，无法计算归一化统计量")
            norm_stats = {
                "price_da": self._scalar_stats(self.price_da[train_mask]),
                "price_rt": self._scalar_stats(self.price_rt[train_mask]),
            }
            for name, values in self.streams.items():
                norm_stats[name] = {
                    "mean": values[train_mask].mean(axis=0),
                    "std": values[train_mask].std(axis=0) + 1e-8,
                }
        self.norm_stats = norm_stats

        local_dates = np.asarray(self.local_index.date, dtype=object)
        positions_by_date: Dict[date, List[int]] = {}
        for position, local_date in enumerate(local_dates):
            positions_by_date.setdefault(local_date, []).append(position)

        split_lo, split_hi = (
            _market_ts(value, self.timezone)
            for value in cfg.split_bounds(split)
        )
        one_hour_ns = pd.Timedelta(hours=1).value
        self.windows: List[DASampleWindow] = []
        self.excluded_non_24h_days: List[date] = []
        for delivery_date, positions_list in positions_by_date.items():
            target_positions = np.asarray(positions_list, dtype=np.int64)
            if len(target_positions) != 24:
                self.excluded_non_24h_days.append(delivery_date)
                continue
            target_times = self.index[target_positions]
            if np.any(np.diff(target_times.as_unit("ns").asi8) != one_hour_ns):
                continue
            if target_times[0] < split_lo or target_times[-1] > split_hi:
                continue

            issue_date = delivery_date - timedelta(days=1)
            issue_naive = datetime.combine(issue_date, time(self.issue_hour_local))
            issue_local = pd.Timestamp(issue_naive).tz_localize(self.timezone)
            issue_utc = issue_local.tz_convert("UTC")
            context_end = int(self.index.searchsorted(issue_utc, side="left"))
            context_start = context_end - self.context_len
            if context_start < 0 or context_end >= len(self.index):
                continue
            # 必须存在issue_time这一行；它本身不进入上下文，因为该小时尚未完成。
            if self.index[context_end] != issue_utc:
                continue
            context_times = self.index[context_start:context_end]
            if len(context_times) != self.context_len:
                continue
            if np.any(np.diff(context_times.as_unit("ns").asi8) != one_hour_ns):
                continue

            self.windows.append(
                DASampleWindow(
                    context_start=context_start,
                    context_end=context_end,
                    target_positions=target_positions,
                    issue_time_utc=issue_utc,
                    delivery_date=delivery_date,
                )
            )

    @staticmethod
    def _scalar_stats(values: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(values.mean()),
            "std": float(values.std() + 1e-8),
        }

    @staticmethod
    def _norm(values: np.ndarray, stats: Dict) -> np.ndarray:
        return ((values - stats["mean"]) / stats["std"]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def sample_time_contract(self, idx: int) -> Dict[str, object]:
        window = self.windows[idx]
        target = window.target_positions
        return {
            "market": self.cfg.market,
            "timezone": self.timezone,
            "issue_time": window.issue_time_utc.tz_convert(self.timezone),
            "context_start": self.index[window.context_start].tz_convert(self.timezone),
            "context_end": self.index[window.context_end - 1].tz_convert(self.timezone),
            "delivery_date": window.delivery_date,
            "target_start": self.index[target[0]].tz_convert(self.timezone),
            "target_end": self.index[target[-1]].tz_convert(self.timezone),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        context = slice(window.context_start, window.context_end)
        target = window.target_positions
        da_stats = self.norm_stats["price_da"]
        rt_stats = self.norm_stats["price_rt"]
        sample = {
            "price_da_ctx": torch.from_numpy(
                self._norm(self.price_da[context], da_stats)
            ),
            "price_rt_ctx": torch.from_numpy(
                self._norm(self.price_rt[context], rt_stats)
            ),
            "price_da_tgt": torch.from_numpy(self.price_da[target].copy()),
            # RT真值只供事后结算/评价，绝不作为DA模型输入或预测监督。
            "price_rt_tgt": torch.from_numpy(self.price_rt[target].copy()),
            "price_da_mean": torch.tensor(da_stats["mean"], dtype=torch.float32),
            "price_da_std": torch.tensor(da_stats["std"], dtype=torch.float32),
            "price_rt_mean": torch.tensor(rt_stats["mean"], dtype=torch.float32),
            "price_rt_std": torch.tensor(rt_stats["std"], dtype=torch.float32),
        }
        for name, values in self.streams.items():
            sample[f"{name}_ctx"] = torch.from_numpy(
                self._norm(values[context], self.norm_stats[name])
            )
        # 目标日的时钟、星期、月份、周末/节假日是起报时已经确定的信息，
        # 可以合法用于24个DA query；这里只输出日历，不输出未来实际负荷/风光。
        sample["cal_tgt"] = torch.from_numpy(
            self._norm(self.streams["cal"][target], self.norm_stats["cal"])
        )
        return sample


def build_da_datasets(cfg: PilotConfig):
    """返回固定D-1截止时点的(train, val, test, wide_df)。"""
    wide_df = load_market_unified(
        market=cfg.market,
        node=cfg.node,
        start="2020-01-01",
        end="2026-06-02",
    )
    train_ds = DecisionAwareDADataset(wide_df, cfg, "train")
    val_ds = DecisionAwareDADataset(
        wide_df, cfg, "val", norm_stats=train_ds.norm_stats
    )
    test_ds = DecisionAwareDADataset(
        wide_df, cfg, "test", norm_stats=train_ds.norm_stats
    )
    return train_ds, val_ds, test_ds, wide_df
