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
from .portable_normalization import (
    CONTEXT_ROBUST_MODE,
    normalization_mode,
    normalize as portable_normalize,
    portable_contract,
    robust_context_stats,
)


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
    lag168_positions: Optional[np.ndarray]
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
        self.normalization_mode = normalization_mode(cfg)
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

        # 市场物理压力使用各ISO都具备、单位一致的历史净负荷：
        # load - wind - solar。它只读取起报时以前的实际量；目标小时仅使用
        # 168小时前同小时的已知值，不接入发布时间不明的未来预测。
        raw_pressure = (
            self.streams["load"][:, 0]
            - self.streams["system"][:, 0]
            - self.streams["system"][:, 1]
        ).astype(np.float32)

        train_lo, train_hi = (
            _market_ts(value, self.timezone)
            for value in cfg.split_bounds("train")
        )
        train_mask = (self.index >= train_lo) & (self.index <= train_hi)
        if self.normalization_mode == CONTEXT_ROBUST_MODE:
            # 该合同不包含任何节点拟合参数；验证和新市场推理都只使用各自
            # 起报时可见的168小时历史窗口。
            norm_stats = portable_contract(self.context_len)
        elif norm_stats is None:
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
        if self.normalization_mode == CONTEXT_ROBUST_MODE:
            self.physical_pressure = raw_pressure
        else:
            pressure_mean = float(raw_pressure[train_mask].mean())
            pressure_std = float(raw_pressure[train_mask].std() + 1e-8)
            self.physical_pressure = (
                (raw_pressure - pressure_mean) / pressure_std
            ).astype(np.float32)

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

            # 168小时前的同一UTC交付小时完全位于已知上下文中。显式保存
            # 位置并核对时间差，避免数据缺口让“减168行”错误地跨过缺失小时。
            lag168_positions: Optional[np.ndarray] = None
            if str(getattr(cfg, "global_residual_anchor", "none")).lower() == "lag168":
                lag168_positions = target_positions - 168
                if (
                    np.any(lag168_positions < context_start)
                    or np.any(lag168_positions >= context_end)
                ):
                    continue
                lag_delta = (
                    target_times.as_unit("ns").asi8
                    - self.index[lag168_positions].as_unit("ns").asi8
                )
                if np.any(lag_delta != pd.Timedelta(hours=168).value):
                    continue

            self.windows.append(
                DASampleWindow(
                    context_start=context_start,
                    context_end=context_end,
                    target_positions=target_positions,
                    lag168_positions=lag168_positions,
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
        if self.normalization_mode == CONTEXT_ROBUST_MODE:
            da_stats = robust_context_stats(self.price_da[context])
            rt_stats = robust_context_stats(self.price_rt[context])
            stream_stats = {
                name: robust_context_stats(values[context])
                for name, values in self.streams.items()
                if name != "cal"
            }
        else:
            da_stats = self.norm_stats["price_da"]
            rt_stats = self.norm_stats["price_rt"]
            stream_stats = {
                name: self.norm_stats[name] for name in self.streams
            }
        sample = {
            "price_da_ctx": torch.from_numpy(
                portable_normalize(self.price_da[context], da_stats)
            ),
            "price_rt_ctx": torch.from_numpy(
                portable_normalize(self.price_rt[context], rt_stats)
            ),
            "price_da_tgt": torch.from_numpy(self.price_da[target].copy()),
            # RT真值只供事后结算/评价，绝不作为DA模型输入或预测监督。
            "price_rt_tgt": torch.from_numpy(self.price_rt[target].copy()),
            "price_da_mean": torch.tensor(da_stats["mean"], dtype=torch.float32),
            "price_da_std": torch.tensor(da_stats["std"], dtype=torch.float32),
            "price_rt_mean": torch.tensor(rt_stats["mean"], dtype=torch.float32),
            "price_rt_std": torch.tensor(rt_stats["std"], dtype=torch.float32),
        }
        if window.lag168_positions is not None:
            lag168 = window.lag168_positions
            # 合法的历史季节锚点：模型可选择预测相对它的残差。
            sample["price_da_lag168"] = torch.from_numpy(
                self.price_da[lag168].copy()
            )
            sample["price_rt_lag168"] = torch.from_numpy(
                self.price_rt[lag168].copy()
            )
        for name, values in self.streams.items():
            if name == "cal" and self.normalization_mode == CONTEXT_ROBUST_MODE:
                normalized = values[context].astype(np.float32, copy=True)
            else:
                normalized = portable_normalize(values[context], stream_stats[name])
            sample[f"{name}_ctx"] = torch.from_numpy(normalized)
        if str(
            getattr(self.cfg, "global_output_decomposition", "direct")
        ).lower() == "physical_scarcity_residual":
            pressure_lag168 = target - 168
            if (
                np.any(pressure_lag168 < window.context_start)
                or np.any(pressure_lag168 >= window.context_end)
            ):
                raise RuntimeError("物理压力lag168越过了DA起报信息边界")
            if self.normalization_mode == CONTEXT_ROBUST_MODE:
                pressure_stats = robust_context_stats(
                    self.physical_pressure[context]
                )
                pressure_context = portable_normalize(
                    self.physical_pressure[context], pressure_stats
                )
                pressure_target = portable_normalize(
                    self.physical_pressure[pressure_lag168], pressure_stats
                )
            else:
                pressure_context = self.physical_pressure[context].copy()
                pressure_target = self.physical_pressure[pressure_lag168].copy()
            sample["physical_pressure_ctx"] = torch.from_numpy(pressure_context)
            sample["physical_pressure_tgt_lag168"] = torch.from_numpy(
                pressure_target
            )
        # 目标日的时钟、星期、月份、周末/节假日是起报时已经确定的信息，
        # 可以合法用于24个DA query；这里只输出日历，不输出未来实际负荷/风光。
        if self.normalization_mode == CONTEXT_ROBUST_MODE:
            target_calendar = self.streams["cal"][target].astype(
                np.float32, copy=True
            )
        else:
            target_calendar = self._norm(
                self.streams["cal"][target], self.norm_stats["cal"]
            )
        sample["cal_tgt"] = torch.from_numpy(target_calendar)
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
