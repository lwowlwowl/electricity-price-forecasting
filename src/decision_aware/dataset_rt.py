"""独立实时（RT）滚动样本：每小时起报，预测后续H个小时。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PilotConfig
from .dataset_da import STREAMS_DA, _market_ts
from .loader_v2 import load_market_unified, market_timezone
from .portable_normalization import (
    CONTEXT_ROBUST_MODE,
    normalization_mode,
    normalize as portable_normalize,
    portable_contract,
    robust_context_stats,
)


@dataclass(frozen=True)
class RTSampleWindow:
    context_start: int
    issue_position: int
    target_positions: np.ndarray
    lag168_positions: Optional[np.ndarray]
    issue_time_utc: pd.Timestamp


class DecisionAwareRTDataset(Dataset):
    """在小时t开始时，用[t-168,t)预测[t,t+H)的RT价格。

    目标时段DA价格和日历在实时起报时已经公布/确定，可以作为未来已知条件；
    未来实际负荷、风光和RT价格只作为标签或事后评价，绝不进入输入。
    """

    def __init__(self, wide_df: pd.DataFrame, cfg: PilotConfig, split: str,
                 norm_stats: Optional[Dict] = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"未知split: {split}")
        if cfg.horizon_rt < 1:
            raise ValueError("horizon_rt必须至少为1")
        self.cfg = cfg
        self.split = split
        self.context_len = cfg.context_len
        self.horizon_tgt = cfg.horizon_rt
        self.normalization_mode = normalization_mode(cfg)
        self.timezone = market_timezone(cfg.market)

        df = wide_df.sort_index().copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        if not df.index.is_unique:
            raise ValueError("RT数据索引存在重复时间戳")
        self.index = df.index
        self.price_da = df["price_da"].to_numpy(dtype=np.float32)
        self.price_rt = df["price_rt"].to_numpy(dtype=np.float32)
        self.streams = {
            name: df[columns].to_numpy(dtype=np.float32)
            for name, columns in STREAMS_DA.items()
        }
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
            norm_stats = portable_contract(self.context_len)
        elif norm_stats is None:
            if split != "train":
                raise ValueError("验证/测试集必须复用训练集norm_stats")
            if not train_mask.any():
                raise ValueError("数据中没有配置所指定的训练时段")
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
        # RT训练会重复读取约4.4万个高度重叠窗口。归一化只依赖固定的训练集
        # 统计量，因此在构造数据集时一次性计算，避免每个epoch重复做相同减除。
        if self.normalization_mode == CONTEXT_ROBUST_MODE:
            self.price_da_normalized = None
            self.price_rt_normalized = None
            self.normalized_streams = None
        else:
            self.price_da_normalized = self._norm(
                self.price_da, self.norm_stats["price_da"]
            )
            self.price_rt_normalized = self._norm(
                self.price_rt, self.norm_stats["price_rt"]
            )
            self.normalized_streams = {
                name: self._norm(values, self.norm_stats[name])
                for name, values in self.streams.items()
            }

        split_lo, split_hi = (
            _market_ts(value, self.timezone)
            for value in cfg.split_bounds(split)
        )
        stride = {
            "train": cfg.train_stride,
            "val": cfg.val_stride,
            "test": cfg.eval_stride,
        }[split]
        if stride < 1:
            raise ValueError("stride必须至少为1")
        one_hour_ns = pd.Timedelta(hours=1).value
        self.windows: List[RTSampleWindow] = []
        last_start = len(self.index) - self.horizon_tgt
        for issue_position in range(self.context_len, last_start + 1, stride):
            target_positions = np.arange(
                issue_position, issue_position + self.horizon_tgt, dtype=np.int64
            )
            target_times = self.index[target_positions]
            if target_times[0] < split_lo or target_times[-1] > split_hi:
                continue
            context_start = issue_position - self.context_len
            context_times = self.index[context_start:issue_position]
            if len(context_times) != self.context_len:
                continue
            if np.any(np.diff(context_times.as_unit("ns").asi8) != one_hour_ns):
                continue
            if np.any(np.diff(target_times.as_unit("ns").asi8) != one_hour_ns):
                continue
            lag168_positions: Optional[np.ndarray] = None
            if str(getattr(cfg, "global_residual_anchor", "none")).lower() == "lag168":
                lag168_positions = target_positions - 168
                if (
                    np.any(lag168_positions < context_start)
                    or np.any(lag168_positions >= issue_position)
                ):
                    continue
                lag_delta = (
                    target_times.as_unit("ns").asi8
                    - self.index[lag168_positions].as_unit("ns").asi8
                )
                if np.any(lag_delta != pd.Timedelta(hours=168).value):
                    continue
            self.windows.append(RTSampleWindow(
                context_start=context_start,
                issue_position=issue_position,
                target_positions=target_positions,
                lag168_positions=lag168_positions,
                issue_time_utc=self.index[issue_position],
            ))

    @staticmethod
    def _scalar_stats(values):
        return {"mean": float(values.mean()), "std": float(values.std() + 1e-8)}

    @staticmethod
    def _norm(values, stats):
        return ((values - stats["mean"]) / stats["std"]).astype(np.float32)

    def __len__(self):
        return len(self.windows)

    def sample_time_contract(self, idx: int):
        window = self.windows[idx]
        target = window.target_positions
        return {
            "market": self.cfg.market,
            "timezone": self.timezone,
            "issue_time": window.issue_time_utc.tz_convert(self.timezone),
            "context_start": self.index[window.context_start].tz_convert(self.timezone),
            "context_end": self.index[window.issue_position - 1].tz_convert(self.timezone),
            "target_start": self.index[target[0]].tz_convert(self.timezone),
            "target_end": self.index[target[-1]].tz_convert(self.timezone),
            "executed_target": self.index[target[0]].tz_convert(self.timezone),
        }

    def __getitem__(self, idx: int):
        window = self.windows[idx]
        context = slice(window.context_start, window.issue_position)
        target = window.target_positions
        if self.normalization_mode == CONTEXT_ROBUST_MODE:
            da_stats = robust_context_stats(self.price_da[context])
            rt_stats = robust_context_stats(self.price_rt[context])
            price_da_context = portable_normalize(self.price_da[context], da_stats)
            price_rt_context = portable_normalize(self.price_rt[context], rt_stats)
            price_da_target_known = portable_normalize(
                self.price_da[target], da_stats
            )
            stream_context = {
                name: (
                    values[context].astype(np.float32, copy=True)
                    if name == "cal"
                    else portable_normalize(
                        values[context], robust_context_stats(values[context])
                    )
                )
                for name, values in self.streams.items()
            }
            calendar_target = self.streams["cal"][target].astype(
                np.float32, copy=True
            )
        else:
            da_stats = self.norm_stats["price_da"]
            rt_stats = self.norm_stats["price_rt"]
            price_da_context = self.price_da_normalized[context]
            price_rt_context = self.price_rt_normalized[context]
            price_da_target_known = self.price_da_normalized[target]
            stream_context = {
                name: self.normalized_streams[name][context]
                for name in self.streams
            }
            calendar_target = self.normalized_streams["cal"][target]
        sample = {
            "price_da_ctx": torch.from_numpy(price_da_context),
            "price_rt_ctx": torch.from_numpy(price_rt_context),
            "price_da_tgt_known": torch.from_numpy(price_da_target_known),
            "price_da_tgt": torch.from_numpy(self.price_da[target].copy()),
            "price_rt_tgt": torch.from_numpy(self.price_rt[target].copy()),
            "price_da_mean": torch.tensor(
                da_stats["mean"], dtype=torch.float32
            ),
            "price_da_std": torch.tensor(
                da_stats["std"], dtype=torch.float32
            ),
            "price_rt_mean": torch.tensor(
                rt_stats["mean"], dtype=torch.float32
            ),
            "price_rt_std": torch.tensor(
                rt_stats["std"], dtype=torch.float32
            ),
        }
        if window.lag168_positions is not None:
            sample["price_rt_lag168"] = torch.from_numpy(
                self.price_rt[window.lag168_positions].copy()
            )
        for name in self.streams:
            sample[f"{name}_ctx"] = torch.from_numpy(stream_context[name])
        if str(
            getattr(self.cfg, "global_output_decomposition", "direct")
        ).lower() == "physical_scarcity_residual":
            pressure_lag168 = target - 168
            if (
                np.any(pressure_lag168 < window.context_start)
                or np.any(pressure_lag168 >= window.issue_position)
            ):
                raise RuntimeError("物理压力lag168越过了RT起报信息边界")
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
        sample["cal_tgt"] = torch.from_numpy(calendar_target)
        return sample


def build_rt_datasets(cfg: PilotConfig):
    wide_df = load_market_unified(
        market=cfg.market,
        node=cfg.node,
        start="2020-01-01",
        end="2026-06-02",
    )
    train_ds = DecisionAwareRTDataset(wide_df, cfg, "train")
    val_ds = DecisionAwareRTDataset(wide_df, cfg, "val", train_ds.norm_stats)
    test_ds = DecisionAwareRTDataset(wide_df, cfg, "test", train_ds.norm_stats)
    return train_ds, val_ds, test_ds, wide_df
