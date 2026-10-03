"""三个独立模型按同一交付日对齐的联合数据集。

每个样本严格包含：一个DA日样本、一个RT-at-DA日样本、该交付日24个
滚动RT起报样本，以及完整且带时区的物理时间合同。这里只读取统一市场表，
不修改 ``data/markets`` 中的任何原始文件。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PilotConfig
from .dataset_da import DecisionAwareDADataset
from .dataset_rt import DecisionAwareRTDataset
from .dataset_rt_da import DecisionAwareRTAtDADataset
from .loader_v2 import load_market_unified


@dataclass(frozen=True)
class JointDeliveryWindow:
    delivery_date: date
    da_index: int
    rt_at_da_index: int
    rt_indices: tuple[int, ...]
    starts_new_segment: bool


class JointDecisionAwareDataset(Dataset):
    """将三个数据集连接成完整交付日样本。"""

    def __init__(
        self,
        da_dataset: DecisionAwareDADataset,
        rt_at_da_dataset: DecisionAwareRTAtDADataset,
        rt_dataset: DecisionAwareRTDataset,
    ):
        self.da_dataset = da_dataset
        self.rt_at_da_dataset = rt_at_da_dataset
        self.rt_dataset = rt_dataset
        if not (
            da_dataset.split == rt_at_da_dataset.split == rt_dataset.split
        ):
            raise ValueError("联合数据集的三个split必须一致")
        self.split = da_dataset.split
        if not (
            da_dataset.timezone
            == rt_at_da_dataset.timezone
            == rt_dataset.timezone
        ):
            raise ValueError("联合数据集的市场时区不一致")
        self.timezone = da_dataset.timezone

        rt_da_by_date = {
            window.delivery_date: index
            for index, window in enumerate(rt_at_da_dataset.windows)
        }
        rt_by_issue = {
            window.issue_time_utc: index
            for index, window in enumerate(rt_dataset.windows)
        }

        self.windows: list[JointDeliveryWindow] = []
        self.excluded_missing_rt_dates: list[date] = []
        previous_last_utc: pd.Timestamp | None = None
        one_hour = pd.Timedelta(hours=1)
        for da_index, da_window in enumerate(da_dataset.windows):
            delivery_date = da_window.delivery_date
            if delivery_date not in rt_da_by_date:
                raise ValueError(
                    f"{delivery_date}存在DA样本但缺少RT-at-DA样本"
                )
            target_times = [
                da_dataset.index[position]
                for position in da_window.target_positions
            ]
            rt_indices = tuple(rt_by_issue.get(timestamp, -1) for timestamp in target_times)
            # 滚动RT的最后一个起报还需要未来H-1小时。若split末尾没有这些
            # 标签，整个交付日不能伪装成完整24小时联合样本。
            if len(rt_indices) != 24 or any(index < 0 for index in rt_indices):
                self.excluded_missing_rt_dates.append(delivery_date)
                continue
            starts_new_segment = (
                previous_last_utc is None
                or target_times[0] - previous_last_utc != one_hour
            )
            self.windows.append(
                JointDeliveryWindow(
                    delivery_date=delivery_date,
                    da_index=da_index,
                    rt_at_da_index=rt_da_by_date[delivery_date],
                    rt_indices=rt_indices,
                    starts_new_segment=starts_new_segment,
                )
            )
            previous_last_utc = target_times[-1]

        if not self.windows:
            raise ValueError("没有可用的完整交付日联合样本")

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _stack_samples(dataset: Dataset, indices: tuple[int, ...]) -> Dict[str, torch.Tensor]:
        rows = [dataset[index] for index in indices]
        return {
            key: torch.stack([row[key] for row in rows], dim=0)
            for key in rows[0]
        }

    def __getitem__(self, index: int) -> dict:
        window = self.windows[index]
        return {
            "da": self.da_dataset[window.da_index],
            "rt_at_da": self.rt_at_da_dataset[window.rt_at_da_index],
            "rt": self._stack_samples(self.rt_dataset, window.rt_indices),
            "starts_new_segment": torch.tensor(
                window.starts_new_segment, dtype=torch.bool
            ),
            "delivery_date_ordinal": torch.tensor(
                window.delivery_date.toordinal(), dtype=torch.int64
            ),
        }

    def delivery_dates(self) -> list[date]:
        return [window.delivery_date for window in self.windows]

    def local_timestamps(self) -> list[pd.Timestamp]:
        timestamps = []
        for window in self.windows:
            for rt_index in window.rt_indices:
                timestamps.append(
                    self.rt_dataset.windows[rt_index].issue_time_utc.tz_convert(
                        self.timezone
                    )
                )
        return timestamps


def build_joint_datasets(
    cfg: PilotConfig,
    norm_stats: Optional[Dict] = None,
):
    """返回(train, val, test, wide_df)，三个任务共用同一市场表和统计量。"""
    wide_df = load_market_unified(
        market=cfg.market,
        node=cfg.node,
        start="2020-01-01",
        end="2026-06-02",
    )
    da_train = DecisionAwareDADataset(
        wide_df, cfg, "train", norm_stats=norm_stats
    )
    shared_stats = da_train.norm_stats
    da_sets = {
        "train": da_train,
        "val": DecisionAwareDADataset(wide_df, cfg, "val", shared_stats),
        "test": DecisionAwareDADataset(wide_df, cfg, "test", shared_stats),
    }
    rt_da_sets = {
        split: DecisionAwareRTAtDADataset(wide_df, cfg, split, shared_stats)
        for split in ("train", "val", "test")
    }
    rt_sets = {
        split: DecisionAwareRTDataset(wide_df, cfg, split, shared_stats)
        for split in ("train", "val", "test")
    }
    joint = tuple(
        JointDecisionAwareDataset(
            da_sets[split], rt_da_sets[split], rt_sets[split]
        )
        for split in ("train", "val", "test")
    )
    return (*joint, wide_df)
