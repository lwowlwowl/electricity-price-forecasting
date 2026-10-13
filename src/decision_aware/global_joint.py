"""多节点三个预测任务按“节点 × 交付日”严格对齐。

该模块只组合已经构造好的 global Dataset，不读取、更不会修改原始市场数据。
每个联合样本仍保留三个独立模型所需的输入，并把稳定的节点/市场编号分别
注入三个分支，供 entity embedding 使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .global_data import EntityKey, MultiNodeDataset, RTDeliveryDayDataset


@dataclass(frozen=True)
class GlobalJointWindow:
    """一个节点在一个完整交付日上的三个任务索引。"""

    delivery_date: date
    da_index: int
    rt_at_da_index: int
    rt_index: int
    starts_new_segment: bool


def _date_index(dataset: Dataset, task: str) -> dict[date, int]:
    """提取单节点任务数据集的唯一交付日索引。"""
    if task == "rt":
        if not isinstance(dataset, RTDeliveryDayDataset):
            raise TypeError("rt分支必须先包装为RTDeliveryDayDataset")
        pairs = [(delivery_date, index) for index, (delivery_date, _) in enumerate(dataset.days)]
    else:
        pairs = []
        for index in range(len(dataset)):
            contract = dataset.sample_time_contract(index)
            delivery_date = contract.get("delivery_date")
            if delivery_date is None:
                raise ValueError(f"{task}样本缺少delivery_date时间合同")
            pairs.append((delivery_date, index))
    result: dict[date, int] = {}
    for delivery_date, index in pairs:
        if delivery_date in result:
            raise ValueError(f"{task}存在重复交付日{delivery_date}")
        result[delivery_date] = index
    return result


def _rt_local_timestamps(dataset: RTDeliveryDayDataset, index: int) -> tuple[pd.Timestamp, ...]:
    """返回一个RT日样本的24个带时区执行时刻。"""
    _, hourly_indices = dataset.days[index]
    timestamps = tuple(
        dataset.hourly_dataset.windows[item].issue_time_utc.tz_convert(
            dataset.hourly_dataset.timezone
        )
        for item in hourly_indices
    )
    if len(timestamps) != 24 or any(value.tzinfo is None for value in timestamps):
        raise ValueError("联合样本要求24个带时区的RT执行时刻")
    one_hour = pd.Timedelta(hours=1)
    utc = [value.tz_convert("UTC") for value in timestamps]
    if [value.hour for value in timestamps] != list(range(24)) or any(
        right - left != one_hour for left, right in zip(utc, utc[1:])
    ):
        raise ValueError("RT日样本不是当地00至23点的连续24小时")
    return timestamps


class GlobalJointDecisionAwareDataset(Dataset):
    """将三个 global 任务组合为节点隔离的联合交付日样本。"""

    def __init__(
        self,
        da: MultiNodeDataset,
        rt_at_da: MultiNodeDataset,
        rt: MultiNodeDataset,
    ) -> None:
        if not (da.entities == rt_at_da.entities == rt.entities):
            raise ValueError("三个global任务的实体集合或顺序不一致")
        if not (
            da.mappings.checkpoint_metadata()
            == rt_at_da.mappings.checkpoint_metadata()
            == rt.mappings.checkpoint_metadata()
        ):
            raise ValueError("三个global任务的节点/市场映射不一致")
        self.mappings = da.mappings
        self.entities = da.entities
        self.task_datasets = {"da": da, "rt_at_da": rt_at_da, "rt": rt}
        self.datasets: tuple[tuple[Dataset, Dataset, Dataset], ...] = tuple(
            zip(da.datasets, rt_at_da.datasets, rt.datasets)
        )
        self.windows_by_entity: list[tuple[GlobalJointWindow, ...]] = []
        self.timestamps_by_entity: list[tuple[tuple[pd.Timestamp, ...], ...]] = []
        self.excluded_dates_by_entity: dict[EntityKey, dict[str, list[str]]] = {}
        self.offsets = [0]

        for entity, (da_ds, rt_da_ds, rt_ds) in zip(self.entities, self.datasets):
            da_dates = _date_index(da_ds, "da")
            rt_da_dates = _date_index(rt_da_ds, "rt_at_da")
            rt_dates = _date_index(rt_ds, "rt")
            common = sorted(set(da_dates) & set(rt_da_dates) & set(rt_dates))
            if not common:
                raise ValueError(f"实体{entity}没有三个任务共同的完整交付日")

            self.excluded_dates_by_entity[entity] = {
                "da_only_or_partial": sorted(str(value) for value in set(da_dates) - set(common)),
                "rt_at_da_only_or_partial": sorted(
                    str(value) for value in set(rt_da_dates) - set(common)
                ),
                "rt_only_or_partial": sorted(str(value) for value in set(rt_dates) - set(common)),
            }
            entity_windows: list[GlobalJointWindow] = []
            entity_timestamps: list[tuple[pd.Timestamp, ...]] = []
            previous_last_utc: pd.Timestamp | None = None
            for delivery_date in common:
                timestamps = _rt_local_timestamps(rt_ds, rt_dates[delivery_date])
                first_utc = timestamps[0].tz_convert("UTC")
                starts_new_segment = (
                    previous_last_utc is None
                    or first_utc - previous_last_utc != pd.Timedelta(hours=1)
                )
                entity_windows.append(GlobalJointWindow(
                    delivery_date=delivery_date,
                    da_index=da_dates[delivery_date],
                    rt_at_da_index=rt_da_dates[delivery_date],
                    rt_index=rt_dates[delivery_date],
                    starts_new_segment=starts_new_segment,
                ))
                entity_timestamps.append(timestamps)
                previous_last_utc = timestamps[-1].tz_convert("UTC")
            self.windows_by_entity.append(tuple(entity_windows))
            self.timestamps_by_entity.append(tuple(entity_timestamps))
            self.offsets.append(self.offsets[-1] + len(entity_windows))

        self.norm_stats_by_task_by_node = {
            task: dict(dataset.norm_stats_by_node)
            for task, dataset in self.task_datasets.items()
        }

    def __len__(self) -> int:
        return self.offsets[-1]

    @property
    def entity_lengths(self) -> tuple[int, ...]:
        return tuple(len(items) for items in self.windows_by_entity)

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entity_index = 0
        while index >= self.offsets[entity_index + 1]:
            entity_index += 1
        return entity_index, index - self.offsets[entity_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        entity_index, local_index = self._locate(index)
        market, node = self.entities[entity_index]
        market_id, node_id = self.mappings.identity(market, node)
        window = self.windows_by_entity[entity_index][local_index]
        da_ds, rt_da_ds, rt_ds = self.datasets[entity_index]

        def inject(sample: Mapping[str, Any]) -> dict[str, Any]:
            result = dict(sample)
            result["market_id"] = torch.tensor(market_id, dtype=torch.long)
            result["node_id"] = torch.tensor(node_id, dtype=torch.long)
            return result

        return {
            "da": inject(da_ds[window.da_index]),
            "rt_at_da": inject(rt_da_ds[window.rt_at_da_index]),
            "rt": inject(rt_ds[window.rt_index]),
            "market_id": torch.tensor(market_id, dtype=torch.long),
            "node_id": torch.tensor(node_id, dtype=torch.long),
            "starts_new_segment": torch.tensor(window.starts_new_segment, dtype=torch.bool),
            "delivery_date_ordinal": torch.tensor(
                window.delivery_date.toordinal(), dtype=torch.int64
            ),
        }

    def sample_identity(self, index: int) -> dict[str, Any]:
        entity_index, local_index = self._locate(index)
        market, node = self.entities[entity_index]
        market_id, node_id = self.mappings.identity(market, node)
        window = self.windows_by_entity[entity_index][local_index]
        return {
            "market": market,
            "node": node,
            "market_id": market_id,
            "node_id": node_id,
            "local_index": local_index,
            "delivery_date": window.delivery_date,
            "starts_new_segment": window.starts_new_segment,
        }

    def local_timestamps(self, index: int) -> tuple[pd.Timestamp, ...]:
        entity_index, local_index = self._locate(index)
        return self.timestamps_by_entity[entity_index][local_index]


class NodeBalancedEpisodeBatchSampler(Sampler[list[int]]):
    """每个节点等量抽取连续episode，且绝不跨越真实时间缺口。"""

    def __init__(
        self,
        dataset: GlobalJointDecisionAwareDataset,
        *,
        episode_days: int,
        episodes_per_node: int,
        seed: int = 0,
    ) -> None:
        if episode_days < 1 or episodes_per_node < 1:
            raise ValueError("episode_days和episodes_per_node必须至少为1")
        self.dataset = dataset
        self.episode_days = int(episode_days)
        self.episodes_per_node = int(episodes_per_node)
        self.seed = int(seed)
        self.epoch = 0
        candidates = []
        for entity, windows in zip(dataset.entities, dataset.windows_by_entity):
            starts = []
            for start in range(len(windows) - self.episode_days + 1):
                # episode首日允许是新段；后续日一旦是新段就说明中间有缺口。
                if any(
                    windows[index].starts_new_segment
                    for index in range(start + 1, start + self.episode_days)
                ):
                    continue
                starts.append(start)
            if not starts:
                raise ValueError(
                    f"实体{entity}没有长度为{self.episode_days}的连续episode"
                )
            candidates.append(tuple(starts))
        self.candidate_starts = tuple(candidates)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset.entities) * self.episodes_per_node

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        entity_count = len(self.dataset.entities)
        for _ in range(self.episodes_per_node):
            for entity_index in torch.randperm(
                entity_count, generator=generator
            ).tolist():
                choices = self.candidate_starts[entity_index]
                selected = int(torch.randint(
                    len(choices), (1,), generator=generator
                ).item())
                local_start = choices[selected]
                global_start = self.dataset.offsets[entity_index] + local_start
                yield list(range(global_start, global_start + self.episode_days))


def build_global_joint_datasets(
    configs: Mapping[str, Any],
    entities: tuple[EntityKey, ...] | list[EntityKey],
    mappings,
    *,
    train_stage="inner_train",
    validation_stage="inner_validation",
) -> tuple[GlobalJointDecisionAwareDataset, GlobalJointDecisionAwareDataset]:
    """按三个独立checkpoint配置构造训练与验证联合数据集。"""
    from .global_data import build_global_task_datasets

    if set(configs) != {"da", "rt_at_da", "rt"}:
        raise ValueError("configs必须且只能包含da、rt_at_da、rt")
    built = {
        task: build_global_task_datasets(
            configs[task], task, entities, mappings,
            train_stage=train_stage, validation_stage=validation_stage,
        )
        for task in ("da", "rt_at_da", "rt")
    }
    return (
        GlobalJointDecisionAwareDataset(*(built[task][0] for task in ("da", "rt_at_da", "rt"))),
        GlobalJointDecisionAwareDataset(*(built[task][1] for task in ("da", "rt_at_da", "rt"))),
    )
