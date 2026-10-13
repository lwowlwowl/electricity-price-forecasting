"""多节点 global 训练的数据身份、包装器和等权采样器。

该模块不读取或修改原始数据。它把已经按节点构造好的单节点 Dataset 组合成
一个 global Dataset，并保证节点/市场编号、逐节点 scaler 和采样方式可审计。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler
import yaml

from .contracts import formal_checkpoint_metadata
from .contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT
from .dataset_da import DecisionAwareDADataset
from .dataset_rt import DecisionAwareRTDataset
from .dataset_rt_da import DecisionAwareRTAtDADataset
from .loader_v2 import load_market_unified_nodes


MARKET_ORDER = ("CAISO", "ERCOT", "NYISO", "PJM")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class EntityMappings:
    """稳定的全量市场、节点及节点所属市场映射。"""

    market_to_id: dict[str, int]
    node_to_id: dict[str, int]
    node_to_market: dict[str, str]

    @classmethod
    def from_nodes_config(cls, path: str | Path) -> "EntityMappings":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = sorted(set(raw) - set(MARKET_ORDER))
        if unknown:
            raise ValueError(f"nodes配置包含未知市场: {unknown}")

        market_to_id = {
            market: market_id
            for market_id, market in enumerate(MARKET_ORDER)
            if market in raw
        }
        node_to_market: dict[str, str] = {}
        ordered_nodes: list[str] = []
        for market in MARKET_ORDER:
            if market not in raw:
                continue
            nodes = raw[market].get("all", [])
            if not isinstance(nodes, list) or not nodes:
                raise ValueError(f"{market}.all必须是非空节点列表")
            for node in sorted(str(item) for item in nodes):
                if node in node_to_market:
                    raise ValueError(
                        f"节点名必须全局唯一，{node!r}同时出现在"
                        f"{node_to_market[node]}和{market}"
                    )
                node_to_market[node] = market
                ordered_nodes.append(node)

        node_to_id = {node: node_id for node_id, node in enumerate(ordered_nodes)}
        return cls(
            market_to_id=market_to_id,
            node_to_id=node_to_id,
            node_to_market=node_to_market,
        )

    @property
    def mapping_hash(self) -> str:
        return sha256(_canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    def identity(self, market: str, node: str) -> tuple[int, int]:
        expected_market = self.node_to_market.get(node)
        if expected_market is None:
            raise ValueError(f"正式节点映射中不存在节点 {node!r}")
        if expected_market != market:
            raise ValueError(
                f"节点 {node!r}属于{expected_market}，不能标记为{market}"
            )
        return self.market_to_id[market], self.node_to_id[node]

    def subset(self, markets: Sequence[str]) -> "EntityMappings":
        """按指定市场生成紧凑映射；市场顺序不影响结果。"""
        requested = set(markets)
        unknown = sorted(requested - set(self.market_to_id))
        if unknown:
            raise ValueError(f"正式映射中不存在市场: {unknown}")
        ordered_markets = [market for market in MARKET_ORDER if market in requested]
        market_to_id = {
            market: market_id for market_id, market in enumerate(ordered_markets)
        }
        nodes = [
            node
            for node, _ in sorted(self.node_to_id.items(), key=lambda item: item[1])
            if self.node_to_market[node] in requested
        ]
        return EntityMappings(
            market_to_id=market_to_id,
            node_to_id={node: node_id for node_id, node in enumerate(nodes)},
            node_to_market={node: self.node_to_market[node] for node in nodes},
        )

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "market_mapping": dict(self.market_to_id),
            "node_mapping": dict(self.node_to_id),
            "node_to_market": dict(self.node_to_market),
            "entity_mapping_hash": self.mapping_hash,
        }

    def validate_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        expected = self.checkpoint_metadata()
        for key, expected_value in expected.items():
            if checkpoint.get(key) != expected_value:
                raise ValueError(f"checkpoint的{key}与当前正式映射不一致")


EntityKey = tuple[str, str]


class RTDeliveryDayDataset(Dataset):
    """把滚动RT的24个小时窗口组合为一个“节点×交付日”样本。"""

    def __init__(self, hourly_dataset: DecisionAwareRTDataset) -> None:
        if hourly_dataset.cfg.train_stride != 1 and hourly_dataset.split == "train":
            raise ValueError("按日RT训练要求train_stride=1，才能覆盖每日24个起报小时")
        self.hourly_dataset = hourly_dataset
        self.norm_stats = hourly_dataset.norm_stats
        self.cfg = hourly_dataset.cfg
        self.split = hourly_dataset.split
        by_date: dict[Any, list[int]] = {}
        for index, window in enumerate(hourly_dataset.windows):
            local_date = window.issue_time_utc.tz_convert(
                hourly_dataset.timezone
            ).date()
            by_date.setdefault(local_date, []).append(index)

        self.days: list[tuple[Any, tuple[int, ...]]] = []
        one_hour_ns = torch.tensor(3_600_000_000_000, dtype=torch.int64)
        for delivery_date, indices in sorted(by_date.items()):
            if len(indices) != 24:
                continue
            times = [hourly_dataset.windows[index].issue_time_utc for index in indices]
            hours = [time.tz_convert(hourly_dataset.timezone).hour for time in times]
            if hours != list(range(24)):
                continue
            time_ns = torch.tensor([time.value for time in times], dtype=torch.int64)
            if not bool(torch.all(torch.diff(time_ns) == one_hour_ns)):
                continue
            self.days.append((delivery_date, tuple(indices)))
        if not self.days:
            raise ValueError("滚动RT数据中没有完整的24小时交付日")

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        _, hourly_indices = self.days[index]
        samples = [self.hourly_dataset[item] for item in hourly_indices]
        return {
            key: torch.stack([sample[key] for sample in samples], dim=0)
            for key in samples[0]
        }

    def sample_time_contract(self, index: int) -> dict[str, Any]:
        delivery_date, hourly_indices = self.days[index]
        first = self.hourly_dataset.sample_time_contract(hourly_indices[0])
        last = self.hourly_dataset.sample_time_contract(hourly_indices[-1])
        return {
            "market": first["market"],
            "timezone": first["timezone"],
            "delivery_date": delivery_date,
            "first_issue_time": first["issue_time"],
            "last_issue_time": last["issue_time"],
            "hourly_windows": 24,
        }


def flatten_rt_delivery_day_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """训练前把[B,24,...]日样本展平为[B×24,...]模型输入。"""
    reference = batch.get("price_rt_tgt")
    if not torch.is_tensor(reference) or reference.ndim < 3:
        raise ValueError("RT日batch必须含形如[B,24,H]的price_rt_tgt")
    batch_size, hours = reference.shape[:2]
    if hours != 24:
        raise ValueError(f"RT日batch必须有24个起报窗口，实际为{hours}")
    flattened: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            flattened[key] = value
        elif key in {"node_id", "market_id"}:
            flattened[key] = value.reshape(batch_size).repeat_interleave(hours)
        elif value.shape[:2] == (batch_size, hours):
            flattened[key] = value.reshape(
                batch_size * hours, *value.shape[2:]
            )
        else:
            raise ValueError(f"RT日batch字段{key}的前两维不一致: {tuple(value.shape)}")
    return flattened


class MultiNodeDataset(Dataset):
    """给单节点样本注入稳定 ``market_id`` 和 ``node_id``。"""

    def __init__(
        self,
        datasets: Mapping[EntityKey, Dataset],
        mappings: EntityMappings,
    ) -> None:
        if not datasets:
            raise ValueError("global数据集至少需要一个节点")

        def sort_key(entity: EntityKey) -> tuple[int, int]:
            market, node = entity
            market_id, node_id = mappings.identity(market, node)
            return market_id, node_id

        self.mappings = mappings
        self.entities = tuple(sorted(datasets, key=sort_key))
        self.datasets = tuple(datasets[entity] for entity in self.entities)
        self.offsets: list[int] = [0]
        for entity, dataset in zip(self.entities, self.datasets):
            if len(dataset) < 1:
                raise ValueError(f"节点{entity}没有可用样本")
            self.offsets.append(self.offsets[-1] + len(dataset))

        self.norm_stats_by_node: dict[str, Any] = {}
        for (_, node), dataset in zip(self.entities, self.datasets):
            if not hasattr(dataset, "norm_stats"):
                raise ValueError(f"节点{node}的数据集缺少norm_stats")
            self.norm_stats_by_node[node] = dataset.norm_stats

    def __len__(self) -> int:
        return self.offsets[-1]

    @property
    def entity_lengths(self) -> tuple[int, ...]:
        return tuple(len(dataset) for dataset in self.datasets)

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entity_index = 0
        # 节点数最多几十个；线性查找简单且不会进入模型计算热点。
        while index >= self.offsets[entity_index + 1]:
            entity_index += 1
        return entity_index, index - self.offsets[entity_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        entity_index, local_index = self._locate(index)
        market, node = self.entities[entity_index]
        market_id, node_id = self.mappings.identity(market, node)
        original = self.datasets[entity_index][local_index]
        if not isinstance(original, Mapping):
            raise TypeError("单节点Dataset样本必须是字典")
        sample = dict(original)
        sample["market_id"] = torch.tensor(market_id, dtype=torch.long)
        sample["node_id"] = torch.tensor(node_id, dtype=torch.long)
        return sample

    def sample_identity(self, index: int) -> dict[str, Any]:
        entity_index, local_index = self._locate(index)
        market, node = self.entities[entity_index]
        market_id, node_id = self.mappings.identity(market, node)
        return {
            "market": market,
            "node": node,
            "market_id": market_id,
            "node_id": node_id,
            "local_index": local_index,
        }


class NodeBalancedSampler(Sampler[int]):
    """每个epoch让每个节点贡献完全相同数量的随机样本。"""

    def __init__(
        self,
        dataset: MultiNodeDataset,
        samples_per_node: int | None = None,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.samples_per_node = (
            max(dataset.entity_lengths)
            if samples_per_node is None
            else int(samples_per_node)
        )
        if self.samples_per_node < 1:
            raise ValueError("samples_per_node必须至少为1")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset.entities) * self.samples_per_node

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        entity_count = len(self.dataset.entities)
        for _ in range(self.samples_per_node):
            # 每一轮都先随机打乱节点，再从该节点随机抽一个日期/窗口。
            for entity_index in torch.randperm(
                entity_count, generator=generator
            ).tolist():
                length = self.dataset.entity_lengths[entity_index]
                local_index = int(
                    torch.randint(length, (1,), generator=generator).item()
                )
                yield self.dataset.offsets[entity_index] + local_index


class MarketNodeBalancedSampler(Sampler[int]):
    """跨市场时先等权抽市场，再在该市场内等权抽节点。"""

    def __init__(
        self,
        dataset: MultiNodeDataset,
        samples_per_market: int,
        seed: int = 0,
    ) -> None:
        if samples_per_market < 1:
            raise ValueError("samples_per_market必须至少为1")
        self.dataset = dataset
        self.samples_per_market = int(samples_per_market)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[str, list[int]] = {}
        for entity_index, (market, _) in enumerate(dataset.entities):
            grouped.setdefault(market, []).append(entity_index)
        self.market_entities = tuple(
            (market, tuple(grouped[market]))
            for market in MARKET_ORDER
            if market in grouped
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.market_entities) * self.samples_per_market

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        market_count = len(self.market_entities)
        for _ in range(self.samples_per_market):
            for market_index in torch.randperm(
                market_count, generator=generator
            ).tolist():
                _, entity_indices = self.market_entities[market_index]
                selected = int(
                    torch.randint(
                        len(entity_indices), (1,), generator=generator
                    ).item()
                )
                entity_index = entity_indices[selected]
                length = self.dataset.entity_lengths[entity_index]
                local_index = int(
                    torch.randint(length, (1,), generator=generator).item()
                )
                yield self.dataset.offsets[entity_index] + local_index


def load_formal_entity_mappings(
    path: str | Path = "configs/nodes.yaml",
) -> EntityMappings:
    return EntityMappings.from_nodes_config(path)


TASK_DATASET_CLASSES = {
    "da": DecisionAwareDADataset,
    "rt_at_da": DecisionAwareRTAtDADataset,
    "rt": DecisionAwareRTDataset,
}


def _range_bounds(stage: AnnualStage | str) -> tuple[str, str]:
    date_range = FORMAL_EXPERIMENT_CONTRACT.annual_split.range_for(stage)
    return f"{date_range.start} 00:00", f"{date_range.end} 23:00"


def build_global_task_datasets(
    config: Any,
    task: str,
    entities: Sequence[EntityKey],
    mappings: EntityMappings,
    *,
    train_stage: AnnualStage | str = AnnualStage.INNER_TRAIN,
    validation_stage: AnnualStage | str = AnnualStage.INNER_VALIDATION,
    read_end_utc: str | None = None,
) -> tuple[MultiNodeDataset, MultiNodeDataset]:
    """读取多个节点并构造训练/验证 global Dataset。

    每个节点独立拟合训练期 ``norm_stats``，验证集只复用该节点统计量。
    """
    if task not in TASK_DATASET_CLASSES:
        raise ValueError(f"未知global任务{task!r}；可选{sorted(TASK_DATASET_CLASSES)}")
    if not entities:
        raise ValueError("至少需要一个market/node实体")
    participating_markets = tuple(dict.fromkeys(market for market, _ in entities))
    require_explicit_issue_hour = len(participating_markets) > 1
    issue_hours = {
        market: config.resolved_da_issue_hour_local(
            market, require_explicit=require_explicit_issue_hour
        )
        for market in participating_markets
    }
    dataset_class = TASK_DATASET_CLASSES[task]
    train_start, train_end = _range_bounds(train_stage)
    val_start, val_end = _range_bounds(validation_stage)
    train_datasets: dict[EntityKey, Dataset] = {}
    validation_datasets: dict[EntityKey, Dataset] = {}

    wide_by_entity: dict[EntityKey, Any] = {}
    nodes_by_market: dict[str, list[str]] = {}
    for market, node in entities:
        mappings.identity(market, node)
        nodes_by_market.setdefault(market, []).append(node)
    for market, nodes in nodes_by_market.items():
        loaded = load_market_unified_nodes(
            market=market,
            nodes=nodes,
            start="2020-01-01",
            end="2026-06-02",
            **({"read_end_utc": read_end_utc} if read_end_utc is not None else {}),
        )
        wide_by_entity.update({(market, node): frame for node, frame in loaded.items()})

    for market, node in entities:
        config_updates = {
            "market": market,
            "node": node,
            "da_issue_hour_local": issue_hours[market],
            "split_train_start": train_start,
            "split_train_end": train_end,
            "split_val_start": val_start,
            "split_val_end": val_end,
        }
        if task == "rt":
            # Global RT的采样单位是完整交付日，必须先保留每日24个小时窗口。
            config_updates.update(train_stride=1, val_stride=1)
        node_config = replace(
            config,
            **config_updates,
        )
        wide_df = wide_by_entity[(market, node)]
        train_dataset: Dataset = dataset_class(wide_df, node_config, "train")
        validation_dataset: Dataset = dataset_class(
            wide_df,
            node_config,
            "val",
            norm_stats=train_dataset.norm_stats,
        )
        if task == "rt":
            train_dataset = RTDeliveryDayDataset(train_dataset)
            validation_dataset = RTDeliveryDayDataset(validation_dataset)
        train_datasets[(market, node)] = train_dataset
        validation_datasets[(market, node)] = validation_dataset

    return (
        MultiNodeDataset(train_datasets, mappings),
        MultiNodeDataset(validation_datasets, mappings),
    )


def global_checkpoint_metadata(
    config: Any,
    dataset: MultiNodeDataset,
    *,
    require_all_nodes: bool = True,
) -> dict[str, Any]:
    """生成正式global checkpoint必须保存的合同、映射和逐节点scaler。"""
    mappings = dataset.mappings
    expected_nodes = len(mappings.node_to_id)
    expected_markets = len(mappings.market_to_id)
    uses_identity = bool(getattr(config, "global_use_entity_embeddings", False))
    if uses_identity:
        if getattr(config, "global_num_nodes", None) != expected_nodes:
            raise ValueError(
                f"global_num_nodes必须为{expected_nodes}，"
                f"实际为{getattr(config, 'global_num_nodes', None)}"
            )
        if getattr(config, "global_num_markets", None) != expected_markets:
            raise ValueError(
                f"global_num_markets必须为{expected_markets}，"
                f"实际为{getattr(config, 'global_num_markets', None)}"
            )
    elif (
        getattr(config, "global_num_nodes", 0) != 0
        or getattr(config, "global_num_markets", 0) != 0
    ):
        raise ValueError("关闭实体embedding时节点数和市场数必须同时为0")
    participating_nodes = {node for _, node in dataset.entities}
    if require_all_nodes and participating_nodes != set(mappings.node_to_id):
        missing = sorted(set(mappings.node_to_id) - participating_nodes)
        extra = sorted(participating_nodes - set(mappings.node_to_id))
        raise ValueError(
            f"正式global数据集节点不完整: missing={missing}, extra={extra}"
        )
    if set(dataset.norm_stats_by_node) != participating_nodes:
        raise ValueError("逐节点norm_stats与参与训练的节点集合不一致")

    metadata = formal_checkpoint_metadata(config)
    metadata.update(mappings.checkpoint_metadata())
    metadata.update({
        "entity_conditioning": {
            "enabled": uses_identity,
            "uses_node_id_for_prediction": uses_identity,
            "uses_market_id_for_prediction": uses_identity,
        },
        "normalization_mode": str(
            getattr(config, "global_normalization_mode", "training_entity")
        ),
        "participating_entities": [
            {"market": market, "node": node}
            for market, node in dataset.entities
        ],
        "norm_stats_by_node": dict(dataset.norm_stats_by_node),
    })
    return metadata
