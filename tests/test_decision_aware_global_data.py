import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig
from decision_aware.global_data import (
    EntityMappings,
    MarketNodeBalancedSampler,
    MultiNodeDataset,
    NodeBalancedSampler,
    RTDeliveryDayDataset,
    flatten_rt_delivery_day_batch,
    global_checkpoint_metadata,
    load_formal_entity_mappings,
)


class _TinyNodeDataset(Dataset):
    def __init__(self, size: int, marker: float):
        self.size = size
        self.marker = marker
        self.norm_stats = {
            "price_da": {"mean": marker, "std": marker + 1.0}
        }

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"feature": torch.tensor([self.marker, float(index)])}


class _FakeHourlyRTDataset(Dataset):
    def __init__(self):
        self.cfg = SimpleNamespace(train_stride=1)
        self.split = "train"
        self.timezone = "UTC"
        self.norm_stats = {"price_rt": {"mean": 0.0, "std": 1.0}}
        start = pd.Timestamp("2020-01-01", tz="UTC")
        self.windows = [
            SimpleNamespace(issue_time_utc=start + pd.Timedelta(hours=hour))
            for hour in range(48)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return {
            "price_rt_tgt": torch.full((4,), float(index)),
            "price_rt_ctx": torch.full((3,), float(index)),
            "price_rt_mean": torch.tensor(0.0),
        }

    def sample_time_contract(self, index):
        return {
            "market": "ERCOT",
            "timezone": "UTC",
            "issue_time": self.windows[index].issue_time_utc,
        }


def _mappings():
    return EntityMappings(
        market_to_id={"CAISO": 0, "ERCOT": 1},
        node_to_id={"TH_NP15_GEN-APND": 0, "LZ_LCRA": 1},
        node_to_market={"TH_NP15_GEN-APND": "CAISO", "LZ_LCRA": "ERCOT"},
    )


def test_repository_mapping_has_all_64_nodes_and_is_stable():
    first = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    second = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")

    assert len(first.market_to_id) == 4
    assert len(first.node_to_id) == 64
    assert first.mapping_hash == second.mapping_hash
    assert first.node_to_market["LZ_LCRA"] == "ERCOT"

    ercot = first.subset(["ERCOT"])
    assert ercot.market_to_id == {"ERCOT": 0}
    assert len(ercot.node_to_id) == 15
    assert set(ercot.node_to_market.values()) == {"ERCOT"}


def test_multinode_dataset_injects_ids_without_sharing_node_scalers():
    mappings = _mappings()
    dataset = MultiNodeDataset(
        {
            ("ERCOT", "LZ_LCRA"): _TinyNodeDataset(3, 20.0),
            ("CAISO", "TH_NP15_GEN-APND"): _TinyNodeDataset(2, 10.0),
        },
        mappings,
    )

    first = dataset[0]
    last = dataset[-1]
    assert first["market_id"].item() == 0
    assert first["node_id"].item() == 0
    assert last["market_id"].item() == 1
    assert last["node_id"].item() == 1
    assert dataset.norm_stats_by_node["TH_NP15_GEN-APND"] != (
        dataset.norm_stats_by_node["LZ_LCRA"]
    )


def test_node_balanced_sampler_is_exactly_balanced_and_reproducible():
    dataset = MultiNodeDataset(
        {
            ("ERCOT", "LZ_LCRA"): _TinyNodeDataset(7, 20.0),
            ("CAISO", "TH_NP15_GEN-APND"): _TinyNodeDataset(2, 10.0),
        },
        _mappings(),
    )
    sampler = NodeBalancedSampler(dataset, samples_per_node=5, seed=123)
    indices = list(sampler)
    identities = [dataset.sample_identity(index)["node"] for index in indices]

    assert len(indices) == 10
    assert identities.count("LZ_LCRA") == 5
    assert identities.count("TH_NP15_GEN-APND") == 5
    assert indices == list(NodeBalancedSampler(dataset, 5, seed=123))

    sampler.set_epoch(1)
    assert indices != list(sampler)


def test_market_node_sampler_equalizes_markets_before_nodes():
    mappings = EntityMappings(
        market_to_id={"CAISO": 0, "ERCOT": 1},
        node_to_id={"CA_A": 0, "TX_A": 1, "TX_B": 2},
        node_to_market={"CA_A": "CAISO", "TX_A": "ERCOT", "TX_B": "ERCOT"},
    )
    dataset = MultiNodeDataset(
        {
            ("CAISO", "CA_A"): _TinyNodeDataset(2, 1.0),
            ("ERCOT", "TX_A"): _TinyNodeDataset(3, 2.0),
            ("ERCOT", "TX_B"): _TinyNodeDataset(9, 3.0),
        },
        mappings,
    )
    sampler = MarketNodeBalancedSampler(dataset, samples_per_market=2000, seed=9)
    identities = [dataset.sample_identity(index) for index in sampler]
    markets = [item["market"] for item in identities]
    nodes = [item["node"] for item in identities]

    assert markets.count("CAISO") == 2000
    assert markets.count("ERCOT") == 2000
    # ERCOT内部按节点等概率，不因节点窗口更多而天然占优。
    assert abs(nodes.count("TX_A") - nodes.count("TX_B")) < 200


def test_rt_hourly_windows_are_grouped_by_delivery_day_then_flattened():
    daily = RTDeliveryDayDataset(_FakeHourlyRTDataset())
    assert len(daily) == 2
    sample = daily[0]
    assert sample["price_rt_tgt"].shape == (24, 4)
    assert daily.sample_time_contract(0)["hourly_windows"] == 24

    batch = {
        key: value.unsqueeze(0).repeat((2,) + (1,) * value.ndim)
        for key, value in sample.items()
    }
    batch["node_id"] = torch.tensor([3, 4])
    batch["market_id"] = torch.tensor([0, 0])
    flattened = flatten_rt_delivery_day_batch(batch)
    assert flattened["price_rt_tgt"].shape == (48, 4)
    assert flattened["price_rt_ctx"].shape == (48, 3)
    assert flattened["node_id"].tolist()[:24] == [3] * 24
    assert flattened["node_id"].tolist()[24:] == [4] * 24


def test_checkpoint_mapping_rejects_reordered_or_missing_ids():
    mappings = _mappings()
    metadata = mappings.checkpoint_metadata()
    mappings.validate_checkpoint(metadata)

    broken = dict(metadata)
    broken["node_mapping"] = {"LZ_LCRA": 0, "TH_NP15_GEN-APND": 1}
    try:
        mappings.validate_checkpoint(broken)
    except ValueError as exc:
        assert "node_mapping" in str(exc)
    else:
        raise AssertionError("错误节点映射没有被拒绝")


def test_global_checkpoint_metadata_requires_exact_mapping_and_scalers():
    mappings = _mappings()
    dataset = MultiNodeDataset(
        {
            ("ERCOT", "LZ_LCRA"): _TinyNodeDataset(3, 20.0),
            ("CAISO", "TH_NP15_GEN-APND"): _TinyNodeDataset(2, 10.0),
        },
        mappings,
    )
    config = PilotConfig(
        use_deviation_penalty=True,
        global_use_entity_embeddings=True,
        global_num_nodes=2,
        global_num_markets=2,
    )

    metadata = global_checkpoint_metadata(config, dataset)
    assert metadata["contract_hash"]
    assert metadata["entity_mapping_hash"] == mappings.mapping_hash
    assert set(metadata["norm_stats_by_node"]) == {
        "LZ_LCRA", "TH_NP15_GEN-APND"
    }

    config.global_num_nodes = 3
    try:
        global_checkpoint_metadata(config, dataset)
    except ValueError as exc:
        assert "global_num_nodes" in str(exc)
    else:
        raise AssertionError("错误节点数没有被拒绝")


def test_universal_checkpoint_metadata_has_no_predictive_entity_dependency():
    mappings = _mappings()
    dataset = MultiNodeDataset(
        {
            ("ERCOT", "LZ_LCRA"): _TinyNodeDataset(3, 20.0),
            ("CAISO", "TH_NP15_GEN-APND"): _TinyNodeDataset(2, 10.0),
        },
        mappings,
    )
    config = PilotConfig(
        use_deviation_penalty=True,
        global_use_entity_embeddings=False,
        global_num_nodes=0,
        global_num_markets=0,
        global_normalization_mode="context_robust",
    )

    metadata = global_checkpoint_metadata(config, dataset)
    assert metadata["entity_conditioning"] == {
        "enabled": False,
        "uses_node_id_for_prediction": False,
        "uses_market_id_for_prediction": False,
    }
    assert metadata["normalization_mode"] == "context_robust"
