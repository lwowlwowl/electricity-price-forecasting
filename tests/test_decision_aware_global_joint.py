from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_da import DecisionAwareDADataset  # noqa: E402
from decision_aware.dataset_rt import DecisionAwareRTDataset  # noqa: E402
from decision_aware.dataset_rt_da import DecisionAwareRTAtDADataset  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    EntityMappings,
    MultiNodeDataset,
    NodeBalancedSampler,
    RTDeliveryDayDataset,
)
from decision_aware.global_joint import (  # noqa: E402
    GlobalJointDecisionAwareDataset,
    NodeBalancedEpisodeBatchSampler,
)


def _frame(offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2024-12-15", "2025-01-15", freq="h", tz="UTC")
    local = index.tz_convert("America/Chicago")
    values = np.arange(len(index), dtype=np.float32) + offset
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()
    return pd.DataFrame({
        "price_da": 30.0 + values / 100.0,
        "price_rt": 35.0 + values / 90.0,
        "load": 1000.0 + values,
        "wind": 100.0 + values / 10.0,
        "solar": 50.0 + values / 20.0,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "is_weekend": (dow >= 5).astype(np.float32),
        "is_holiday": np.zeros(len(index), dtype=np.float32),
    }, index=index)


def _task_sets():
    cfg = PilotConfig(
        context_len=24,
        horizon_da=24,
        horizon_rt=4,
        train_stride=1,
        split_train_start="2025-01-02 00:00",
        split_train_end="2025-01-10 23:00",
    )
    mappings = EntityMappings(
        market_to_id={"ERCOT": 0},
        node_to_id={"NODE_A": 0, "NODE_B": 1},
        node_to_market={"NODE_A": "ERCOT", "NODE_B": "ERCOT"},
    )
    by_task = {"da": {}, "rt_at_da": {}, "rt": {}}
    for node, offset in (("NODE_A", 0.0), ("NODE_B", 100.0)):
        frame = _frame(offset)
        da = DecisionAwareDADataset(frame, cfg, "train")
        rt_da = DecisionAwareRTAtDADataset(frame, cfg, "train", da.norm_stats)
        rt = DecisionAwareRTDataset(frame, cfg, "train", da.norm_stats)
        entity = ("ERCOT", node)
        by_task["da"][entity] = da
        by_task["rt_at_da"][entity] = rt_da
        by_task["rt"][entity] = RTDeliveryDayDataset(rt)
    return tuple(
        MultiNodeDataset(by_task[task], mappings)
        for task in ("da", "rt_at_da", "rt")
    )


def test_global_joint_aligns_tasks_and_resets_state_per_node():
    dataset = GlobalJointDecisionAwareDataset(*_task_sets())

    assert dataset.entities == (("ERCOT", "NODE_A"), ("ERCOT", "NODE_B"))
    assert dataset.entity_lengths[0] == dataset.entity_lengths[1]
    assert dataset.entity_lengths[0] > 1
    first_a = dataset.sample_identity(0)
    second_a = dataset.sample_identity(1)
    first_b_index = dataset.offsets[1]
    first_b = dataset.sample_identity(first_b_index)
    assert first_a["starts_new_segment"] is True
    assert second_a["starts_new_segment"] is False
    assert first_b["starts_new_segment"] is True

    sample = dataset[first_b_index]
    assert sample["da"]["node_id"].item() == 1
    assert sample["rt_at_da"]["node_id"].item() == 1
    assert sample["rt"]["node_id"].item() == 1
    assert sample["rt"]["price_rt_tgt"].shape == (24, 4)
    assert len(dataset.local_timestamps(first_b_index)) == 24
    assert dataset.sample_identity(first_b_index)["delivery_date"] == (
        dataset.sample_identity(0)["delivery_date"]
    )


def test_global_joint_remains_compatible_with_node_balanced_sampler():
    dataset = GlobalJointDecisionAwareDataset(*_task_sets())
    sampler = NodeBalancedSampler(dataset, samples_per_node=4, seed=7)
    nodes = [dataset.sample_identity(index)["node"] for index in sampler]
    assert nodes.count("NODE_A") == 4
    assert nodes.count("NODE_B") == 4


def test_episode_sampler_is_node_balanced_and_keeps_days_consecutive():
    dataset = GlobalJointDecisionAwareDataset(*_task_sets())
    sampler = NodeBalancedEpisodeBatchSampler(
        dataset, episode_days=3, episodes_per_node=2, seed=11
    )
    batches = list(sampler)
    assert len(batches) == 4
    nodes = []
    for indices in batches:
        identities = [dataset.sample_identity(index) for index in indices]
        nodes.append(identities[0]["node"])
        assert len({item["node"] for item in identities}) == 1
        assert identities[0]["starts_new_segment"] in {True, False}
        assert all(not item["starts_new_segment"] for item in identities[1:])
        ordinals = [item["delivery_date"].toordinal() for item in identities]
        assert ordinals == list(range(ordinals[0], ordinals[0] + 3))
    assert nodes.count("NODE_A") == 2
    assert nodes.count("NODE_B") == 2
