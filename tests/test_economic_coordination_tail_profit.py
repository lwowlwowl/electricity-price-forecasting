from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.decision_aware.audit_economic_coordination_tail_profit import (
    _exclusive_masks,
)


def test_exclusive_tail_masks_partition_every_hour() -> None:
    hourly = {
        "realized_da_price": np.asarray([-1.0, 120.0, 50.0, 30.0, 20.0]),
        "realized_rt_price": np.asarray([5.0, 20.0, 0.0, 0.0, 19.0]),
    }
    thresholds = SimpleNamespace(
        da_upper_1pct=100.0,
        rt_upper_1pct=100.0,
        abs_spread_upper_1pct=40.0,
        abs_spread_upper_5pct=20.0,
    )
    masks = _exclusive_masks(hourly, thresholds)
    stacked = np.stack(list(masks.values()))
    assert np.all(stacked.sum(axis=0) == 1)
    assert masks["negative_price"][0]
    assert masks["positive_spike"][1]
    assert masks["absolute_spread_top_1pct"][2]
    assert masks["absolute_spread_top_5pct_remainder"][3]
    assert masks["ordinary"][4]
