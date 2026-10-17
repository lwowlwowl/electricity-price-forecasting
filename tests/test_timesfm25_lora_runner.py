from __future__ import annotations

import numpy as np
import torch

from scripts.decision_aware.run_universal_lomo_timesfm25_lora import (
    _flatten_batch,
    _mean_prediction,
    task_loss,
)


def test_timesfm25_task_losses_are_finite_and_task_specific():
    prediction = torch.tensor([[0.0, 3.0]])
    target = torch.tensor([[1.0, 1.0]])
    mae = task_loss(prediction, target, "da")
    huber = task_loss(prediction, target, "rt_at_da")
    assert float(mae) == 1.5
    assert torch.isfinite(huber)
    assert float(huber) < float(mae)


def test_timesfm25_rt_day_batch_flattens_hours_without_changing_horizon():
    context = torch.from_numpy(np.zeros((2, 24, 168), dtype=np.float32))
    target = torch.from_numpy(np.zeros((2, 24, 4), dtype=np.float32))
    flat_context, flat_target = _flatten_batch(context, target)
    assert flat_context.shape == (48, 168)
    assert flat_target.shape == (48, 4)


def test_timesfm25_wrapper_left_pads_168_hours_to_patch_aligned_context():
    class Output:
        mean_predictions = torch.zeros((2, 32))

    class Model:
        def __call__(self, **kwargs):
            assert kwargs["past_values"].shape == (2, 168)
            assert kwargs["forecast_context_len"] == 192
            return Output()

    prediction = _mean_prediction(Model(), torch.zeros((2, 168)), 24)
    assert prediction.shape == (2, 24)
