from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.smoke_timesfm3_finetune import LoRALinear


def test_lora_linear_starts_as_exact_frozen_base_and_receives_gradients():
    torch.manual_seed(0)
    base = nn.Linear(5, 3)
    values = torch.randn(4, 5)
    expected = base(values).detach()
    adapter = LoRALinear(base, rank=2, alpha=4.0)

    actual = adapter(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in adapter.base.parameters())
    assert all(parameter.requires_grad for parameter in adapter.lora_down.parameters())
    assert all(parameter.requires_grad for parameter in adapter.lora_up.parameters())

    actual.square().mean().backward()
    assert adapter.lora_up.weight.grad is not None
    assert torch.isfinite(adapter.lora_up.weight.grad).all()
    assert torch.count_nonzero(adapter.lora_up.weight.grad) > 0


def test_lora_adapter_state_round_trip_is_exact():
    torch.manual_seed(1)
    base = nn.Linear(4, 2)
    first = LoRALinear(base, rank=2, alpha=2.0)
    with torch.no_grad():
        first.lora_up.weight.fill_(0.25)
        first.lora_down.weight.fill_(-0.5)
    state = first.adapter_state_dict()

    second_base = nn.Linear(4, 2)
    second_base.load_state_dict(base.state_dict())
    second = LoRALinear(second_base, rank=2, alpha=2.0)
    second.load_adapter_state_dict(state)
    values = torch.randn(3, 4)

    torch.testing.assert_close(first(values), second(values), rtol=0, atol=0)
