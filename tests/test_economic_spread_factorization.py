from __future__ import annotations

import torch
import torch.nn as nn

from src.decision_aware.economic_spread_factorization import (
    DA_HIGH,
    NEUTRAL,
    RT_HIGH,
    EconomicSpreadFactorizationHead,
    EconomicSpreadFactorizedTransformer,
    conditional_excess_target,
    economic_spread_factorization_loss,
    economic_spread_regime,
)


def test_regime_and_excess_use_open_six_dollar_boundary() -> None:
    spread = torch.tensor([-6.1, -6.0, 0.0, 6.0, 6.1])
    assert economic_spread_regime(spread).tolist() == [
        RT_HIGH,
        NEUTRAL,
        NEUTRAL,
        NEUTRAL,
        DA_HIGH,
    ]
    assert torch.allclose(
        conditional_excess_target(spread),
        torch.tensor([0.1, 0.0, 0.0, 0.0, 0.1]),
        atol=1.0e-6,
    )


def test_zero_initialized_head_is_exactly_no_op_for_hard_predictions() -> None:
    head = EconomicSpreadFactorizationHead(4, dropout=0.0)
    da_rep = torch.randn(2, 3, 4)
    rt_rep = torch.randn(2, 3, 4)
    da = torch.tensor([[20.0, 30.0, 7.0], [5.0, 15.0, 25.0]])
    rt = torch.tensor([[22.0, 18.0, 20.0], [5.5, 30.0, 10.0]])
    output = head(da_rep, rt_rep, da, rt)
    assert torch.allclose(output["hard_da_price"], da)
    assert torch.allclose(output["hard_rt_at_da_price"], rt)
    assert torch.allclose(output["hard_spread"], da - rt)
    assert torch.allclose(output["soft_da_price"], da)
    assert torch.allclose(output["soft_rt_at_da_price"], rt)
    assert torch.allclose(output["soft_spread"], da - rt)


def test_soft_reconstruction_preserves_level_and_deployment_keeps_da() -> None:
    head = EconomicSpreadFactorizationHead(2, dropout=0.0)
    da = torch.tensor([[10.0, 40.0]])
    rt = torch.tensor([[20.0, 10.0]])
    output = head(torch.randn(1, 2, 2), torch.randn(1, 2, 2), da, rt)
    expected_level = 0.5 * (da + rt)
    assert torch.allclose(
        0.5 * (output["soft_da_price"] + output["soft_rt_at_da_price"]),
        expected_level,
    )
    assert torch.allclose(output["hard_da_price"], da)
    assert torch.allclose(
        output["hard_rt_at_da_price"], da - output["hard_spread"]
    )


def test_forced_states_obey_direction_and_threshold() -> None:
    head = EconomicSpreadFactorizationHead(3, dropout=0.0)
    with torch.no_grad():
        head.regime_residual.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))
    output = head(
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.tensor([[20.0, 21.0]]),
        torch.tensor([[19.0, 22.0]]),
    )
    assert torch.all(output["regime_class"] == DA_HIGH)
    assert torch.all(output["factorized_spread"] >= 6.0)

    with torch.no_grad():
        head.regime_residual.bias.copy_(torch.tensor([-10.0, -10.0, 10.0]))
    output = head(
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.tensor([[20.0, 21.0]]),
        torch.tensor([[19.0, 22.0]]),
    )
    assert torch.all(output["regime_class"] == RT_HIGH)
    assert torch.all(output["factorized_spread"] <= -6.0)


def test_loss_is_finite_and_gradients_reach_both_task_representations() -> None:
    torch.manual_seed(7)
    head = EconomicSpreadFactorizationHead(5, dropout=0.0)
    da_rep = torch.randn(2, 4, 5, requires_grad=True)
    rt_rep = torch.randn(2, 4, 5, requires_grad=True)
    base_da = torch.tensor(
        [[10.0, 30.0, 5.0, 20.0], [10.0, 9.0, 8.0, 7.0]]
    )
    base_rt = torch.tensor(
        [[11.0, 20.0, 15.0, 19.0], [20.0, 9.0, 1.0, 7.0]]
    )
    output = head(da_rep, rt_rep, base_da, base_rt)
    true_da = base_da + torch.tensor(
        [[0.0, 5.0, -2.0, 1.0], [1.0, 0.0, 3.0, 0.0]]
    )
    true_rt = base_rt
    loss = economic_spread_factorization_loss(output, true_da, true_rt)
    assert torch.isfinite(loss.total)
    assert loss.tradeable_count > 0
    # 输出残差层故意零初始化以保证首个硬预测完全不改变基础模型。因此首步先
    # 更新输出层；第二步开始，梯度才应进入融合层及两路任务表征。
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss.total.backward()
    optimizer.step()
    da_rep.grad = None
    rt_rep.grad = None
    second_output = head(da_rep, rt_rep, base_da, base_rt)
    second_loss = economic_spread_factorization_loss(
        second_output, true_da, true_rt
    )
    second_loss.total.backward()
    assert da_rep.grad is not None and torch.isfinite(da_rep.grad).all()
    assert rt_rep.grad is not None and torch.isfinite(rt_rep.grad).all()
    assert float(da_rep.grad.abs().sum()) > 0.0
    assert float(rt_rep.grad.abs().sum()) > 0.0


def test_neutral_only_batch_has_zero_conditional_excess_loss() -> None:
    head = EconomicSpreadFactorizationHead(2, dropout=0.0)
    output = head(
        torch.zeros(1, 3, 2),
        torch.zeros(1, 3, 2),
        torch.tensor([[10.0, 11.0, 12.0]]),
        torch.tensor([[10.0, 10.0, 10.0]]),
    )
    loss = economic_spread_factorization_loss(
        output,
        torch.tensor([[10.0, 11.0, 12.0]]),
        torch.tensor([[10.0, 10.0, 10.0]]),
    )
    assert loss.tradeable_count == 0
    assert float(loss.conditional_excess.detach()) == 0.0


class _FakePriceModel(nn.Module):
    def __init__(self, key: str) -> None:
        super().__init__()
        self.decoder = nn.Linear(3, 3)
        self.price_head = nn.Linear(3, 1)
        self.key = key

    def forward(self, batch: dict) -> dict:
        representation = self.decoder(batch["features"])
        return {
            "rep": representation,
            self.key: self.price_head(representation).squeeze(-1),
        }


def test_wrapper_decoder_head_scope_opens_both_decoders_and_joint_head() -> None:
    model = EconomicSpreadFactorizedTransformer(
        _FakePriceModel("p_da"),
        _FakePriceModel("p_rt_at_da"),
        d_model=3,
        dropout=0.0,
    )
    model.set_trainable_scope("decoder_head")
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert any(name.startswith("da_model.decoder.") for name in trainable)
    assert any(
        name.startswith("rt_at_da_model.decoder.") for name in trainable
    )
    assert any(name.startswith("factorization_head.") for name in trainable)
    output = model({"features": torch.randn(2, 4, 3)})
    assert output["hard_da_price"].shape == (2, 4)
    assert output["hard_rt_at_da_price"].shape == (2, 4)

