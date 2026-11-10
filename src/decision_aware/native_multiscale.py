"""Pure-v4 causal multiscale residual decoder.

The module uses only the five streams already visible to v4, three frozen v4
DA members, and a deterministic lag-168 anchor.  All prices are normalized by
the causal 168-hour context statistics before entering the decoder.
"""
from __future__ import annotations

import torch
from torch import nn


class PureV4MultiscaleDecoder(nn.Module):
    """Seven-day patch encoder with target-hour queries and convex experts."""

    def __init__(
        self,
        *,
        context_channels: int = 13,
        expert_count: int = 4,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        residual_bound: float = 2.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model必须能被n_heads整除")
        self.context_channels = int(context_channels)
        self.expert_count = int(expert_count)
        self.residual_bound = float(residual_bound)
        self.patch_projection = nn.Sequential(
            nn.Linear(24 * context_channels, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.aligned_day_projection = nn.Sequential(
            nn.Linear(24, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.day_position = nn.Parameter(torch.zeros(1, 7, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.target_calendar = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.aligned_hour_projection = nn.Sequential(
            nn.Linear(7, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.register_buffer(
            "weather_center",
            torch.tensor([15.0, 50.0, 10.0, 50.0, 0.0, 5.0, 10.0, 200.0]),
        )
        self.register_buffer(
            "weather_scale",
            torch.tensor([20.0, 50.0, 20.0, 50.0, 5.0, 10.0, 15.0, 400.0]),
        )
        self.weather_projection = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.fundamental_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3),
        )
        # Target weather is used only to infer the three physical fundamentals.
        # The price head itself consumes those fundamentals, not a market-
        # specific weather embedding.  This gives deployment a clean switch:
        # source markets can infer load/wind/solar from their weather forecasts,
        # while Austria can inject its official day-ahead physical forecasts.
        head_input = 2 * d_model + expert_count + 6
        self.shared_head = nn.Sequential(
            nn.Linear(head_input, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(d_model, expert_count)
        self.residual = nn.Linear(d_model, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        context: torch.Tensor,
        aligned_da_history: torch.Tensor,
        target_calendar: torch.Tensor,
        future_weather: torch.Tensor,
        experts: torch.Tensor,
        price_mean: torch.Tensor,
        price_std: torch.Tensor,
        known_fundamentals: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 3 or context.shape[1:] != (
            168, self.context_channels
        ):
            raise ValueError(
                f"context必须是[B,168,{self.context_channels}]，实际{tuple(context.shape)}"
            )
        if aligned_da_history.ndim != 3 or aligned_da_history.shape[1:] != (7, 24):
            raise ValueError("aligned_da_history必须是[B,7,24]")
        if target_calendar.ndim != 3 or target_calendar.shape[1:] != (24, 8):
            raise ValueError("target_calendar必须是[B,24,8]")
        if experts.ndim != 3 or experts.shape[1:] != (24, self.expert_count):
            raise ValueError(
                f"experts必须是[B,24,{self.expert_count}]"
            )
        if future_weather.ndim != 3 or future_weather.shape[1:] != (24, 8):
            raise ValueError("future_weather必须是[B,24,8]")
        if known_fundamentals is not None and (
            known_fundamentals.ndim != 3
            or known_fundamentals.shape[1:] != (24, 3)
        ):
            raise ValueError("known_fundamentals必须是[B,24,3]")
        mean = price_mean.reshape(-1, 1, 1)
        std = price_std.reshape(-1, 1, 1).clamp_min(1.0e-4)
        normalized_experts = (experts - mean) / std

        patches = context.reshape(
            context.shape[0], 7, 24 * self.context_channels
        )
        memory = self.encoder(
            self.patch_projection(patches)
            + self.aligned_day_projection(aligned_da_history)
            + self.day_position
        )
        query = (
            self.target_calendar(target_calendar)
            + self.aligned_hour_projection(aligned_da_history.transpose(1, 2))
        )
        weather = self.weather_projection(
            (future_weather - self.weather_center) / self.weather_scale
        )
        attended, attention = self.cross_attention(
            query, memory, memory, need_weights=True
        )
        predicted_fundamentals = self.fundamental_head(torch.cat([
            query, attended, weather
        ], dim=-1))
        fundamentals = (
            predicted_fundamentals
            if known_fundamentals is None else known_fundamentals
        )
        disagreement = normalized_experts.std(dim=-1, keepdim=True)
        seasonal_gap = (
            normalized_experts[..., -1:] - normalized_experts[..., :3].mean(
                dim=-1, keepdim=True
            )
        )
        v4_range = (
            normalized_experts[..., :3].amax(dim=-1, keepdim=True)
            - normalized_experts[..., :3].amin(dim=-1, keepdim=True)
        )
        hidden = self.shared_head(torch.cat([
            query,
            attended,
            normalized_experts,
            fundamentals,
            disagreement,
            seasonal_gap,
            v4_range,
        ], dim=-1))
        weights = torch.softmax(self.gate(hidden), dim=-1)
        convex = (weights * normalized_experts).sum(dim=-1)
        residual = self.residual_bound * torch.tanh(
            self.residual(hidden).squeeze(-1)
        )
        normalized_prediction = convex + residual
        prediction = normalized_prediction * std.squeeze(-1) + mean.squeeze(-1)
        return {
            "prediction": prediction,
            "normalized_prediction": normalized_prediction,
            # Exposed for source-market domain generalization.  The deployment
            # path ignores this tensor; training may attach an adversarial
            # market classifier without changing forecast semantics.
            "hidden": hidden,
            "weights": weights,
            "normalized_residual": residual,
            "day_attention": attention,
            "predicted_fundamentals": predicted_fundamentals,
            "used_fundamentals": fundamentals,
        }
