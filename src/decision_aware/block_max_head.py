"""Distributional daily block-maximum head for a frozen or adapted TSFM."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class DailyBlockMaxQuantileHead(nn.Module):
    """Correct baseline block-max quantiles from forecast-token embeddings.

    The zero-initialized output makes the initial head exactly reproduce the
    baseline block-max distribution.  ``cummax`` enforces non-crossing output
    quantiles while retaining a direct residual interpretation.
    """

    def __init__(
        self,
        *,
        model_dim: int,
        num_output_patches: int,
        num_quantiles: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = model_dim * num_output_patches + num_quantiles
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_quantiles),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        forecast_embeddings: torch.Tensor,
        baseline_quantiles: torch.Tensor,
    ) -> torch.Tensor:
        if forecast_embeddings.ndim != 3:
            raise ValueError("forecast_embeddings must be [batch, patch, model_dim]")
        if baseline_quantiles.ndim != 2:
            raise ValueError("baseline_quantiles must be [batch, quantile]")
        features = torch.cat(
            [forecast_embeddings.flatten(start_dim=1), baseline_quantiles], dim=1
        )
        corrected = baseline_quantiles + self.network(features)
        return torch.cummax(corrected, dim=1).values


class DailyExtremeEventHead(nn.Module):
    """Bernoulli occurrence head sharing the block-max head's inputs."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_output_patches: int,
        num_quantiles: int,
        hidden_dim: int = 128,
        initial_probability: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_probability < 1.0:
            raise ValueError("initial_probability must lie strictly between zero and one")
        input_dim = model_dim * num_output_patches + num_quantiles
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(
            final.bias,
            float(np.log(initial_probability / (1.0 - initial_probability))),
        )

    def forward(
        self,
        forecast_embeddings: torch.Tensor,
        baseline_quantiles: torch.Tensor,
    ) -> torch.Tensor:
        if forecast_embeddings.ndim != 3 or baseline_quantiles.ndim != 2:
            raise ValueError("expected [batch, patch, model_dim] and [batch, quantile]")
        features = torch.cat(
            [forecast_embeddings.flatten(start_dim=1), baseline_quantiles], dim=1
        )
        return self.network(features).squeeze(1)


class MonotoneLogitCalibrator(nn.Module):
    """Two-parameter monotone Platt calibration for rare-event probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.log_slope = nn.Parameter(torch.zeros(()))
        self.intercept = nn.Parameter(torch.zeros(()))

    def forward(self, raw_probabilities: torch.Tensor) -> torch.Tensor:
        probabilities = raw_probabilities.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(probabilities)
        return torch.sigmoid(torch.exp(self.log_slope) * logits + self.intercept)


def gumbel_parameters_from_quantiles(
    quantile_predictions: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit Gumbel location/scale from the nearest median and 0.9 quantiles."""

    levels = torch.as_tensor(
        quantile_levels,
        dtype=quantile_predictions.dtype,
        device=quantile_predictions.device,
    )
    if quantile_predictions.ndim != 2 or quantile_predictions.shape[1] != levels.numel():
        raise ValueError("quantile_predictions must be [batch, quantile]")
    low_index = int(torch.argmin(torch.abs(levels - 0.5)))
    high_index = int(torch.argmin(torch.abs(levels - 0.9)))
    low_level, high_level = levels[low_index], levels[high_index]
    low_z = -torch.log(-torch.log(low_level))
    high_z = -torch.log(-torch.log(high_level))
    scale = (
        (quantile_predictions[:, high_index] - quantile_predictions[:, low_index])
        / (high_z - low_z)
    ).clamp_min(1e-3)
    location = quantile_predictions[:, low_index] - scale * low_z
    return location, scale


class DailyBlockMaxGumbelHead(nn.Module):
    """Conditional Gumbel correction over a baseline block-max distribution."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_output_patches: int,
        num_quantiles: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = model_dim * num_output_patches + num_quantiles
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        forecast_embeddings: torch.Tensor,
        baseline_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            [forecast_embeddings.flatten(start_dim=1), baseline_quantiles], dim=1
        )
        baseline_location, baseline_scale = gumbel_parameters_from_quantiles(
            baseline_quantiles, quantile_levels
        )
        residual = self.network(features)
        location = baseline_location + residual[:, 0]
        scale = baseline_scale * torch.exp(residual[:, 1].clamp(-5.0, 5.0))
        return location, scale.clamp_min(1e-4)


def gumbel_negative_log_likelihood(
    location: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if location.shape != scale.shape or location.shape != target.shape:
        raise ValueError("location, scale and target must have identical shapes")
    standardized = ((target - location) / scale).clamp(-30.0, 30.0)
    return (torch.log(scale) + standardized + torch.exp(-standardized)).mean()


def gumbel_quantiles(
    location: torch.Tensor,
    scale: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    levels = torch.as_tensor(
        quantile_levels, dtype=location.dtype, device=location.device
    )
    if torch.any((levels <= 0) | (levels >= 1)):
        raise ValueError("Gumbel quantile levels must lie inside (0, 1)")
    return location[:, None] - scale[:, None] * torch.log(-torch.log(levels))[None, :]


def extend_upper_quantiles(
    hourly_quantiles: torch.Tensor,
    source_levels: torch.Tensor | Sequence[float],
    target_levels: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Form baseline daily-max quantiles, extending above the source grid.

    Existing source levels use the corresponding maximum-over-hours curve.
    Higher levels start from the highest available baseline quantile; the
    learned residual head is responsible for identifying additional tail mass.
    """

    if hourly_quantiles.ndim != 3:
        raise ValueError("hourly_quantiles must be [batch, quantile, horizon]")
    source = torch.as_tensor(
        source_levels, dtype=hourly_quantiles.dtype, device=hourly_quantiles.device
    )
    target = torch.as_tensor(
        target_levels, dtype=hourly_quantiles.dtype, device=hourly_quantiles.device
    )
    if source.ndim != 1 or source.numel() != hourly_quantiles.shape[1]:
        raise ValueError("source_levels must match the hourly quantile dimension")
    if target.ndim != 1 or torch.any(target[1:] <= target[:-1]):
        raise ValueError("target_levels must be a strictly increasing vector")
    daily = hourly_quantiles.amax(dim=-1)
    indices = torch.searchsorted(source, target, right=False).clamp(max=source.numel() - 1)
    return daily.index_select(1, indices)


def select_delivery_patch_embeddings(
    forecast_embeddings: torch.Tensor,
    offsets: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    """Select patch tokens overlapping each sample's 24-hour delivery day."""

    if forecast_embeddings.ndim != 3 or offsets.ndim != 1:
        raise ValueError("forecast_embeddings and offsets must be [B,P,D] and [B]")
    if forecast_embeddings.shape[0] != offsets.shape[0] or patch_size < 1:
        raise ValueError("incompatible batch sizes or invalid patch_size")
    start = torch.div(offsets, patch_size, rounding_mode="floor")
    end = torch.div(offsets + 23, patch_size, rounding_mode="floor")
    widths = end - start + 1
    if torch.any(widths != widths[0]):
        raise ValueError("delivery windows span different numbers of patches")
    relative = torch.arange(int(widths[0]), device=forecast_embeddings.device)
    indices = start[:, None] + relative[None, :]
    if int(indices.max()) >= forecast_embeddings.shape[1]:
        raise ValueError("delivery patch index exceeds forecast embeddings")
    batch = torch.arange(forecast_embeddings.shape[0], device=forecast_embeddings.device)
    return forecast_embeddings[batch[:, None], indices]


def scalar_quantile_score(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Mean doubled pinball score for scalar block-max quantiles."""

    if predictions.ndim != 2 or target.ndim != 1:
        raise ValueError("predictions and target must be [batch, quantile] and [batch]")
    levels = torch.as_tensor(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    )
    if levels.numel() != predictions.shape[1]:
        raise ValueError("quantile_levels must match predictions")
    error = target[:, None] - predictions
    return (2.0 * torch.maximum(levels * error, (levels - 1.0) * error)).mean()


def select_oof_tail_shrinkage(
    baseline: np.ndarray,
    proposed: np.ndarray,
    target: np.ndarray,
    quantile_levels: np.ndarray | Sequence[float],
    *,
    minimum_quantile: float = 0.9,
) -> tuple[float, list[dict[str, float]]]:
    """Select a convex tail correction weight using only OOF proper scores."""

    base = np.asarray(baseline, dtype=np.float32)
    candidate = np.asarray(proposed, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    levels = np.asarray(quantile_levels, dtype=np.float32)
    if base.shape != candidate.shape or base.shape != (len(truth), len(levels)):
        raise ValueError("OOF baseline, proposal, target and quantiles are incompatible")
    keep = levels >= float(minimum_quantile)
    if not np.any(keep):
        raise ValueError("minimum_quantile excludes every forecast quantile")
    audit: list[dict[str, float]] = []
    for alpha in np.linspace(0.0, 1.0, 21):
        prediction = base[:, keep] + float(alpha) * (
            candidate[:, keep] - base[:, keep]
        )
        error = truth[:, None] - prediction
        selected_levels = levels[keep][None, :]
        score = 2.0 * np.maximum(
            selected_levels * error, (selected_levels - 1.0) * error
        )
        audit.append({"alpha": float(alpha), "upper_tail_pinball": float(score.mean())})
    best = min(audit, key=lambda item: (item["upper_tail_pinball"], item["alpha"]))
    return best["alpha"], audit


def online_conformalize_block_quantiles(
    predictions: np.ndarray,
    target: np.ndarray,
    quantile_levels: np.ndarray | Sequence[float],
    *,
    num_entities: int,
    window_days: int = 90,
    minimum_history_days: int = 30,
) -> np.ndarray:
    """Prequential rolling quantile calibration with no future-target access.

    Samples must be concatenated entity by entity and have equal day counts.
    The correction for day ``t`` uses residuals through ``t-1`` only.
    """

    forecast = np.asarray(predictions, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    levels = np.asarray(quantile_levels, dtype=np.float32)
    if forecast.shape != (len(truth), len(levels)):
        raise ValueError("predictions, target and quantile levels are incompatible")
    if num_entities < 1 or len(truth) % num_entities != 0:
        raise ValueError("samples must divide evenly across entities")
    if window_days < 1 or minimum_history_days < 1 or minimum_history_days > window_days:
        raise ValueError("invalid rolling calibration window")
    days = len(truth) // num_entities
    calibrated = forecast.copy()
    for entity in range(num_entities):
        start = entity * days
        stop = start + days
        entity_forecast = forecast[start:stop]
        entity_truth = truth[start:stop]
        residuals = entity_truth[:, None] - entity_forecast
        for day in range(minimum_history_days, days):
            history_start = max(0, day - window_days)
            history = residuals[history_start:day]
            corrections = np.asarray(
                [
                    np.quantile(history[:, index], float(level))
                    for index, level in enumerate(levels)
                ],
                dtype=np.float32,
            )
            calibrated[start + day] = entity_forecast[day] + corrections
    return np.maximum.accumulate(calibrated, axis=1)


def event_probability_from_scalar_quantiles(
    predictions: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
    threshold: float,
    *,
    temperature: float = 5.0,
) -> torch.Tensor:
    """Smooth exceedance probability from a scalar quantile function."""

    levels = torch.as_tensor(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    )
    if predictions.ndim != 2 or predictions.shape[1] != levels.numel():
        raise ValueError("predictions must be [batch, quantile]")
    if levels.numel() < 2:
        raise ValueError("at least two quantiles are required")
    edges = torch.cat(
        [
            torch.zeros(1, dtype=levels.dtype, device=levels.device),
            (levels[:-1] + levels[1:]) / 2.0,
            torch.ones(1, dtype=levels.dtype, device=levels.device),
        ]
    )
    masses = edges[1:] - edges[:-1]
    exceedance = torch.sigmoid((predictions - float(threshold)) / float(temperature))
    return (exceedance * masses[None, :]).sum(dim=1)


def conditional_tail_mean_from_quantiles(
    predictions: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
    threshold: float,
    *,
    temperature: float = 5.0,
) -> torch.Tensor:
    """Smooth conditional mean above a threshold from scalar quantiles."""

    levels = torch.as_tensor(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    )
    if predictions.ndim != 2 or predictions.shape[1] != levels.numel():
        raise ValueError("predictions must be [batch, quantile]")
    edges = torch.cat(
        [
            torch.zeros(1, dtype=levels.dtype, device=levels.device),
            (levels[:-1] + levels[1:]) / 2.0,
            torch.ones(1, dtype=levels.dtype, device=levels.device),
        ]
    )
    masses = edges[1:] - edges[:-1]
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    membership = torch.sigmoid(
        (predictions - float(threshold)) / float(temperature)
    )
    tail_mass = (membership * masses[None, :]).sum(dim=1)
    tail_sum = (membership * predictions * masses[None, :]).sum(dim=1)
    fallback = torch.full_like(tail_sum, float(threshold))
    return torch.where(tail_mass > 0, tail_sum / tail_mass.clamp_min(1e-8), fallback)


def expected_tail_localized_adjustment(
    hourly_point: np.ndarray,
    predicted_tail_mean: np.ndarray,
    event_probability: np.ndarray,
) -> np.ndarray:
    """Distribute expected positive-tail contribution around the peak hour.

    The three-hour kernel sums to one.  Hence the total curve increment equals
    ``P(event) * max(E[max | event] - current_max, 0)`` for every day.
    """

    curves = np.asarray(hourly_point, dtype=np.float32)
    tail_mean = np.asarray(predicted_tail_mean, dtype=np.float32)
    probabilities = np.asarray(event_probability, dtype=np.float32)
    if curves.ndim != 2 or tail_mean.shape != (len(curves),) or probabilities.shape != (len(curves),):
        raise ValueError("incompatible hourly and tail prediction shapes")
    adjusted = curves.copy()
    kernel = ((-1, 0.35 / 1.70), (0, 1.0 / 1.70), (1, 0.35 / 1.70))
    for row in range(len(curves)):
        peak = int(np.argmax(curves[row]))
        expected_excess = float(probabilities[row]) * max(
            float(tail_mean[row] - curves[row, peak]), 0.0
        )
        for offset, weight in kernel:
            hour = peak + offset
            if 0 <= hour < curves.shape[1]:
                adjusted[row, hour] += expected_excess * weight
    return adjusted


def peak_localized_adjustment(
    hourly_point: np.ndarray,
    predicted_daily_max: np.ndarray,
    event_probability: np.ndarray,
    *,
    probability_floor: float = 0.1,
    probability_full: float = 1.0,
) -> np.ndarray:
    """Apply a probability-gated three-hour correction around the peak hour."""

    curves = np.asarray(hourly_point, dtype=np.float32)
    maxima = np.asarray(predicted_daily_max, dtype=np.float32)
    probabilities = np.asarray(event_probability, dtype=np.float32)
    if curves.ndim != 2 or maxima.shape != (len(curves),) or probabilities.shape != (len(curves),):
        raise ValueError("incompatible hourly and daily prediction shapes")
    adjusted = curves.copy()
    if probability_full <= probability_floor:
        raise ValueError("probability_full must exceed probability_floor")
    blend = np.clip(
        (probabilities - probability_floor) / (probability_full - probability_floor),
        0.0,
        1.0,
    )
    for row in range(len(curves)):
        peak = int(np.argmax(curves[row]))
        correction = float(maxima[row] - curves[row, peak]) * float(blend[row])
        for hour, weight in ((peak - 1, 0.35), (peak, 1.0), (peak + 1, 0.35)):
            if 0 <= hour < curves.shape[1]:
                adjusted[row, hour] += correction * weight
    return adjusted

