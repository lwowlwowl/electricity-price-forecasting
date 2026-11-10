#!/usr/bin/env python
"""Train and audit the source-only pure-v4 multiscale residual decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.native_multiscale import PureV4MultiscaleDecoder  # noqa: E402


SOURCE_MARKETS = ("CAISO", "NYISO", "PJM")


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float):
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


class MarketDiscriminator(nn.Module):
    def __init__(self, width: int, markets: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(width, markets),
        )

    def forward(self, hidden: torch.Tensor, strength: float) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        reversed_hidden = _GradientReversal.apply(pooled, strength)
        return self.network(reversed_hidden)


class CacheDataset(Dataset):
    def __init__(self, cache: dict[str, np.ndarray], day_indices: np.ndarray):
        self.cache = cache
        self.rows = np.asarray([
            (node, day)
            for node in range(cache["target"].shape[0])
            for day in day_indices
        ], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        node, day = self.rows[index]
        return {
            "context": torch.from_numpy(
                self.cache["context"][node, day].astype(np.float32)
            ),
            "calendar": torch.from_numpy(
                self.cache["calendar"][node, day].astype(np.float32)
            ),
            "aligned_da_history": torch.from_numpy(
                self.cache["aligned_da_history"][node, day].astype(np.float32)
            ),
            "future_weather": torch.from_numpy(
                self.cache["future_weather"][node, day].astype(np.float32)
            ),
            "fundamental_target": torch.from_numpy(
                self.cache["fundamental_target"][node, day].astype(np.float32)
            ),
            "experts": torch.from_numpy(self.cache["experts"][node, day]),
            "target": torch.from_numpy(self.cache["target"][node, day]),
            "mean": torch.tensor(self.cache["price_mean"][node, day]),
            "std": torch.tensor(self.cache["price_std"][node, day]),
            "node": torch.tensor(node),
        }


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _forward(
    model: PureV4MultiscaleDecoder,
    batch: dict[str, torch.Tensor],
    known_fundamentals: torch.Tensor | None = None,
):
    return model(
        batch["context"], batch["aligned_da_history"],
        batch["calendar"], batch["future_weather"],
        batch["experts"],
        batch["mean"], batch["std"],
        known_fundamentals=known_fundamentals,
    )


@torch.inference_mode()
def evaluate(model, datasets, device, batch_size: int) -> dict:
    model.eval()
    report = {}
    for market, dataset in datasets.items():
        errors, normalized_errors, weights, residuals, attention = [], [], [], [], []
        known_errors, known_normalized_errors = [], []
        fundamental_errors = []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for raw in loader:
            batch = _move(raw, device)
            output = _forward(model, batch)
            known_output = _forward(
                model, batch, known_fundamentals=batch["fundamental_target"]
            )
            error = (output["prediction"] - batch["target"]).abs()
            errors.append(error.cpu())
            normalized_errors.append(
                (error / batch["std"].reshape(-1, 1).clamp_min(1.0e-4)).cpu()
            )
            weights.append(output["weights"].cpu())
            residuals.append(output["normalized_residual"].cpu())
            attention.append(output["day_attention"].cpu())
            fundamental_errors.append(
                (output["predicted_fundamentals"] - batch["fundamental_target"])
                .abs().cpu()
            )
            known_error = (known_output["prediction"] - batch["target"]).abs()
            known_errors.append(known_error.cpu())
            known_normalized_errors.append(
                (
                    known_error
                    / batch["std"].reshape(-1, 1).clamp_min(1.0e-4)
                ).cpu()
            )
        error = torch.cat(errors)
        normalized = torch.cat(normalized_errors)
        report[market] = {
            "mae": float(error.mean()),
            "normalized_mae": float(normalized.mean()),
            "known_fundamentals_mae": float(torch.cat(known_errors).mean()),
            "known_fundamentals_normalized_mae": float(
                torch.cat(known_normalized_errors).mean()
            ),
            "expert_weights": torch.cat(weights).mean(dim=(0, 1)).tolist(),
            "mean_abs_normalized_residual": float(
                torch.cat(residuals).abs().mean()
            ),
            "day_attention": torch.cat(attention).mean(dim=(0, 1)).tolist(),
            "fundamental_normalized_mae": torch.cat(
                fundamental_errors
            ).mean(dim=(0, 1)).tolist(),
            "samples": int(len(dataset)),
        }
    report["market_macro"] = {
        "mae": float(np.mean([report[m]["mae"] for m in datasets])),
        "normalized_mae": float(np.mean([
            report[m]["normalized_mae"] for m in datasets
        ])),
        "worst_normalized_mae": float(max(
            report[m]["normalized_mae"] for m in datasets
        )),
        "known_fundamentals_mae": float(np.mean([
            report[m]["known_fundamentals_mae"] for m in datasets
        ])),
        "known_fundamentals_normalized_mae": float(np.mean([
            report[m]["known_fundamentals_normalized_mae"] for m in datasets
        ])),
        "known_fundamentals_worst_normalized_mae": float(max(
            report[m]["known_fundamentals_normalized_mae"] for m in datasets
        )),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", default="data/results/pure_v4_multiscale_cache_v1"
    )
    parser.add_argument("--cache-suffix", default="2023")
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument(
        "--output-dir", default="data/results/pure_v4_multiscale_decoder_v1"
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--final-all-markets",
        action="store_true",
        help="Freeze the selected structure and refit on all four source markets.",
    )
    parser.add_argument("--domain-generalization", action="store_true")
    parser.add_argument("--rex-weight", type=float, default=0.75)
    parser.add_argument("--domain-weight", type=float, default=0.20)
    parser.add_argument("--domain-strength", type=float, default=0.50)
    parser.add_argument("--groupdro-eta", type=float, default=0.10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache_dir = ROOT / args.cache_dir
    caches = {
        market: _load(cache_dir / f"{market.lower()}_{args.cache_suffix}.npz")
        for market in (*SOURCE_MARKETS, "ERCOT")
    }
    day_count = caches["CAISO"]["target"].shape[1]
    source_markets = (
        (*SOURCE_MARKETS, "ERCOT") if args.final_all_markets else SOURCE_MARKETS
    )
    split = max(1, day_count - args.validation_days)
    train_days = np.arange(day_count) if args.final_all_markets else np.arange(0, split)
    validation_days = np.arange(split, day_count)
    train_sets = {
        market: CacheDataset(caches[market], train_days)
        for market in source_markets
    }
    validation_sets = {
        market: CacheDataset(caches[market], validation_days)
        for market in SOURCE_MARKETS
    } if not args.final_all_markets else {}
    holdout_set = CacheDataset(caches["ERCOT"], np.arange(day_count))
    loaders = {
        market: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        for market, dataset in train_sets.items()
    }

    model = PureV4MultiscaleDecoder().to(device)
    discriminator = MarketDiscriminator(96, len(source_markets)).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=args.learning_rate, weight_decay=1.0e-3
    )
    group_weights = torch.full(
        (len(source_markets),), 1.0 / len(source_markets), device=device
    )
    best = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        iterators = {market: iter(loader) for market, loader in loaders.items()}
        steps = min(len(loader) for loader in loaders.values())
        epoch_losses = []
        for _ in range(steps):
            market_losses = []
            domain_logits = []
            domain_labels = []
            for market_index, market in enumerate(source_markets):
                batch = _move(next(iterators[market]), device)
                output = _forward(model, batch)
                # Source-market target fundamentals are used only as a noisy
                # surrogate for an operational day-ahead physical forecast.
                # Price labels and target-market values are never injected.
                noisy_fundamentals = (
                    batch["fundamental_target"]
                    + 0.10 * torch.randn_like(batch["fundamental_target"])
                )
                known_output = _forward(
                    model, batch, known_fundamentals=noisy_fundamentals
                )
                normalized = (
                    (output["prediction"] - batch["target"]).abs()
                    / batch["std"].reshape(-1, 1).clamp_min(1.0e-4)
                )
                known_normalized = (
                    (known_output["prediction"] - batch["target"]).abs()
                    / batch["std"].reshape(-1, 1).clamp_min(1.0e-4)
                )
                per_sample = 0.35 * normalized.mean(dim=1) + 0.65 * known_normalized.mean(dim=1)
                tail_count = max(1, int(np.ceil(0.2 * len(per_sample))))
                market_losses.append(
                    per_sample.mean()
                    + 0.20 * torch.topk(per_sample, tail_count).values.mean()
                    + 0.10 * (
                        output["predicted_fundamentals"]
                        - batch["fundamental_target"]
                    ).abs().mean()
                )
                if args.domain_generalization:
                    domain_logits.append(discriminator(
                        known_output["hidden"], args.domain_strength
                    ))
                    domain_labels.append(torch.full(
                        (len(batch["target"]),), market_index,
                        dtype=torch.long, device=device,
                    ))
            stacked = torch.stack(market_losses)
            if args.domain_generalization:
                with torch.no_grad():
                    group_weights *= torch.exp(args.groupdro_eta * stacked.detach())
                    group_weights /= group_weights.sum()
                prediction_loss = (group_weights * stacked).sum()
                rex = torch.var(stacked, unbiased=False)
                domain_loss = F.cross_entropy(
                    torch.cat(domain_logits), torch.cat(domain_labels)
                )
                loss = (
                    prediction_loss
                    + args.rex_weight * rex
                    + args.domain_weight * domain_loss
                )
            else:
                # Historical market macro plus smooth worst-market surrogate.
                loss = stacked.mean() + 0.20 * torch.logsumexp(5.0 * stacked, 0) / 5.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        validation = (
            {}
            if args.final_all_markets
            else evaluate(model, validation_sets, device, batch_size=args.batch_size)
        )
        score = (
            -float(epoch)
            if args.final_all_markets
            else (
                validation["market_macro"]["known_fundamentals_normalized_mae"]
                + 0.25
                * validation["market_macro"][
                    "known_fundamentals_worst_normalized_mae"
                ]
            )
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "selection_score": score,
            "validation": validation,
            "group_weights": group_weights.detach().cpu().tolist(),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "validation": validation,
            }

    assert best is not None
    model.load_state_dict(best["state"], strict=True)
    # ERCOT is read exactly once after source-only model selection.  The final
    # all-market refit has no holdout because its structure and epoch count are
    # already frozen by the preceding run.
    holdout = (
        {}
        if args.final_all_markets
        else evaluate(model, {"ERCOT": holdout_set}, device, batch_size=args.batch_size)
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best["state"],
        "model_config": {
            "context_channels": 13,
            "expert_count": 4,
            "d_model": 96,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.1,
            "residual_bound": 2.0,
        },
        "selected_epoch": best["epoch"],
        "selection_score": best["score"],
        "source_validation": best["validation"],
        "ercot_holdout": holdout,
        "seed": args.seed,
        "training_markets": list(source_markets),
        "holdout_market": None if args.final_all_markets else "ERCOT",
        "final_all_markets_refit": bool(args.final_all_markets),
        "uses_exaa": False,
        "uses_external_model_predictions": False,
        "uses_austria_labels": False,
        "domain_generalization": bool(args.domain_generalization),
        "domain_generalization_config": {
            "rex_weight": args.rex_weight,
            "domain_weight": args.domain_weight,
            "domain_strength": args.domain_strength,
            "groupdro_eta": args.groupdro_eta,
            "final_group_weights": group_weights.detach().cpu().tolist(),
        },
    }
    torch.save(checkpoint, output_dir / "decoder.pt")
    summary = {key: value for key, value in checkpoint.items() if key != "model_state"}
    summary["history"] = history
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_epoch": best["epoch"], "holdout": holdout}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
