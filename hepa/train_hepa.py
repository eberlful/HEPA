"""Paper-oriented HEPA training harness.

This module implements the two-stage HEPA recipe described in
``paper/hepa.md``:

1. Self-supervised JEPA pretraining with a causal patch Transformer context
   encoder, a weight-shared bidirectional target path, a horizon-conditioned
   predictor, L1 latent prediction loss, and SIGReg.
2. Predictor finetuning with the encoder frozen, a shared hazard head, and a
   monotone discrete survival CDF over dense horizons.

The first supported data source is synthetic mock data. Real datasets should
plug into the same batch contract used by ``EventBatch``:

    context: (B, T, S)
    future:  (B, K, S)
    horizons: (K,)
    labels:  (B, K), where labels[:, k] = 1[event within horizon k + 1]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class HEPAConfig:
    seed: int = 0
    data: str = "mock"
    preset: str = "paper"
    device: str = "auto"

    channels: int = 9
    context_len: int = 512
    max_horizon: int = 200
    patch_size: int = 16

    d_model: int = 256
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    max_position_patches: int = 512

    pretrain_lr: float = 3e-4
    pretrain_weight_decay: float = 1e-2
    pretrain_epochs: int = 100
    finetune_lr: float = 1e-3
    finetune_weight_decay: float = 1e-2
    finetune_epochs: int = 50
    batch_size: int = 64
    patience: int = 10
    sigreg_alpha: float = 0.1
    sigreg_sketches: int = 16
    sigreg_sketch_dim: int = 32
    grad_clip: float = 1.0

    mock_train_episodes: int = 128
    mock_val_episodes: int = 32
    mock_test_episodes: int = 32
    mock_samples_per_epoch: int = 1024
    mock_series_len: int = 1024
    mock_precursor_strength: float = 1.0

    checkpoint_dir: str = "checkpoints/hepa"
    checkpoint: bool = True


@dataclass(frozen=True)
class EventBatch:
    """Batch contract for HEPA event prediction datasets."""

    context: Tensor
    future: Tensor
    horizons: Tensor
    labels: Tensor

    def to(self, device: torch.device) -> "EventBatch":
        return EventBatch(
            context=self.context.to(device),
            future=self.future.to(device),
            horizons=self.horizons.to(device),
            labels=self.labels.to(device),
        )


class MockEventDataset(Dataset[EventBatch]):
    """Synthetic multivariate episodes with event precursors.

    The generator creates smooth multi-channel time series, assigns one event
    time per episode, and injects a ramping precursor into all channels before
    the event. It is intentionally deterministic for a fixed split and seed.
    """

    def __init__(
        self,
        split: str,
        episodes: int,
        samples: int,
        series_len: int,
        channels: int,
        context_len: int,
        max_horizon: int,
        precursor_strength: float,
        seed: int,
    ) -> None:
        if series_len <= context_len + 2 * max_horizon + 8:
            raise ValueError("mock_series_len must exceed context_len + 2 * max_horizon + 8")

        self.split = split
        self.samples = samples
        self.context_len = context_len
        self.max_horizon = max_horizon
        self.horizons = torch.arange(1, max_horizon + 1, dtype=torch.long)

        split_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
        self.item_seed_offset = split_offset
        generator = torch.Generator().manual_seed(seed + split_offset)

        time = torch.linspace(0, 1, series_len).view(1, series_len, 1)
        freqs = torch.arange(1, channels + 1).view(1, 1, channels)
        seasonal = 0.15 * torch.sin(2 * math.pi * time * freqs)
        trend = 0.05 * torch.cos(2 * math.pi * time * (freqs + 1))
        noise = 0.08 * torch.randn(episodes, series_len, channels, generator=generator)
        self.series = seasonal.repeat(episodes, 1, 1) + trend.repeat(episodes, 1, 1) + noise

        low = context_len + max_horizon
        high = series_len - max_horizon - 1
        self.event_times = torch.randint(low, high, (episodes,), generator=generator)

        steps = torch.arange(series_len).view(1, series_len)
        distance = self.event_times.view(episodes, 1) - steps
        ramp_width = max(2 * max_horizon, 16)
        precursor = (1.0 - distance.float().clamp(0, ramp_width) / ramp_width).clamp(0, 1)
        precursor = precursor * (distance >= 0)
        channel_weights = torch.linspace(1.0, 0.3, channels).view(1, 1, channels)
        self.series = self.series + precursor_strength * precursor.unsqueeze(-1) * channel_weights

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> EventBatch:
        episode = index % self.series.shape[0]
        rng = random.Random(self.item_seed_offset + 1_000_003 * index)
        t = rng.randint(self.context_len, self.series.shape[1] - self.max_horizon - 1)

        context = self.series[episode, t - self.context_len : t]
        future = self.series[episode, t : t + self.max_horizon]

        labels = torch.zeros(self.max_horizon, dtype=torch.float32)
        event_time = int(self.event_times[episode])
        if event_time > t:
            first_positive = event_time - t
            if first_positive <= self.max_horizon:
                labels[first_positive - 1 :] = 1.0

        return EventBatch(
            context=context,
            future=future,
            horizons=self.horizons,
            labels=labels,
        )


def collate_event_batches(samples: list[EventBatch]) -> EventBatch:
    horizons = samples[0].horizons
    return EventBatch(
        context=torch.stack([sample.context for sample in samples]),
        future=torch.stack([sample.future for sample in samples]),
        horizons=horizons.clone(),
        labels=torch.stack([sample.labels for sample in samples]),
    )


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens + self.pe[:, : tokens.shape[1]]


class PatchTokenizer(nn.Module):
    def __init__(self, channels: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.proj = nn.Linear(channels * patch_size, d_model)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, steps, channels = x.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        pad = (-steps) % self.patch_size
        valid_steps = torch.full((batch,), steps, dtype=torch.long, device=x.device)
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        patches = x.unfold(1, self.patch_size, self.patch_size)
        patches = patches.transpose(2, 3).contiguous().view(batch, -1, channels * self.patch_size)
        valid_patches = torch.ceil(valid_steps.float() / self.patch_size).long().clamp_min(1)
        return self.proj(patches), valid_patches


class HEPAEncoder(nn.Module):
    """Shared Transformer used causally for context and bidirectionally for targets."""

    def __init__(self, cfg: HEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = PatchTokenizer(cfg.channels, cfg.patch_size, cfg.d_model)
        self.positional = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_position_patches)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.pool_query = nn.Parameter(torch.randn(cfg.d_model) * 0.02)

    def _instance_norm(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        return (x - mean) / std

    def _tokens(self, x: Tensor, causal: bool) -> tuple[Tensor, Tensor]:
        tokens, valid_patches = self.tokenizer(self._instance_norm(x))
        if tokens.shape[1] > self.cfg.max_position_patches:
            raise ValueError("increase max_position_patches for this sequence length")
        tokens = self.positional(tokens)

        patch_index = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        key_padding_mask = patch_index >= valid_patches.unsqueeze(1)

        attention_mask = None
        if causal:
            length = tokens.shape[1]
            attention_mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=tokens.device),
                diagonal=1,
            )

        encoded = self.transformer(
            tokens,
            mask=attention_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return encoded, valid_patches

    def encode_context(self, context: Tensor) -> Tensor:
        tokens, valid_patches = self._tokens(context, causal=True)
        last_index = (valid_patches - 1).view(-1, 1, 1).expand(-1, 1, tokens.shape[-1])
        return self.norm(tokens.gather(1, last_index).squeeze(1))

    def encode_future(self, future: Tensor, lengths: Tensor) -> Tensor:
        max_len = int(lengths.max().item())
        tokens, valid_patches = self._tokens(future[:, :max_len], causal=False)
        valid_patches = torch.ceil(lengths.float() / self.cfg.patch_size).long().clamp_min(1)
        patch_index = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        padding_mask = patch_index >= valid_patches.unsqueeze(1)
        scores = tokens @ self.pool_query
        scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = scores.softmax(dim=1).unsqueeze(-1)
        pooled = (tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0) * weights).sum(dim=1)
        return self.norm(pooled)


class HorizonPredictor(nn.Module):
    def __init__(self, cfg: HEPAConfig) -> None:
        super().__init__()
        self.horizon = nn.Embedding(cfg.max_horizon + 1, cfg.d_model)
        self.net = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, h_t: Tensor, horizons: Tensor) -> Tensor:
        horizons = horizons.clamp(1, self.horizon.num_embeddings - 1)
        return self.net(torch.cat([h_t, self.horizon(horizons)], dim=-1))


class EventHead(nn.Module):
    def __init__(self, cfg: HEPAConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))

    def forward(self, h_hat: Tensor) -> Tensor:
        return self.net(h_hat).squeeze(-1)


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularization on predictor outputs."""

    def __init__(self, d_model: int, n_sketches: int, sketch_dim: int) -> None:
        super().__init__()
        sketch = torch.randn(n_sketches, d_model, sketch_dim) / math.sqrt(d_model)
        self.register_buffer("sketch", sketch)

    def forward(self, z: Tensor) -> Tensor:
        batch = z.shape[0]
        sketches, _, sketch_dim = self.sketch.shape
        projected = torch.einsum("bd,sdk->bsk", z, self.sketch)
        mean = projected.mean(dim=0)
        centered = projected - mean.unsqueeze(0)
        cov = torch.einsum("bsk,bsl->skl", centered, centered) / max(batch - 1, 1)
        eye = torch.eye(sketch_dim, device=z.device, dtype=z.dtype).expand(sketches, -1, -1)
        std = projected.std(dim=0)
        return mean.square().mean() + (cov - eye).square().mean() + (std - 1.0).square().mean()


class HEPA(nn.Module):
    def __init__(self, cfg: HEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = HEPAEncoder(cfg)
        self.predictor = HorizonPredictor(cfg)
        self.event_head = EventHead(cfg)
        self.sigreg = SIGReg(cfg.d_model, cfg.sigreg_sketches, cfg.sigreg_sketch_dim)

    def pretrain_components(self, batch: EventBatch) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = batch.context.shape[0]
        horizons = sample_log_uniform_horizons(batch_size, self.cfg.max_horizon, batch.context.device)
        h_t = self.encoder.encode_context(batch.context)
        h_hat = self.predictor(h_t, horizons)
        h_star = self.encoder.encode_future(batch.future, horizons)
        latent_loss = F.l1_loss(F.normalize(h_hat, dim=-1), F.normalize(h_star, dim=-1))
        sigreg_loss = self.sigreg(h_hat)
        loss = (1.0 - self.cfg.sigreg_alpha) * latent_loss + self.cfg.sigreg_alpha * sigreg_loss
        return loss, latent_loss.detach(), sigreg_loss.detach()

    def probability_surface(self, context: Tensor, horizons: Tensor) -> Tensor:
        h_t = self.encoder.encode_context(context)
        return self.probability_surface_from_embedding(h_t, horizons)

    def probability_surface_from_embedding(self, h_t: Tensor, horizons: Tensor) -> Tensor:
        batch_size = h_t.shape[0]
        num_horizons = horizons.numel()
        h_expanded = h_t.unsqueeze(1).expand(-1, num_horizons, -1).reshape(batch_size * num_horizons, -1)
        horizon_expanded = horizons.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        h_hat = self.predictor(h_expanded, horizon_expanded)
        logits = self.event_head(h_hat).view(batch_size, num_horizons)
        hazards = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        return 1.0 - survival


def sample_log_uniform_horizons(batch_size: int, max_horizon: int, device: torch.device) -> Tensor:
    u = torch.rand(batch_size, device=device)
    return torch.exp(u * math.log(max_horizon)).floor().long().clamp(1, max_horizon)


def positive_weighted_bce(probabilities: Tensor, labels: Tensor) -> Tensor:
    probabilities = probabilities.clamp(1e-7, 1 - 1e-7)
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 1_000.0)
    loss = F.binary_cross_entropy(probabilities, labels, reduction="none")
    weights = torch.where(labels > 0.5, pos_weight.unsqueeze(0), torch.ones_like(labels))
    return (loss * weights).mean()


def h_auroc(probabilities: Tensor, labels: Tensor) -> tuple[float, int, int]:
    probs_np = probabilities.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    aucs: list[float] = []
    skipped = 0
    for index in range(labels_np.shape[1]):
        prevalence = float(labels_np[:, index].mean())
        if prevalence < 0.001 or prevalence > 0.999:
            skipped += 1
            continue
        aucs.append(float(roc_auc_score(labels_np[:, index], probs_np[:, index])))
    if not aucs:
        return float("nan"), 0, skipped
    return float(np.mean(aucs)), len(aucs), skipped


def monotonicity_violations(probabilities: Tensor) -> int:
    return int(((probabilities[:, 1:] - probabilities[:, :-1]) < -1e-6).sum().item())


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_mock_loaders(cfg: HEPAConfig) -> tuple[DataLoader[EventBatch], DataLoader[EventBatch], DataLoader[EventBatch]]:
    train = MockEventDataset(
        "train",
        cfg.mock_train_episodes,
        cfg.mock_samples_per_epoch,
        cfg.mock_series_len,
        cfg.channels,
        cfg.context_len,
        cfg.max_horizon,
        cfg.mock_precursor_strength,
        cfg.seed,
    )
    val = MockEventDataset(
        "val",
        cfg.mock_val_episodes,
        max(cfg.batch_size * 2, cfg.mock_val_episodes),
        cfg.mock_series_len,
        cfg.channels,
        cfg.context_len,
        cfg.max_horizon,
        cfg.mock_precursor_strength,
        cfg.seed,
    )
    test = MockEventDataset(
        "test",
        cfg.mock_test_episodes,
        max(cfg.batch_size * 2, cfg.mock_test_episodes),
        cfg.mock_series_len,
        cfg.channels,
        cfg.context_len,
        cfg.max_horizon,
        cfg.mock_precursor_strength,
        cfg.seed,
    )
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_event_batches, drop_last=True),
        DataLoader(val, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_event_batches),
        DataLoader(test, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_event_batches),
    )


def pretrain_epoch(model: HEPA, loader: DataLoader[EventBatch], optimizer: torch.optim.Optimizer, device: torch.device, cfg: HEPAConfig) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "latent": 0.0, "sigreg": 0.0}
    batches = 0
    for raw_batch in loader:
        batch = raw_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, latent, sigreg_loss = model.pretrain_components(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pretrain_parameters(model), cfg.grad_clip)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["latent"] += float(latent)
        totals["sigreg"] += float(sigreg_loss)
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_pretrain(model: HEPA, loader: DataLoader[EventBatch], device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "latent": 0.0, "sigreg": 0.0}
    batches = 0
    for raw_batch in loader:
        batch = raw_batch.to(device)
        loss, latent, sigreg_loss = model.pretrain_components(batch)
        totals["loss"] += float(loss)
        totals["latent"] += float(latent)
        totals["sigreg"] += float(sigreg_loss)
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def finetune_epoch(model: HEPA, loader: DataLoader[EventBatch], optimizer: torch.optim.Optimizer, device: torch.device, cfg: HEPAConfig) -> float:
    model.train()
    total = 0.0
    batches = 0
    for raw_batch in loader:
        batch = raw_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            h_t = model.encoder.encode_context(batch.context)
        probabilities = model.probability_surface_from_embedding(h_t, batch.horizons)
        loss = positive_weighted_bce(probabilities, batch.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(finetune_parameters(model), cfg.grad_clip)
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def evaluate_finetune(model: HEPA, loader: DataLoader[EventBatch], device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    all_probs = []
    all_labels = []
    for raw_batch in loader:
        batch = raw_batch.to(device)
        probabilities = model.probability_surface(batch.context, batch.horizons)
        losses.append(float(positive_weighted_bce(probabilities, batch.labels)))
        all_probs.append(probabilities.cpu())
        all_labels.append(batch.labels.cpu())
    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    auc, auc_horizons, skipped = h_auroc(probs, labels)
    return {
        "bce": float(np.mean(losses)) if losses else float("nan"),
        "h_auroc": auc,
        "auc_horizons": float(auc_horizons),
        "skipped_horizons": float(skipped),
        "monotonicity_violations": float(monotonicity_violations(probs)),
    }


def pretrain_parameters(model: HEPA) -> list[nn.Parameter]:
    return list(model.encoder.parameters()) + list(model.predictor.parameters()) + list(model.sigreg.parameters())


def finetune_parameters(model: HEPA) -> list[nn.Parameter]:
    return list(model.predictor.parameters()) + list(model.event_head.parameters())


def freeze_for_finetuning(model: HEPA) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.sigreg.parameters():
        parameter.requires_grad = False
    for parameter in model.predictor.parameters():
        parameter.requires_grad = True
    for parameter in model.event_head.parameters():
        parameter.requires_grad = True


def assert_architecture_contract(model: HEPA, cfg: HEPAConfig, device: torch.device) -> None:
    batch = EventBatch(
        context=torch.randn(2, cfg.context_len, cfg.channels, device=device),
        future=torch.randn(2, cfg.max_horizon, cfg.channels, device=device),
        horizons=torch.arange(1, cfg.max_horizon + 1, device=device),
        labels=torch.zeros(2, cfg.max_horizon, device=device),
    )
    h_t = model.encoder.encode_context(batch.context)
    assert h_t.shape == (2, cfg.d_model)
    h_hat = model.predictor(h_t, torch.ones(2, dtype=torch.long, device=device))
    assert h_hat.shape == (2, cfg.d_model)
    logits = model.event_head(h_hat)
    assert logits.shape == (2,)
    surface = model.probability_surface(batch.context, batch.horizons)
    assert surface.shape == (2, cfg.max_horizon)
    assert bool((surface[:, 1:] >= surface[:, :-1] - 1e-6).all())


def assert_metric_contract() -> None:
    labels = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    probabilities = torch.tensor(
        [
            [0.1, 0.2, 0.8],
            [0.1, 0.9, 0.9],
            [0.1, 0.3, 0.7],
            [0.1, 0.8, 0.95],
        ]
    )
    auc, used, skipped = h_auroc(probabilities, labels)
    assert used == 1
    assert skipped == 2
    assert abs(auc - 1.0) < 1e-9


def save_checkpoint(path: Path, model: HEPA, cfg: HEPAConfig, stage: str, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "epoch": epoch,
            "config": asdict(cfg),
            "metrics": metrics,
            "model": model.state_dict(),
        },
        path,
    )


def run_training(cfg: HEPAConfig) -> dict[str, float]:
    set_reproducibility(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.device = str(device)

    if cfg.data != "mock":
        raise ValueError("only --data mock is implemented; real loaders should implement EventBatch")

    train_loader, val_loader, test_loader = build_mock_loaders(cfg)
    model = HEPA(cfg).to(device)

    assert_architecture_contract(model, cfg, device)
    assert_metric_contract()

    print_run_header(cfg, model)

    pretrain_optimizer = torch.optim.AdamW(
        pretrain_parameters(model),
        lr=cfg.pretrain_lr,
        weight_decay=cfg.pretrain_weight_decay,
    )
    best_pretrain = float("inf")
    stale_epochs = 0
    for epoch in range(1, cfg.pretrain_epochs + 1):
        train_metrics = pretrain_epoch(model, train_loader, pretrain_optimizer, device, cfg)
        val_metrics = evaluate_pretrain(model, val_loader, device)
        print(
            f"[pretrain] epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_latent={train_metrics['latent']:.4f} "
            f"train_sigreg={train_metrics['sigreg']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        if val_metrics["loss"] < best_pretrain:
            best_pretrain = val_metrics["loss"]
            stale_epochs = 0
            if cfg.checkpoint:
                save_checkpoint(Path(cfg.checkpoint_dir) / "best_pretrain.pt", model, cfg, "pretrain", epoch, val_metrics)
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.patience:
            print(f"[pretrain] early_stop epoch={epoch:03d} patience={cfg.patience}")
            break

    freeze_for_finetuning(model)
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.predictor.parameters())
    assert all(parameter.requires_grad for parameter in model.event_head.parameters())

    finetune_optimizer = torch.optim.AdamW(
        finetune_parameters(model),
        lr=cfg.finetune_lr,
        weight_decay=cfg.finetune_weight_decay,
    )
    best_finetune = float("inf")
    stale_epochs = 0
    for epoch in range(1, cfg.finetune_epochs + 1):
        train_bce = finetune_epoch(model, train_loader, finetune_optimizer, device, cfg)
        val_metrics = evaluate_finetune(model, val_loader, device)
        h_auc = format_metric(val_metrics["h_auroc"])
        print(
            f"[finetune] epoch={epoch:03d} "
            f"train_bce={train_bce:.4f} "
            f"val_bce={val_metrics['bce']:.4f} "
            f"val_h_auroc={h_auc} "
            f"auc_horizons={int(val_metrics['auc_horizons'])} "
            f"skipped={int(val_metrics['skipped_horizons'])} "
            f"monotonicity_violations={int(val_metrics['monotonicity_violations'])}"
        )
        if val_metrics["bce"] < best_finetune:
            best_finetune = val_metrics["bce"]
            stale_epochs = 0
            if cfg.checkpoint:
                save_checkpoint(Path(cfg.checkpoint_dir) / "best_finetune.pt", model, cfg, "finetune", epoch, val_metrics)
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.patience:
            print(f"[finetune] early_stop epoch={epoch:03d} patience={cfg.patience}")
            break

    test_metrics = evaluate_finetune(model, test_loader, device)
    print_final_diagnostics(model, test_loader, test_metrics, device)
    return test_metrics


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")


def print_run_header(cfg: HEPAConfig, model: HEPA) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    pretrain_count = sum(parameter.numel() for parameter in pretrain_parameters(model))
    finetune_count = sum(parameter.numel() for parameter in finetune_parameters(model))
    print("== HEPA paper-reproduction harness ==")
    print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    print(f"parameters.total={total:,}")
    print(f"parameters.pretrain={pretrain_count:,}")
    print(f"parameters.finetune={finetune_count:,}")


@torch.no_grad()
def print_final_diagnostics(model: HEPA, loader: DataLoader[EventBatch], metrics: dict[str, float], device: torch.device) -> None:
    batch = next(iter(loader)).to(device)
    probabilities = model.probability_surface(batch.context, batch.horizons)
    sample = probabilities[0, torch.linspace(0, probabilities.shape[1] - 1, steps=min(8, probabilities.shape[1])).long()]
    print("[test] " + " ".join(f"{key}={format_metric(value)}" for key, value in metrics.items()))
    print(f"[test] survival_cdf_monotone={monotonicity_violations(probabilities) == 0}")
    print(f"[test] probability_surface_sample={np.array2string(sample.cpu().numpy(), precision=4)}")


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def apply_preset(args: argparse.Namespace) -> HEPAConfig:
    if args.preset == "smoke":
        cfg = HEPAConfig(
            preset="smoke",
            context_len=64,
            max_horizon=16,
            channels=6,
            d_model=32,
            layers=1,
            heads=4,
            batch_size=8,
            pretrain_epochs=1,
            finetune_epochs=1,
            patience=3,
            sigreg_sketches=4,
            sigreg_sketch_dim=8,
            mock_train_episodes=16,
            mock_val_episodes=8,
            mock_test_episodes=8,
            mock_samples_per_epoch=32,
            mock_series_len=160,
            checkpoint=False,
        )
    else:
        cfg = HEPAConfig(preset="paper")

    override_map = {
        "seed": "seed",
        "data": "data",
        "device": "device",
        "channels": "channels",
        "context_len": "context_len",
        "max_horizon": "max_horizon",
        "pretrain_epochs": "pretrain_epochs",
        "finetune_epochs": "finetune_epochs",
        "batch_size": "batch_size",
        "checkpoint_dir": "checkpoint_dir",
    }
    for arg_name, field_name in override_map.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, field_name, value)
    if args.no_checkpoint:
        cfg.checkpoint = False
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HEPA with a paper-reproduction harness.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data", choices=["mock"], default="mock")
    parser.add_argument("--preset", choices=["smoke", "paper"], default="paper")
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    cfg = apply_preset(parse_args())
    run_training(cfg)


if __name__ == "__main__":
    main()
