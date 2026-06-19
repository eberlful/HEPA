from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class MockBatch:
    context: Tensor
    future: Tensor
    horizon: Tensor
    labels: Tensor


class MockEventSeries(Dataset[MockBatch]):
    """Synthetic multivariate episodes with event precursors.

    Each episode receives one event time. A smooth latent precursor starts before
    the event and is mixed into a subset of channels, giving the self-supervised
    and supervised stages a learnable temporal signal without external data.
    """

    def __init__(
        self,
        episodes: int,
        length: int,
        channels: int,
        context_length: int,
        max_horizon: int,
        samples: int,
        seed: int,
    ) -> None:
        self.context_length = context_length
        self.max_horizon = max_horizon
        self.samples = samples

        generator = torch.Generator().manual_seed(seed)
        time = torch.linspace(0, 1, length).view(1, length, 1)
        seasonal = torch.sin(2 * math.pi * time * torch.arange(1, channels + 1).view(1, 1, channels))
        noise = 0.08 * torch.randn(episodes, length, channels, generator=generator)
        self.series = 0.12 * seasonal.repeat(episodes, 1, 1) + noise

        low = context_length + max_horizon + 8
        high = length - max_horizon - 1
        if high <= low:
            raise ValueError("length must exceed context_length + 2 * max_horizon")

        self.event_times = torch.randint(low, high, (episodes,), generator=generator)
        ramp_width = max(max_horizon * 2, 16)
        channel_weights = torch.linspace(1.0, 0.25, channels).view(1, 1, channels)

        steps = torch.arange(length).view(1, length)
        distance = self.event_times.view(episodes, 1) - steps
        precursor = (1.0 - distance.float().clamp(0, ramp_width) / ramp_width).clamp(0, 1)
        precursor = precursor * (distance >= 0)
        self.series = self.series + 0.9 * precursor.unsqueeze(-1) * channel_weights

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> MockBatch:
        episode = index % self.series.shape[0]
        max_t = self.series.shape[1] - self.max_horizon - 1
        t = random.randint(self.context_length, max_t)
        horizon = int(round(math.exp(random.uniform(0.0, math.log(self.max_horizon)))))
        start = t - self.context_length
        future = self.series[episode, t : t + horizon]

        labels = torch.zeros(self.max_horizon)
        event_time = int(self.event_times[episode])
        if event_time > t:
            first_positive = event_time - t
            if first_positive <= self.max_horizon:
                labels[first_positive - 1 :] = 1.0

        return MockBatch(
            context=self.series[episode, start:t],
            future=future,
            horizon=torch.tensor(horizon, dtype=torch.long),
            labels=labels,
        )


def collate_mock(samples: list[MockBatch]) -> MockBatch:
    max_future = max(sample.future.shape[0] for sample in samples)
    channels = samples[0].context.shape[-1]
    futures = torch.zeros(len(samples), max_future, channels)
    for index, sample in enumerate(samples):
        futures[index, : sample.future.shape[0]] = sample.future
    return MockBatch(
        context=torch.stack([sample.context for sample in samples]),
        future=futures,
        horizon=torch.stack([sample.horizon for sample in samples]),
        labels=torch.stack([sample.labels for sample in samples]),
    )


class PatchEmbedding(nn.Module):
    def __init__(self, channels: int, patch_size: int, model_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(channels * patch_size, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, channels = x.shape
        pad = (-steps) % self.patch_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        patches = patches.transpose(2, 3).contiguous().view(batch, -1, channels * self.patch_size)
        return self.proj(patches)


class HEPAEncoder(nn.Module):
    def __init__(self, channels: int, patch_size: int, model_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.patch = PatchEmbedding(channels, patch_size, model_dim)
        self.positional = nn.Parameter(torch.randn(1, 512, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.pool_query = nn.Parameter(torch.randn(model_dim) * 0.02)

    def _instance_norm(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        if lengths is None:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
            return (x - mean) / std
        else:
            batch, steps, channels = x.shape
            steps_range = torch.arange(steps, device=x.device).unsqueeze(0)
            mask = (steps_range < lengths.unsqueeze(1)).unsqueeze(-1)
            
            x_masked = x * mask
            sum_x = x_masked.sum(dim=1, keepdim=True)
            lengths_expanded = lengths.view(-1, 1, 1).clamp_min(1)
            mean = sum_x / lengths_expanded
            
            sq_diff = ((x_masked - mean) * mask).pow(2).sum(dim=1, keepdim=True)
            divisor = (lengths_expanded - 1).clamp_min(1)
            variance = sq_diff / divisor
            std = variance.clamp_min(1e-5).sqrt()
            
            return ((x - mean) / std) * mask

    def _encode_tokens(self, x: Tensor, causal: bool, lengths: Tensor | None = None) -> Tensor:
        tokens = self.patch(self._instance_norm(x, lengths))
        if tokens.shape[1] > self.positional.shape[1]:
            raise ValueError("increase positional embedding length for this context")
        tokens = tokens + self.positional[:, : tokens.shape[1]]
        mask = None
        key_padding_mask = None
        
        if lengths is not None:
            batch, num_patches, _ = tokens.shape
            patch_indices = torch.arange(num_patches, device=tokens.device).unsqueeze(0)
            key_padding_mask = (patch_indices * self.patch.patch_size) >= lengths.unsqueeze(1)
            
        if causal:
            length = tokens.shape[1]
            mask = torch.full((length, length), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            
        return self.transformer(tokens, mask=mask, src_key_padding_mask=key_padding_mask)

    def encode_context(self, x: Tensor) -> Tensor:
        tokens = self._encode_tokens(x, causal=True)
        return self.norm(tokens[:, -1])

    def encode_interval(self, x: Tensor, lengths: Tensor) -> Tensor:
        tokens = self._encode_tokens(x, causal=False, lengths=lengths)
        scores = tokens @ self.pool_query
        
        batch, num_patches, _ = tokens.shape
        patch_indices = torch.arange(num_patches, device=tokens.device).unsqueeze(0)
        padding_mask = (patch_indices * self.patch.patch_size) >= lengths.unsqueeze(1)
        
        scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = scores.softmax(dim=1).unsqueeze(-1)
        
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.norm((tokens * weights).sum(dim=1))


class HorizonPredictor(nn.Module):
    def __init__(self, model_dim: int, max_horizon: int) -> None:
        super().__init__()
        self.horizon = nn.Embedding(max_horizon + 1, model_dim)
        self.net = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, context: Tensor, horizon: Tensor) -> Tensor:
        horizon = horizon.clamp(1, self.horizon.num_embeddings - 1)
        return self.net(torch.cat([context, self.horizon(horizon)], dim=-1))


class HEPA(nn.Module):
    def __init__(
        self,
        channels: int,
        max_horizon: int,
        patch_size: int = 16,
        model_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.encoder = HEPAEncoder(channels, patch_size, model_dim, layers, heads, dropout)
        self.predictor = HorizonPredictor(model_dim, max_horizon)
        self.event_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))

    def pretrain_loss(self, context: Tensor, future: Tensor, horizon: Tensor, sigreg_weight: float) -> Tensor:
        past = self.encoder.encode_context(context)
        pred = self.predictor(past, horizon)
        target = self.encoder.encode_interval(future, horizon)
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        prediction_loss = F.l1_loss(pred_norm, target_norm)
        return (1.0 - sigreg_weight) * prediction_loss + sigreg_weight * sigreg(pred)

    def event_cdf(self, context: Tensor) -> Tensor:
        past = self.encoder.encode_context(context)
        horizons = torch.arange(1, self.max_horizon + 1, device=context.device)
        past = past.unsqueeze(1).expand(-1, self.max_horizon, -1)
        pred = self.predictor(past.reshape(-1, past.shape[-1]), horizons.repeat(context.shape[0]))
        logits = self.event_head(pred).view(context.shape[0], self.max_horizon)
        hazards = logits.sigmoid().clamp(1e-5, 1 - 1e-5)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        return 1.0 - survival


def sigreg(z: Tensor) -> Tensor:
    z = F.normalize(z, dim=-1)
    centered = z - z.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, z.shape[0] - 1)
    identity = torch.eye(covariance.shape[0], device=z.device, dtype=z.dtype) / covariance.shape[0]
    return F.mse_loss(covariance, identity) + z.mean(dim=0).pow(2).mean()


def weighted_bce(pred: Tensor, target: Tensor) -> Tensor:
    pred = pred.clamp(1e-7, 1 - 1e-7)
    positives = target.sum().clamp_min(1.0)
    negatives = (target.numel() - target.sum()).clamp_min(1.0)
    pos_weight = (negatives / positives).detach()
    loss = F.binary_cross_entropy(pred, target, reduction="none")
    weights = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    return (loss * weights).mean()


def train_epoch(model: HEPA, loader: DataLoader[MockBatch], optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = model.pretrain_loss(
            batch.context.to(device),
            batch.future.to(device),
            batch.horizon.to(device),
            sigreg_weight=0.1,
        )
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / len(loader)


def finetune_epoch(model: HEPA, loader: DataLoader[MockBatch], optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            labels = batch.labels.to(device)
        pred = model.event_cdf(batch.context.to(device))
        loss = weighted_bce(pred, labels)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / len(loader)


@torch.no_grad()
def evaluate(model: HEPA, loader: DataLoader[MockBatch], device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0.0
    monotonic_errors = 0
    batches = 0
    for batch in loader:
        pred = model.event_cdf(batch.context.to(device))
        total += float(weighted_bce(pred, batch.labels.to(device)))
        monotonic_errors += int(((pred[:, 1:] - pred[:, :-1]) < -1e-6).any())
        batches += 1
    return total / batches, monotonic_errors / max(1, batches)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader[MockBatch], DataLoader[MockBatch]]:
    train_data = MockEventSeries(
        episodes=args.episodes,
        length=args.series_length,
        channels=args.channels,
        context_length=args.context_length,
        max_horizon=args.horizons,
        samples=args.train_samples,
        seed=args.seed,
    )
    val_data = MockEventSeries(
        episodes=max(8, args.episodes // 4),
        length=args.series_length,
        channels=args.channels,
        context_length=args.context_length,
        max_horizon=args.horizons,
        samples=args.val_samples,
        seed=args.seed + 1,
    )
    return (
        DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mock),
        DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mock),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HEPA on synthetic event-prediction data.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--series-length", type=int, default=512)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--horizons", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    train_loader, val_loader = build_loaders(args)
    model = HEPA(
        channels=args.channels,
        max_horizon=args.horizons,
        patch_size=args.patch_size,
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    pretrain_optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=3e-4,
        weight_decay=1e-2,
    )
    for epoch in range(1, args.pretrain_epochs + 1):
        loss = train_epoch(model, train_loader, pretrain_optimizer, device)
        print(f"pretrain epoch {epoch:03d} loss={loss:.4f}")

    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    finetune_optimizer = torch.optim.AdamW(
        list(model.predictor.parameters()) + list(model.event_head.parameters()),
        lr=1e-3,
        weight_decay=1e-2,
    )
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = finetune_epoch(model, train_loader, finetune_optimizer, device)
        val_loss, monotonic_error_rate = evaluate(model, val_loader, device)
        print(
            f"finetune epoch {epoch:03d} "
            f"train_bce={train_loss:.4f} val_bce={val_loss:.4f} "
            f"monotonic_error_rate={monotonic_error_rate:.3f}"
        )


if __name__ == "__main__":
    main()
