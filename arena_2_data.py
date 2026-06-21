"""
Self-contained PyTorch implementation of HEPA:
Horizon-conditioned Event Predictive Architecture.

This script uses synthetic FD001-like turbofan degradation data because the real
NASA C-MAPSS FD001 files are not bundled. The generated data mimics multichannel
sensor trajectories with gradual degradation and failure at the end of each engine
life.

Run:
    python hepa_demo.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    seed: int = 7

    # Synthetic FD001-like data
    num_engines: int = 28
    train_fraction: float = 0.8
    min_engine_length: int = 330
    max_engine_length: int = 430
    num_sensors: int = 14

    # HEPA / C-MAPSS-like setup
    context_length: int = 128
    max_horizon: int = 150  # HEPA uses dense C-MAPSS horizons; paper uses K=150.
    patch_size: int = 16

    # Model
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    horizon_emb_dim: int = 32
    predictor_hidden: int = 512

    # Training
    batch_size: int = 16
    pretrain_epochs: int = 2
    finetune_epochs: int = 3
    pretrain_lr: float = 3e-4
    finetune_lr: float = 1e-3
    weight_decay: float = 1e-2
    sigreg_alpha: float = 0.1

    # Dataset subsampling for speed
    sample_stride: int = 4


# -----------------------------
# Synthetic FD001-like data
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_synthetic_fd001_like_engines(cfg: Config) -> List[torch.Tensor]:
    """
    Generate FD001-like multivariate turbofan logs.

    Each engine has shape [T, S], where T is lifetime and S is number of sensors.
    Failure occurs immediately after the last observed cycle. Sensor channels
    combine operating-condition variation, slow degradation, nonlinear drift,
    random walk, and measurement noise.
    """
    g = torch.Generator().manual_seed(cfg.seed)
    engines: List[torch.Tensor] = []

    for _ in range(cfg.num_engines):
        T = int(torch.randint(cfg.min_engine_length, cfg.max_engine_length + 1, (1,), generator=g).item())
        S = cfg.num_sensors
        t = torch.arange(T, dtype=torch.float32)
        tau = t / max(T - 1, 1)

        # Degradation starts after a random healthy period.
        onset = 0.30 + 0.25 * torch.rand(1, generator=g).item()
        sharpness = 1.1 + 1.4 * torch.rand(1, generator=g).item()
        degradation = ((tau - onset) / max(1.0 - onset, 1e-6)).clamp(0.0, 1.0).pow(sharpness)

        # Smooth operating conditions.
        op1 = torch.sin(2.0 * math.pi * tau * (1.0 + 0.4 * torch.rand(1, generator=g).item()))
        op2 = torch.cos(2.0 * math.pi * tau * (0.5 + 0.5 * torch.rand(1, generator=g).item()))

        baseline = 0.2 * torch.randn(S, generator=g)
        op1_w = 0.25 * torch.randn(S, generator=g)
        op2_w = 0.20 * torch.randn(S, generator=g)

        # First sensors are more failure-informative; later sensors are weaker/noisier.
        informative_scale = torch.linspace(1.25, 0.25, S)
        deg_w = informative_scale * (0.8 + 0.5 * torch.randn(S, generator=g))
        quad_w = informative_scale * (0.5 + 0.3 * torch.randn(S, generator=g))

        noise_scale = 0.05 + 0.06 * tau.unsqueeze(-1) + 0.08 * degradation.unsqueeze(-1)
        measurement_noise = noise_scale * torch.randn(T, S, generator=g)

        random_walk = 0.01 * torch.randn(T, S, generator=g).cumsum(dim=0)

        x = (
            baseline.unsqueeze(0)
            + op1.unsqueeze(-1) * op1_w.unsqueeze(0)
            + op2.unsqueeze(-1) * op2_w.unsqueeze(0)
            + degradation.unsqueeze(-1) * deg_w.unsqueeze(0)
            + degradation.pow(2).unsqueeze(-1) * quad_w.unsqueeze(0)
            + random_walk
            + measurement_noise
        )

        engines.append(x.float())

    return engines


def fit_standardizer(engines: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit train-set z-score normalizer."""
    concat = torch.cat(engines, dim=0)
    mean = concat.mean(dim=0)
    std = concat.std(dim=0).clamp_min(1e-6)
    return mean, std


def normalize_engines(
    engines: List[torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> List[torch.Tensor]:
    """Apply sensor-wise z-score normalization."""
    return [(e - mean) / std for e in engines]


class HEPAPretrainDataset(Dataset):
    """
    Unlabeled pretraining samples.

    Returns:
        context: [context_length, num_sensors]
        future:  [max_horizon, num_sensors]

    The training loop samples Δt log-uniformly and asks the predictor to match
    the target representation of future[:, :Δt].
    """

    def __init__(self, engines: List[torch.Tensor], context_length: int, max_horizon: int, stride: int):
        self.engines = engines
        self.context_length = context_length
        self.max_horizon = max_horizon
        self.indices: List[Tuple[int, int]] = []

        for engine_idx, engine in enumerate(engines):
            T = engine.shape[0]
            # t is the first future index; context is [t-context_length, t).
            last_t = T - max_horizon
            for t in range(context_length, last_t, stride):
                self.indices.append((engine_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        engine_idx, t = self.indices[idx]
        x = self.engines[engine_idx]
        context = x[t - self.context_length:t]
        future = x[t:t + self.max_horizon]
        return context, future


class HEPAFinetuneDataset(Dataset):
    """
    Labeled downstream samples for horizon-conditioned event prediction.

    For an engine that fails after its final cycle, RUL at context time t is T - t.
    Labels are:
        y(t, Δt) = 1[RUL <= Δt], for Δt = 1..K.

    Returns:
        context: [context_length, num_sensors]
        labels:  [max_horizon]
        rul:     scalar remaining useful life in cycles
    """

    def __init__(self, engines: List[torch.Tensor], context_length: int, max_horizon: int, stride: int):
        self.engines = engines
        self.context_length = context_length
        self.max_horizon = max_horizon
        self.indices: List[Tuple[int, int]] = []

        for engine_idx, engine in enumerate(engines):
            T = engine.shape[0]
            for t in range(context_length, T, stride):
                self.indices.append((engine_idx, t))

        self.horizons = torch.arange(1, max_horizon + 1, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        engine_idx, t = self.indices[idx]
        x = self.engines[engine_idx]
        T = x.shape[0]
        context = x[t - self.context_length:t]
        rul = float(T - t)
        labels = (self.horizons >= rul).float()
        return context, labels, torch.tensor(rul, dtype=torch.float32)


# -----------------------------
# HEPA model components
# -----------------------------

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Create standard sinusoidal positional encodings with shape [1, max_len, d_model]."""
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class HorizonEmbedding(nn.Module):
    """
    Deterministic sinusoidal horizon embedding.

    This is where Δt enters the model. The predictor receives [h_t ; emb(Δt)].
    """

    def __init__(self, emb_dim: int, max_horizon: int):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError("horizon_emb_dim must be even.")
        self.emb_dim = emb_dim
        self.max_horizon = max_horizon

        half = emb_dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, horizons: torch.Tensor) -> torch.Tensor:
        """
        Args:
            horizons: arbitrary shape tensor containing horizons in [1, K].

        Returns:
            embedding with shape horizons.shape + [emb_dim]
        """
        h = horizons.float().clamp_min(1.0)
        scaled = h / float(self.max_horizon)
        angles = 2.0 * math.pi * scaled.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class PatchTransformerEncoder(nn.Module):
    """
    Patch-based Transformer encoder used by HEPA.

    - For context x_{<=t}, it is run with causal attention and the final patch
      representation is used as h_t.
    - For future interval x_{(t,t+Δt]}, it is run bidirectionally and attention
      pooling summarizes the interval into h*.
    """

    def __init__(
        self,
        num_sensors: int,
        patch_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_patches: int = 128,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_projection = nn.Linear(num_sensors * patch_size, d_model)
        self.register_buffer(
            "positional_encoding",
            sinusoidal_positional_encoding(max_patches, d_model),
            persistent=False,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

        # Used only for bidirectional target interval pooling.
        self.pool_score = nn.Linear(d_model, 1)

    def _instance_normalize(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Per-sample, per-channel instance normalization over valid timesteps.
        """
        B, T, _ = x.shape
        device = x.device
        mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        mask_expanded = mask.unsqueeze(-1)

        denom = lengths.float().clamp_min(1.0).view(B, 1, 1)
        mean = (x * mask_expanded).sum(dim=1, keepdim=True) / denom
        var = ((x - mean) * mask_expanded).pow(2).sum(dim=1, keepdim=True) / denom
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)

        # Zero invalid padded positions.
        return x_norm * mask_expanded

    def _patchify(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert [B, T, S] to patch tokens [B, N, P*S] and key-padding mask [B, N].
        """
        B, T, S = x.shape
        pad_len = (self.patch_size - (T % self.patch_size)) % self.patch_size
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, S, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        padded_T = x.shape[1]
        n_patches = padded_T // self.patch_size
        x = x.contiguous().view(B, n_patches, self.patch_size * S)

        patch_lengths = torch.div(lengths + self.patch_size - 1, self.patch_size, rounding_mode="floor")
        patch_ids = torch.arange(n_patches, device=x.device).unsqueeze(0)
        key_padding_mask = patch_ids >= patch_lengths.unsqueeze(1)
        return x, key_padding_mask

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        causal: bool = True,
        pool: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:       [B, T, S]
            lengths: valid timestep lengths. If None, all T steps are valid.
            causal:  if True, use causal self-attention.
            pool:    if True, attention-pool all valid patches; otherwise return
                     final valid patch representation.

        Returns:
            representation [B, d_model]
        """
        B, T, _ = x.shape
        device = x.device
        if lengths is None:
            lengths = torch.full((B,), T, device=device, dtype=torch.long)

        x = self._instance_normalize(x, lengths)
        patches, key_padding_mask = self._patchify(x, lengths)
        n_patches = patches.shape[1]

        z = self.patch_projection(patches)
        z = z + self.positional_encoding[:, :n_patches].to(device)

        attn_mask = None
        if causal:
            attn_mask = torch.triu(
                torch.ones(n_patches, n_patches, device=device, dtype=torch.bool),
                diagonal=1,
            )

        z = self.transformer(z, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        z = self.output_norm(z)

        if pool:
            # Bidirectional target encoder summary via attention pooling.
            scores = self.pool_score(z).squeeze(-1)
            scores = scores.masked_fill(key_padding_mask, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (z * weights).sum(dim=1)

        # Causal context summary: final valid patch.
        patch_lengths = (~key_padding_mask).sum(dim=1).clamp_min(1)
        last_indices = patch_lengths - 1
        batch_indices = torch.arange(B, device=device)
        return z[batch_indices, last_indices]


class HorizonConditionedPredictor(nn.Module):
    """
    HEPA predictor g_phi(h_t, Δt).

    During pretraining:
        predicts future representation h*_{(t,t+Δt]}.

    During finetuning:
        produces horizon-specific representations that the event head maps to hazards.
    """

    def __init__(
        self,
        d_model: int,
        horizon_emb_dim: int,
        hidden_dim: int,
        max_horizon: int,
        dropout: float,
    ):
        super().__init__()
        self.horizon_embedding = HorizonEmbedding(horizon_emb_dim, max_horizon)
        in_dim = d_model + horizon_emb_dim

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, h: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:
                [B, d] or [B, K, d]
            horizons:
                [B] or [B, K]

        Returns:
            predicted representation with same leading dims as h.
        """
        emb = self.horizon_embedding(horizons.to(h.device))
        x = torch.cat([h, emb], dim=-1)
        return self.net(x)


class EventHead(nn.Module):
    """Shared linear hazard head: LayerNorm + Linear -> logit."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).squeeze(-1)


class HEPAModel(nn.Module):
    """
    Complete HEPA model.

    Components:
        f_theta: causal/bidirectional patch Transformer encoder
        g_phi:   horizon-conditioned predictor
        event head: shared hazard head

    The encoder is jointly trained during JEPA pretraining and frozen during
    downstream predictor finetuning.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        max_patches = math.ceil(max(cfg.context_length, cfg.max_horizon) / cfg.patch_size) + 4

        self.cfg = cfg
        self.encoder = PatchTransformerEncoder(
            num_sensors=cfg.num_sensors,
            patch_size=cfg.patch_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            max_patches=max_patches,
        )
        self.predictor = HorizonConditionedPredictor(
            d_model=cfg.d_model,
            horizon_emb_dim=cfg.horizon_emb_dim,
            hidden_dim=cfg.predictor_hidden,
            max_horizon=cfg.max_horizon,
            dropout=cfg.dropout,
        )
        self.event_head = EventHead(cfg.d_model)

    def pretrain_forward(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        horizons: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        JEPA pretraining forward pass.

        Online branch:
            h_t = f_theta(x_{<=t}) with causal attention.

        Target branch:
            h* = f_theta(x_{(t,t+Δt]}) with bidirectional attention and pooling.

        Predictor:
            h_hat = g_phi(h_t, Δt)

        Returns:
            h_hat, h_star
        """
        h_context = self.encoder(context, causal=True, pool=False)
        h_pred = self.predictor(h_context, horizons)

        # The target encoder sees only the first Δt steps of the future tensor.
        h_target = self.encoder(future, lengths=horizons.long(), causal=False, pool=True)
        return h_pred, h_target

    def event_surface(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Downstream event prediction.

        For each horizon j = 1..K:
            z_j = g_phi(h_t, j)
            λ_j = sigmoid(w^T z_j + b)
            p(t, Δt) = 1 - Π_{j<=Δt}(1 - λ_j)

        Returns:
            cdf_probs: [B, K], monotone event probability surface
            hazards:   [B, K], per-interval conditional hazards
        """
        B = context.shape[0]
        device = context.device
        K = self.cfg.max_horizon

        h_context = self.encoder(context, causal=True, pool=False)  # [B, d]

        horizons = torch.arange(1, K + 1, device=device).view(1, K).expand(B, K)
        h_repeated = h_context.unsqueeze(1).expand(B, K, self.cfg.d_model)

        # Horizon-conditioning is executed here for every discrete horizon.
        z = self.predictor(h_repeated, horizons)  # [B, K, d]

        hazard_logits = self.event_head(z)
        hazards = torch.sigmoid(hazard_logits).clamp(1e-6, 1.0 - 1e-6)

        survival = torch.cumprod(1.0 - hazards, dim=1)
        cdf_probs = (1.0 - survival).clamp(1e-6, 1.0 - 1e-6)
        return cdf_probs, hazards

    def freeze_encoder(self) -> None:
        """Freeze f_theta for predictor finetuning."""
        for p in self.encoder.parameters():
            p.requires_grad = False


# -----------------------------
# Losses and metrics
# -----------------------------

def sigreg_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Practical SIGReg-style isotropic Gaussian regularizer.

    Encourages predicted representations to have:
        - zero mean
        - unit per-dimension standard deviation
        - low off-diagonal covariance

    This prevents the JEPA predictor from collapsing to a constant vector.
    """
    if z.ndim != 2:
        z = z.view(-1, z.shape[-1])

    B, D = z.shape
    if B < 2:
        return z.new_tensor(0.0)

    mean_loss = z.mean(dim=0).pow(2).mean()
    std_loss = (z.std(dim=0, unbiased=False) - 1.0).pow(2).mean()

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / float(B - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).mean()

    return mean_loss + std_loss + cov_loss


def hepa_pretraining_loss(
    h_pred: torch.Tensor,
    h_target: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HEPA pretraining loss:
        L = (1 - α) * ||normalize(h_hat) - normalize(h*)||_1
            + α * SIGReg(h_hat)
    """
    pred_norm = F.normalize(h_pred, dim=-1)
    target_norm = F.normalize(h_target, dim=-1)

    l1 = F.l1_loss(pred_norm, target_norm)
    sig = sigreg_loss(h_pred)
    total = (1.0 - alpha) * l1 + alpha * sig
    return total, l1.detach(), sig.detach()


def positive_weighted_bce_on_cdf(
    probs: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    """
    Positive-weighted BCE applied to cumulative event probabilities p(t, Δt).

    This matches the HEPA downstream design: BCE is applied to the survival CDF,
    not directly to per-step hazards.
    """
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    labels = labels.float()
    loss = -(
        pos_weight * labels * torch.log(probs)
        + (1.0 - labels) * torch.log(1.0 - probs)
    )
    return loss.mean()


def compute_pos_weight(dataset: Dataset, batch_size: int = 128) -> float:
    """Compute N_neg / N_pos over all horizon labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pos = 0.0
    total = 0.0
    for _, labels, _ in loader:
        pos += float(labels.sum().item())
        total += float(labels.numel())
    neg = total - pos
    return float(neg / max(pos, 1.0))


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """
    Compute AUROC using rank statistics. Returns None for degenerate labels.
    """
    scores = scores.detach().flatten().cpu()
    labels = labels.detach().flatten().cpu().bool()

    n_pos = int(labels.sum().item())
    n = labels.numel()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float32)

    sum_pos_ranks = ranks[labels].sum().item()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def evaluate_h_auroc(model: HEPAModel, loader: DataLoader, device: torch.device) -> float:
    """
    Horizon-averaged AUROC over the full probability surface p(t, Δt).
    Degenerate horizons with all-positive or all-negative labels are skipped.
    """
    model.eval()
    all_probs = []
    all_labels = []

    for context, labels, _ in loader:
        context = context.to(device)
        probs, _ = model.event_surface(context)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    aucs = []
    for k in range(probs.shape[1]):
        auc = binary_auc(probs[:, k], labels[:, k])
        if auc is not None:
            aucs.append(auc)

    return float(sum(aucs) / max(len(aucs), 1))


@torch.no_grad()
def evaluate_event_loss(
    model: HEPAModel,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
) -> float:
    """Average downstream event loss."""
    model.eval()
    total_loss = 0.0
    total_count = 0

    for context, labels, _ in loader:
        context = context.to(device)
        labels = labels.to(device)
        probs, _ = model.event_surface(context)
        loss = positive_weighted_bce_on_cdf(probs, labels, pos_weight)
        total_loss += float(loss.item()) * context.shape[0]
        total_count += context.shape[0]

    return total_loss / max(total_count, 1)


# -----------------------------
# Training
# -----------------------------

def sample_log_uniform_horizons(batch_size: int, max_horizon: int, device: torch.device) -> torch.Tensor:
    """
    Sample Δt ~ LogUniform[1, K] as used in HEPA pretraining.
    """
    u = torch.rand(batch_size, device=device)
    horizons = torch.exp(u * math.log(float(max_horizon)))
    horizons = horizons.floor().long().clamp(1, max_horizon)
    return horizons


def pretrain_hepa(
    model: HEPAModel,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> None:
    """Self-supervised HEPA/JEPA pretraining."""
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=cfg.pretrain_lr,
        weight_decay=cfg.weight_decay,
    )

    model.train()

    for epoch in range(1, cfg.pretrain_epochs + 1):
        total = 0.0
        total_l1 = 0.0
        total_sig = 0.0
        count = 0

        for context, future in loader:
            context = context.to(device)
            future = future.to(device)
            B = context.shape[0]

            horizons = sample_log_uniform_horizons(B, cfg.max_horizon, device)

            h_pred, h_target = model.pretrain_forward(context, future, horizons)
            loss, l1, sig = hepa_pretraining_loss(h_pred, h_target, cfg.sigreg_alpha)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total += float(loss.item()) * B
            total_l1 += float(l1.item()) * B
            total_sig += float(sig.item()) * B
            count += B

        print(
            f"[Pretrain] Epoch {epoch:02d}/{cfg.pretrain_epochs} | "
            f"loss={total / count:.4f} | "
            f"L1={total_l1 / count:.4f} | "
            f"SIGReg={total_sig / count:.4f}"
        )


def finetune_hepa(
    model: HEPAModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    pos_weight: float,
) -> None:
    """
    Downstream predictor finetuning.

    Encoder is frozen. Only:
        - horizon-conditioned predictor g_phi
        - event hazard head
    are optimized.
    """
    model.freeze_encoder()

    trainable_params = list(model.predictor.parameters()) + list(model.event_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.finetune_lr,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(1, cfg.finetune_epochs + 1):
        model.train()
        model.encoder.eval()  # Frozen encoder should not use training-time dropout.

        total_loss = 0.0
        count = 0

        for context, labels, _ in train_loader:
            context = context.to(device)
            labels = labels.to(device)
            B = context.shape[0]

            probs, hazards = model.event_surface(context)

            # RUL/Event prediction loss over the monotone CDF surface p(t, Δt).
            loss = positive_weighted_bce_on_cdf(probs, labels, pos_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * B
            count += B

        val_loss = evaluate_event_loss(model, val_loader, device, pos_weight)
        h_auc = evaluate_h_auroc(model, val_loader, device)

        print(
            f"[Finetune] Epoch {epoch:02d}/{cfg.finetune_epochs} | "
            f"train_event_loss={total_loss / count:.4f} | "
            f"val_event_loss={val_loss:.4f} | "
            f"val_h-AUROC={h_auc:.4f}"
        )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic FD001-like engines.
    engines = generate_synthetic_fd001_like_engines(cfg)
    random.shuffle(engines)

    n_train = int(cfg.train_fraction * len(engines))
    raw_train_engines = engines[:n_train]
    raw_val_engines = engines[n_train:]

    mean, std = fit_standardizer(raw_train_engines)
    train_engines = normalize_engines(raw_train_engines, mean, std)
    val_engines = normalize_engines(raw_val_engines, mean, std)

    pretrain_dataset = HEPAPretrainDataset(
        train_engines,
        context_length=cfg.context_length,
        max_horizon=cfg.max_horizon,
        stride=cfg.sample_stride,
    )
    train_ft_dataset = HEPAFinetuneDataset(
        train_engines,
        context_length=cfg.context_length,
        max_horizon=cfg.max_horizon,
        stride=cfg.sample_stride,
    )
    val_ft_dataset = HEPAFinetuneDataset(
        val_engines,
        context_length=cfg.context_length,
        max_horizon=cfg.max_horizon,
        stride=cfg.sample_stride,
    )

    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    train_ft_loader = DataLoader(
        train_ft_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_ft_loader = DataLoader(
        val_ft_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(
        f"Synthetic FD001-like split: "
        f"{len(train_engines)} train engines, {len(val_engines)} validation engines"
    )
    print(
        f"Samples: pretrain={len(pretrain_dataset)}, "
        f"finetune_train={len(train_ft_dataset)}, "
        f"finetune_val={len(val_ft_dataset)}"
    )

    pos_weight = compute_pos_weight(train_ft_dataset)
    print(f"Positive BCE weight N_neg/N_pos = {pos_weight:.3f}")

    model = HEPAModel(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")

    print("\nStage 1: Self-supervised JEPA pretraining")
    pretrain_hepa(model, pretrain_loader, cfg, device)

    print("\nStage 2: Predictor finetuning with frozen encoder")
    finetune_hepa(model, train_ft_loader, val_ft_loader, cfg, device, pos_weight)

    final_h_auc = evaluate_h_auroc(model, val_ft_loader, device)
    final_loss = evaluate_event_loss(model, val_ft_loader, device, pos_weight)

    print("\nFinal validation metrics")
    print(f"  Event loss: {final_loss:.4f}")
    print(f"  h-AUROC:    {final_h_auc:.4f}")

    # Demonstrate strict monotonicity of the survival CDF for one batch.
    model.eval()
    with torch.no_grad():
        context, labels, rul = next(iter(val_ft_loader))
        context = context.to(device)
        probs, hazards = model.event_surface(context)
        monotone_violations = (probs[:, 1:] < probs[:, :-1]).sum().item()

    print(f"  Monotonicity violations in sample batch: {monotone_violations}")


if __name__ == "__main__":
    main()
```