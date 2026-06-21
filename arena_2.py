#!/usr/bin/env python3
"""
HEPA — Horizon-Conditioned Event Predictive Architecture
========================================================

Complete PyTorch implementation based on:
  https://arxiv.org/abs/2605.11130

Architecture summary:
  Stage 1 (Pretrain)  : Causal Transformer encoder + horizon-conditioned
                         predictor via Joint-Embedding Predictive Architecture
                         (JEPA) with SIGReg regularisation.
  Stage 2 (Finetune)  : Freeze encoder; finetune predictor + lightweight event
                         head that outputs a discrete-time survival CDF over
                         densely sampled horizons.

Key equations referenced:
  (1) ĥ = g_φ(h_t, Δt)                         — predictor
  (2) L  = (1-α)‖ĥ−h*‖₁ + α·L_SIG             — pretraining loss
  (3) λ  = σ(wᵀĥ + b)                          — hazard rate
  (4) p(t,Δt) = 1 − ∏(1−λ_j)                  — survival CDF
  (5) L_FT = Σ w⁺·BCE(p(t,Δt), y(t,Δt))       — finetuning loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ======================================================================
# 1.  PATCH EMBEDDING  (non-overlapping patches + per-instance InstanceNorm)
# ======================================================================

class PatchEmbedding(nn.Module):
    """Tokenises a multivariate time-series into non-overlapping patches,
    applies per-instance per-channel Instance Normalisation, and projects
    to d_model dimensions (following the PatchTST convention adopted by HEPA).

    Input : [B, seq_len, num_channels]   (seq_len must be divisible by patch_size)
    Output: [B, num_patches, d_model]
    """

    def __init__(self, num_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.instance_norm = nn.InstanceNorm1d(num_channels, affine=False)
        self.projection = nn.Linear(patch_size * num_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        num_patches = S // self.patch_size                       # (a) split into patches

        x_norm = self.instance_norm(x.transpose(1, 2)).transpose(1, 2)  # (b) InstanceNorm
        patches = x_norm.reshape(B, num_patches, -1)             # (c) [B, N, P·C]
        return self.projection(patches)                          # (d) [B, N, d]


# ======================================================================
# 2.  CAUSAL TRANSFORMER ENCODER
# ======================================================================

class CausalTransformerEncoder(nn.Module):
    """Stacked Transformer encoder layers with an optional causal mask.

    causal=True  → lower-triangular mask (online / past encoder)
    causal=False → full bidirectional attention (target / future encoder)
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if causal:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
            return self.transformer(x, mask=mask)
        return self.transformer(x)  # bidirectional, no mask


# ======================================================================
# 3.  SIGREG  (Sketched Isotropic Gaussian Regularisation)
# ======================================================================

class SIGRegLoss(nn.Module):
    """Prevents representation collapse by encouraging the predicted
    embeddings to follow an isotropic Gaussian distribution.

        L_SIG = ‖Cov(Z) − I‖_F² + ‖Mean(Z)‖²

    Reference: HEPA §3.1 (replaces the EMA schedule of standard JEPA).
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        mu = z.mean(dim=0)
        z_c = z - mu
        cov = (z_c.t() @ z_c) / B
        return torch.norm(cov - torch.eye(cov.size(0), device=z.device), p='fro') ** 2 \
               + torch.norm(mu) ** 2


# ======================================================================
# 4.  HEPA  —  MAIN MODEL
# ======================================================================

class HEPA(nn.Module):
    """Horizon-Conditioned Event Predictive Architecture.

    Components
    ----------
    patch_embed   : shared PatchEmbedding for both encoder branches
    encoder       : causal Transformer (online); weight-shared with target
    predictor g_φ : 3-layer MLP  [h_t ; Δt] → ĥ  (horizon-conditioned)
    event_head    : LayerNorm + Linear + Sigmoid → hazard rate λ ∈ (0,1)

    Two-stage training
    ------------------
    Stage 1 – Pretrain (encoder + predictor, no labels):
              JEPA loss  L = (1−α)‖ĥ−h*‖₁ + α·L_SIG
    Stage 2 – Finetune (predictor + event head only):
              Positive-weighted BCE on survival CDF  Σ w⁺·BCE(p(t,Δt), y)
    """

    def __init__(
        self,
        num_sensors: int = 14,
        patch_size: int = 16,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        context_length: int = 512,
        max_horizon: int = 200,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.context_length = context_length
        self.max_horizon = max_horizon

        # ── Sinusoidal positional encoding (fixed, non-learned) ──
        max_pos = max(context_length, max_horizon + patch_size) + patch_size
        self.register_buffer('pe', self._build_sinusoidal_pe(max_pos, d_model))

        # ── Shared patch embedding ──
        self.patch_embed = PatchEmbedding(num_sensors, patch_size, d_model)

        # ── Causal Transformer encoder (weight-shared online ↔ target) ──
        self.encoder = CausalTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )

        # ── Horizon-conditioned predictor g_φ  (Eq. 1) ──
        #    3-layer MLP:  (d_model + 1) → d_model → d_model → d_model
        self.predictor = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # ── Event / hazard head  (Eq. 3) ──
        #    LayerNorm → Linear → Sigmoid   (≈ 769 params)
        self.event_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    # ── Static helpers ──────────────────────────────────────────

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)          # [1, max_len, d_model]

    # ── Core encoding methods ───────────────────────────────────

    def _encode(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Shared encoder: patch-embed → add PE → Transformer."""
        patches = self.patch_embed(x)                          # [B, N, d]
        B, N, _ = patches.shape
        patches = patches + self.pe[:, :N, :]                  # add sinusoidal PE
        return self.encoder(patches, causal=causal)             # [B, N, d]

    def encode_context(self, x_context: torch.Tensor) -> torch.Tensor:
        """Online encoder: causal attention over past context.
        Returns the last token → compact summary of x_{≤t}.
        """
        return self._encode(x_context, causal=True)[:, -1, :]  # [B, d]

    def encode_future(self, x_future: torch.Tensor) -> torch.Tensor:
        """Target encoder: bidirectional attention over future window.
        Same weights as online encoder (joint training, no stop-gradient).
        Mean-pools over tokens → single representation h*.
        """
        return self._encode(x_future, causal=False).mean(dim=1)  # [B, d]

    # ── Horizon conditioning ────────────────────────────────────

    def predict(self, h_t: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Horizon-conditioned prediction  (Eq. 1).

        Concatenates the encoded context with a *normalised* horizon scalar,
        then passes through a 3-layer MLP to produce ĥ.

        h_t:     [B, d_model]
        delta_t: [B]  (integer steps)
        Returns: [B, d_model]
        """
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)
        delta_t_norm = delta_t.float() / self.max_horizon       # → [0, 1]
        return self.predictor(torch.cat([h_t, delta_t_norm], dim=-1))

    # ── Forward passes for the two training stages ──────────────

    def forward_pretrain(self, x_context, x_future, delta_t):
        """JEPA pretraining forward pass  (Stage 1).

        HORIZON-CONDITIONING: the predictor g_φ receives both h_t and Δt,
        learning to forecast *representations* at many timescales.

        Returns:
            h_pred  [B, d_model]  — predicted future embedding
            h_star  [B, d_model]  — target (bidirectional encoder on future)
        """
        h_t = self.encode_context(x_context)
        h_pred = self.predict(h_t, delta_t)
        h_star = self.encode_future(x_future)     # weight-shared, joint grads
        return h_pred, h_star

    def forward_finetune(self, x_context):
        """Finetuning forward pass  (Stage 2).

        The encoder is frozen; the predictor + event head run at *all* K
        horizons to build a full survival CDF surface.

        Returns:
            survival_cdf  [B, K]  — p(t, k) = P(event within k steps)
            hazards       [B, K]  — λ_k  (per-horizon hazard rates)
        """
        # ── Encoder forward (frozen) ──
        h_t = self.encode_context(x_context).detach()       # [B, d]
        B, K = h_t.size(0), self.max_horizon
        device = h_t.device

        # ── Efficient batched prediction across all K horizons ──
        horizons = torch.arange(1, K + 1, device=device).unsqueeze(0).expand(B, -1)
        h_t_exp = h_t.unsqueeze(1).expand(B, K, -1).reshape(-1, self.d_model)
        horizons_flat = horizons.reshape(-1, 1)

        h_all = self.predict(h_t_exp, horizons_flat)        # [B·K, d]
        lam = self.event_head(h_all).squeeze(-1)             # [B·K]
        hazards = lam.reshape(B, K)                          # [B, K]

        # ── Survival CDF  (Eq. 4, computed in log-space for stability) ──
        eps = 1e-7
        log_one_minus = torch.log(1.0 - hazards + eps)      # [B, K]
        log_surv = torch.cumsum(log_one_minus, dim=1)       # [B, K]
        survival_cdf = 1.0 - torch.exp(log_surv)             # [B, K]

        return survival_cdf, hazards

    # ── Utilities ───────────────────────────────────────────────

    def freeze_encoder(self):
        """Freeze encoder weights so only predictor + event head are trained."""
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# 5.  SYNTHETIC DEGRADATION DATASET  (mock data)
# ======================================================================

class SyntheticDegradationDataset:
    """
    Generates synthetic multivariate time-series that mimic turbofan-like
    degradation patterns.

    Each "engine" has:
      • 14 sensor channels with correlated drift
      • Random operating conditions
      • A failure event at a random lifetime

    Provides two batch-generation methods for the two HEPA stages.
    """

    def __init__(
        self,
        num_sequences: int = 200,
        num_sensors: int = 14,
        context_length: int = 512,
        max_horizon: int = 150,
        patch_size: int = 16,
        seed: int = 42,
    ):
        self.num_sensors = num_sensors
        self.context_length = context_length
        self.max_horizon = max_horizon
        self.patch_size = patch_size
        # Total length must accommodate context + max_horizon + padding
        self.total_length = context_length + max_horizon + patch_size

        np.random.seed(seed)
        self.data, self.rul, self.event_times = [], [], []

        for _ in range(num_sequences):
            sensors, rul, evt = self._generate_engine()
            self.data.append(sensors)
            self.rul.append(rul)
            self.event_times.append(evt)

        self.data = np.array(self.data, dtype=np.float32)    # [N, S, C]
        self.rul = np.array(self.rul, dtype=np.float32)       # [N, S]
        self.event_times = np.array(self.event_times)

    # ── Engine simulation ──────────────────────────────────────

    def _generate_engine(self):
        """Simulate one engine run-to-failure with gradual sensor drift."""
        S, C = self.total_length, self.num_sensors
        lifetime = np.random.randint(300, 600)                 # random EOL

        op_cond = np.random.randn(2) * 0.5                     # 2 operating modes
        # Per-sensor degradation sensitivity (some sensors more affected)
        sensitivity = np.clip(
            np.array([2.0, 1.5, 1.2, 0.9, 0.7, 0.5, 0.3, 1.0,
                       0.4, 0.3, 0.15, 0.1, 0.08, 0.05])[:C], 0.05, 2.5)
        baseline = np.random.randn(C) * 0.3                   # per-engine offset

        sensors = np.zeros((S, C))
        rul = np.zeros(S)

        for t in range(S):
            if t < lifetime:
                # Quadratic degradation: slow early, accelerating near EOL
                age_ratio = 1.0 - (lifetime - t) / lifetime
                degradation = age_ratio ** 1.5
                noise = np.random.randn(C) * 0.03
                sensors[t] = (baseline
                              + op_cond[0] * 0.15
                              + op_cond[1] * 0.1 * np.sin(t * 0.05)
                              + degradation * sensitivity
                              + noise)
                rul[t] = lifetime - t
            else:
                # Post-failure: hover near failure-state + tiny noise
                sensors[t] = sensors[max(0, min(lifetime - 1, t - 1))] \
                             + np.random.randn(C) * 0.01
                rul[t] = 0.0

        return sensors, rul, lifetime

    # ── Batch generators ───────────────────────────────────────

    def get_pretraining_batch(self, batch_size: int, device: torch.device):
        """Return (context, future_window, delta_t) for JEPA pretraining.

        Δt is sampled from a log-uniform distribution over [1, max_horizon],
        matching the paper's §3.1 specification.
        The future window is right-padded to the nearest patch_size multiple.
        """
        N = len(self)
        idx = np.random.choice(N, batch_size, replace=False)

        ctx_list, fut_list, dt_list = [], [], []
        for i in idx:
            # Ensure enough room for context + future
            max_start = max(1, self.total_length - self.context_length
                            - self.max_horizon - self.patch_size)
            start = np.random.randint(0, max_start)
            ctx_end = start + self.context_length

            # ── Log-uniform horizon sampling (paper: LogUniform[1, K]) ──
            log_min, log_max = 0.0, np.log(self.max_horizon)
            dt = int(np.floor(np.exp(np.random.uniform(log_min, log_max))))
            dt = max(1, min(dt, self.max_horizon))
            # Pad to multiple of patch_size for clean tokenisation
            dt_padded = ((dt + self.patch_size - 1) // self.patch_size) * self.patch_size

            ctx_list.append(torch.FloatTensor(self.data[i, start:ctx_end]))
            fut_list.append(torch.FloatTensor(self.data[i, ctx_end:ctx_end + dt_padded]))
            dt_list.append(dt)

        return (torch.stack(ctx_list).to(device),
                torch.stack(fut_list).to(device),
                torch.FloatTensor(dt_list).to(device))

    def get_finetuning_batch(self, batch_size: int, device: torch.device):
        """Return (context, binary_labels) for finetuning.

        For each sample, labels[k] = 1 iff the event occurs within (k+1) steps.
        """
        N = len(self)
        idx = np.random.choice(N, batch_size, replace=False)

        ctx_list, lbl_list = [], []
        for i in idx:
            # Time index: must have full context ahead and enough horizon room
            min_t = self.context_length
            max_t = max(min_t + 1, self.total_length - self.max_horizon - 1)
            t = np.random.randint(min_t, max_t)

            ctx_list.append(torch.FloatTensor(self.data[i, t - self.context_length: t]))

            # Build binary event labels from RUL
            rul_t = self.rul[i, t]
            label = np.zeros(self.max_horizon, dtype=np.float32)
            for k in range(self.max_horizon):
                label[k] = 1.0 if rul_t <= (k + 1) else 0.0
            lbl_list.append(torch.FloatTensor(label))

        return (torch.stack(ctx_list).to(device),
                torch.stack(lbl_list).to(device))


# ======================================================================
# 6.  h-AUROC EVALUATION  (no sklearn dependency)
# ======================================================================

def horizon_auroc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """h-AUROC: mean per-horizon AUROC across the probability surface.

    predictions: [B, K]  survival CDF values
    labels:      [B, K]  binary event indicators
    """
    B, K = predictions.shape
    aurocs = []
    for k in range(K):
        p, l = predictions[:, k], labels[:, k]
        if 0 < l.sum() < len(l):                # both classes present
            aurocs.append(_single_auroc(p, l))
    return float(np.mean(aurocs)) if aurocs else 0.0


def _single_auroc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Trapezoidal AUROC for a single horizon (pure NumPy)."""
    order = np.argsort(-y_pred)
    y_true = y_true[order]
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fpr = prev_tpr = 0.0
    for lbl in y_true:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr, fpr = tp / n_pos, fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return auc


# ======================================================================
# 7.  MAIN TRAINING PIPELINE
# ======================================================================

def main():
    """Full two-stage HEPA training loop on synthetic degradation data."""

    # ── Configuration (matches paper Table 7 defaults) ──────────
    cfg = dict(
        # Architecture
        num_sensors=14, patch_size=16, d_model=256, nhead=4,
        num_encoder_layers=2, dim_feedforward=1024, dropout=0.1,
        context_length=512, max_horizon=150,

        # Data sizes
        num_train_seq=200, num_val_seq=20,

        # Stage 1 – Pretraining
        pre_epochs=5, pre_batch=32,
        pre_lr=3e-4, pre_wd=1e-2, sigreg_alpha=0.1,

        # Stage 2 – Finetuning
        ft_epochs=5, ft_batch=32,
        ft_lr=1e-3, ft_wd=1e-2,

        batches_per_epoch=50, log_interval=1,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  HEPA — Horizon-Conditioned Event Predictive Architecture")
    print(f"  Device: {device.upper()}")
    print("=" * 70)

    # ── Step A: Data generation ─────────────────────────────────
    print("\n[1/4] Generating synthetic degradation data...")
    train_ds = SyntheticDegradationDataset(
        num_sequences=cfg['num_train_seq'], num_sensors=cfg['num_sensors'],
        context_length=cfg['context_length'], max_horizon=cfg['max_horizon'],
        patch_size=cfg['patch_size'], seed=42,
    )
    val_ds = SyntheticDegradationDataset(
        num_sequences=cfg['num_val_seq'], num_sensors=cfg['num_sensors'],
        context_length=cfg['context_length'], max_horizon=cfg['max_horizon'],
        patch_size=cfg['patch_size'], seed=123,
    )
    print(f"  Train: {len(train_ds)} sequences | Val: {len(val_ds)} sequences")
    print(f"  Sensors: {cfg['num_sensors']} | Context: {cfg['context_length']} | "
          f"Max horizon: {cfg['max_horizon']} | Patch: {cfg['patch_size']}")

    # ── Step B: Model initialisation ────────────────────────────
    print("\n[2/4] Initialising HEPA model...")
    model = HEPA(
        num_sensors=cfg['num_sensors'], patch_size=cfg['patch_size'],
        d_model=cfg['d_model'], nhead=cfg['nhead'],
        num_encoder_layers=cfg['num_encoder_layers'],
        dim_feedforward=cfg['dim_feedforward'], dropout=cfg['dropout'],
        context_length=cfg['context_length'], max_horizon=cfg['max_horizon'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ── Sanity check: forward pass shapes ───────────────────────
    print("\n  Sanity check (forward shapes):")
    _ctx = torch.randn(2, cfg['context_length'], cfg['num_sensors']).to(device)
    _fut = torch.randn(2, cfg['patch_size'] * 10, cfg['num_sensors']).to(device)
    _dt = torch.randint(1, cfg['max_horizon'], (2,)).to(device)
    hp, hs = model.forward_pretrain(_ctx, _fut, _dt)
    sc, hz = model.forward_finetune(_ctx)
    print(f"    h_pred: {tuple(hp.shape)}  |  h_star: {tuple(hs.shape)}")
    print(f"    surv:   {tuple(sc.shape)}  |  hazards:{tuple(hz.shape)}")

    # ── Step C: Pretraining (Stage 1) ───────────────────────────
    print("\n[3/4] Stage 1 — Pretraining (JEPA + SIGReg)...")
    print(f"  Optimizer : AdamW (lr={cfg['pre_lr']}, wd={cfg['pre_wd']})")
    print(f"  Loss      : (1-α)·‖ĥ−h*‖₁ + α·L_SIG  (α={cfg['sigreg_alpha']})")
    print(f"  Encoder & predictor trained jointly (weight-shared target, §3.1)")

    optimizer_pre = torch.optim.AdamW(
        model.parameters(), lr=cfg['pre_lr'], weight_decay=cfg['pre_wd'])
    sig_reg = SIGRegLoss()

    for ep in range(1, cfg['pre_epochs'] + 1):
        model.train()
        m_l1 = m_sig = m_tot = 0.0
        n_b = 0
        for _ in range(cfg['batches_per_epoch']):
            ctx, fut, dt = train_ds.get_pretraining_batch(cfg['pre_batch'], device)

            optimizer_pre.zero_grad()

            # ── HORIZON-CONDITIONED JEPA FORWARD PASS ──
            #    Encoder processes context causally → h_t
            #    Predictor conditions on (h_t, Δt) → ĥ
            #    Target encoder (same weights) processes future bidirectionally → h*
            h_pred, h_star = model.forward_pretrain(ctx, fut, dt)

            # ── Loss: L1 prediction + SIGReg collapse prevention (Eq. 2) ──
            l1 = F.l1_loss(h_pred, h_star, reduction='mean')
            sr = sig_reg(h_pred)
            loss = (1 - cfg['sigreg_alpha']) * l1 + cfg['sigreg_alpha'] * sr

            loss.backward()
            optimizer_pre.step()

            m_l1 += l1.item(); m_sig += sr.item(); m_tot += loss.item(); n_b += 1

        if ep % cfg['log_interval'] == 0 or ep == 1:
            print(f"  Epoch {ep:>3d}/{cfg['pre_epochs']} | "
                  f"L1: {m_l1 / n_b:.6f}  SIGReg: {m_sig / n_b:.6f}  "
                  f"Total: {m_tot / n_b:.6f}")

    # ── Step D: Finetuning (Stage 2) ────────────────────────────
    print("\n[4/4] Stage 2 — Finetuning (Predictor + Event Head)...")
    print(f"  Optimizer : AdamW (lr={cfg['ft_lr']}, wd={cfg['ft_wd']})")
    print(f"  Loss      : Positive-weighted BCE on survival CDF  (Eq. 5)")
    print(f"  Encoder   : FROZEN  |  Predictor + Event Head : TRAINABLE")

    # Freeze the pretrained encoder (only predictor + head updated)
    model.freeze_encoder()
    print(f"  Trainable params: {model.count_trainable():,}")

    optimizer_ft = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['ft_lr'], weight_decay=cfg['ft_wd'])

    for ep in range(1, cfg['ft_epochs'] + 1):
        # ── Training pass ──
        model.train()
        m_loss = m_auc = 0.0
        n_b = 0

        for _ in range(cfg['batches_per_epoch']):
            ctx, labels = train_ds.get_finetuning_batch(cfg['ft_batch'], device)
            optimizer_ft.zero_grad()

            # ── FINETUNE FORWARD: survival CDF over K horizons ──
            #    Encoder frozen → predictor runs at each horizon → event head
            surv, _ = model.forward_finetune(ctx)      # [B, K]

            # ── Positive-weighted BCE (Eq. 5, w⁺ = N_neg / N_pos) ──
            N_pos = labels.sum().item()
            N_neg = (1.0 - labels).sum().item()
            pw = N_neg / N_pos if N_pos > 0 else 1.0

            eps = 1e-7
            p = torch.clamp(surv, eps, 1.0 - eps)
            bce = -(pw * labels * torch.log(p) + (1.0 - labels) * torch.log(1.0 - p))
            loss = bce.mean()

            loss.backward()
            optimizer_ft.step()

            with torch.no_grad():
                auc = horizon_auroc(surv.cpu().numpy(), labels.cpu().numpy())

            m_loss += loss.item(); m_auc += auc; n_b += 1

        # ── Validation pass ──
        model.eval()
        v_loss = v_auc = 0.0
        v_n = 0
        with torch.no_grad():
            for _ in range(10):
                ctx, labels = val_ds.get_finetuning_batch(cfg['ft_batch'], device)
                surv, _ = model.forward_finetune(ctx)

                N_pos = labels.sum().item()
                N_neg = (1.0 - labels).sum().item()
                pw = N_neg / N_pos if N_pos > 0 else 1.0

                eps = 1e-7
                p = torch.clamp(surv, eps, 1.0 - eps)
                bce = -(pw * labels * torch.log(p) + (1 - labels) * torch.log(1 - p))
                v_loss += bce.mean().item()
                v_auc += horizon_auroc(surv.cpu().numpy(), labels.cpu().numpy())
                v_n += 1

        if ep % cfg['log_interval'] == 0 or ep == 1:
            print(f"  Epoch {ep:>3d}/{cfg['ft_epochs']} | "
                  f"Train Loss: {m_loss / n_b:.6f}  Train h-AUROC: {m_auc / n_b:.4f} | "
                  f"Val Loss: {v_loss / v_n:.6f}  Val h-AUROC: {v_auc / v_n:.4f}")

    print("\n" + "=" * 70)
    print(f"  Training complete!  Final Val h-AUROC: {v_auc / v_n:.4f}")
    print("=" * 70)
    return model


if __name__ == "__main__":
    model = main()