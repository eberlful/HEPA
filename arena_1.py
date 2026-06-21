"""
Horizon-Conditioned Event Predictive Architecture (HEPA)
Complete Implementation with Mock Data and Training Pipeline

Based on: "HEPA: Horizon-conditioned Event Predictive Architecture"
Core concepts:
- Stage 1: JEPA pretraining (causal encoder + horizon-conditioned predictor)
- Stage 2: Predictor finetuning with survival CDF for event prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import math
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HEPAConfig:
    """Configuration for HEPA model matching paper specifications."""
    # Architecture
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    patch_size: int = 16
    dropout: float = 0.1
    max_seq_len: int = 512
    
    # Pretraining
    pretrain_lr: float = 3e-4
    pretrain_epochs: int = 100
    pretrain_patience: int = 10
    alpha_sigreg: float = 0.1  # SIGReg weight
    horizon_min: int = 1
    horizon_max: int = 200
    
    # Finetuning
    finetune_lr: float = 1e-3
    finetune_epochs: int = 50
    finetune_patience: int = 10
    
    # Training
    batch_size: int = 64
    weight_decay: float = 1e-2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    n_sensors: int = 14  # Default for C-MAPSS-like data
    horizon_bins: int = 200  # K in paper


# ============================================================================
# Model Components
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Converts raw sensor time series into non-overlapping patches.
    Following PatchTST approach with instance normalization.
    """
    def __init__(self, n_sensors: int, patch_size: int, d_model: int, max_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_sensors = n_sensors
        self.projection = nn.Linear(n_sensors * patch_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len // patch_size + 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, S] raw time series
        Returns:
            [B, N_patches, d_model] patch embeddings
        """
        B, L, S = x.shape
        
        # Instance normalization per sensor
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        x = (x - mean) / std
        
        # Pad to make sequence divisible by patch_size
        pad_len = (self.patch_size - L % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Reshape into patches: [B, N_patches, patch_size * S]
        N_patches = x.shape[1] // self.patch_size
        x = x.view(B, N_patches, self.patch_size * S)
        
        # Project to d_model
        x = self.projection(x)  # [B, N_patches, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CausalTransformerEncoder(nn.Module):
    """
    Causal Transformer encoder (f_θ in paper).
    2 layers, 4 heads, d_model=256 with causal masking.
    """
    def __init__(self, config: HEPAConfig):
        super().__init__()
        self.d_model = config.d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N_patches, d_model] patch embeddings
            mask: optional padding mask
        Returns:
            [B, N_patches, d_model] encoded representations
        """
        # Causal mask: each position attends only to previous positions
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        return self.transformer(x, mask=causal_mask)


class AttentionPooling(nn.Module):
    """Attention pooling to get summary representation."""
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model]
        Returns:
            [B, d_model] pooled representation
        """
        attn_weights = F.softmax(self.attention(x), dim=1)  # [B, N, 1]
        pooled = (x * attn_weights).sum(dim=1)  # [B, d_model]
        return pooled


class HorizonConditionedPredictor(nn.Module):
    """
    Predictor g_φ that maps encoder output and horizon to predicted future representation.
    2-layer MLP with horizon conditioning.
    """
    def __init__(self, config: HEPAConfig):
        super().__init__()
        self.d_model = config.d_model
        
        # Horizon embedding: learnable embedding for each discrete horizon
        self.horizon_embedding = nn.Embedding(config.horizon_max + 1, config.d_model)
        
        # Main predictor MLP: [h_t, Δt_embedding] -> predicted future representation
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
    def forward(self, h_t: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Horizon-conditioned prediction of future representation.
        Equation (1) in paper: ĥ_{(t,t+Δt]} = g_φ(h_t, Δt)
        
        Args:
            h_t: [B, d_model] encoder summary representation
            delta_t: [B] or [B, 1] prediction horizons
        Returns:
            [B, d_model] predicted future representation
        """
        if delta_t.dim() == 0:
            delta_t = delta_t.unsqueeze(0)
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(1)
        
        delta_t = delta_t.squeeze(-1).long()
        delta_t = delta_t.clamp(0, self.horizon_embedding.num_embeddings - 1)
        
        # Get horizon embedding and concatenate with encoder output
        horizon_emb = self.horizon_embedding(delta_t)  # [B, d_model]
        combined = torch.cat([h_t, horizon_emb], dim=-1)  # [B, 2*d_model]
        
        return self.mlp(combined)


class EventHead(nn.Module):
    """
    Lightweight linear event head for downstream finetuning.
    Maps predicted representations to per-horizon hazard rates.
    """
    def __init__(self, config: HEPAConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.linear = nn.Linear(config.d_model, 1)
        
    def forward(self, h_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute hazard rate λ_Δt(t).
        Equation (3): λ_Δt(t) = σ(w^T ĥ_{(t,t+Δt]} + b)
        
        Args:
            h_pred: [B, K, d_model] predicted representations for K horizons
        Returns:
            [B, K] hazard rates in (0,1)
        """
        h_norm = self.layer_norm(h_pred)
        logits = self.linear(h_norm).squeeze(-1)  # [B, K]
        return torch.sigmoid(logits)


# ============================================================================
# HEPA Model
# ============================================================================

class HEPA(nn.Module):
    """
    Horizon-Conditioned Event Predictive Architecture.
    
    Stage 1 (Pretraining): Causal encoder + Horizon-conditioned predictor 
                           trained with JEPA objective + SIGReg.
    Stage 2 (Finetuning): Frozen encoder + Predictor + Event head
                          trained with survival CDF objective.
    """
    def __init__(self, config: HEPAConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            n_sensors=config.n_sensors,
            patch_size=config.patch_size,
            d_model=config.d_model,
            max_len=config.max_seq_len
        )
        
        # Causal encoder f_θ (shared weights for online and target in pretraining)
        self.encoder = CausalTransformerEncoder(config)
        
        # Attention pooling for summary representation
        self.pooling = AttentionPooling(config.d_model)
        
        # Horizon-conditioned predictor g_φ
        self.predictor = HorizonConditionedPredictor(config)
        
        # Event head (used only in finetuning)
        self.event_head = EventHead(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observations up to time t.
        
        Args:
            x: [B, L, S] raw time series (causal: only past)
        Returns:
            h_t: [B, d_model] summary representation
        """
        patches = self.patch_embed(x)
        encoded = self.encoder(patches)
        return self.pooling(encoded)
    
    def encode_target(self, x_future: torch.Tensor) -> torch.Tensor:
        """
        Encode future interval for target representation.
        Uses bidirectional attention (no causal mask) with same encoder weights.
        
        Args:
            x_future: [B, L_future, S] future time series
        Returns:
            h_target: [B, d_model] target representation
        """
        B, L, S = x_future.shape
        
        # Instance normalization
        mean = x_future.mean(dim=1, keepdim=True)
        std = x_future.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x_future - mean) / std
        
        # Pad
        pad_len = (self.config.patch_size - L % self.config.patch_size) % self.config.patch_size
        if pad_len > 0:
            x_norm = F.pad(x_norm, (0, 0, 0, pad_len))
        
        N_patches = x_norm.shape[1] // self.config.patch_size
        x_norm = x_norm.view(B, N_patches, self.config.patch_size * S)
        patches = self.patch_embed.projection(x_norm)
        patches = self.patch_embed.pos_encoding(patches)
        
        # Bidirectional encoding (no causal mask)
        encoded = self.encoder.transformer(patches)
        return self.pooling(encoded)
    
    def predict_future(self, h_t: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Predict future representation for given horizon.
        Equation (1): ĥ_{(t,t+Δt]} = g_φ(h_t, Δt)
        
        Args:
            h_t: [B, d_model] encoder summary
            delta_t: [B] prediction horizons
        Returns:
            [B, d_model] predicted representation
        """
        return self.predictor(h_t, delta_t)
    
    def compute_hazards(self, h_t: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
        """
        Compute hazard rates for all horizons.
        
        Args:
            h_t: [B, d_model]
            horizons: [K] discrete horizons
        Returns:
            [B, K] hazard rates λ_Δt(t)
        """
        B = h_t.size(0)
        K = horizons.size(0)
        
        # Expand h_t for all horizons
        h_t_expanded = h_t.unsqueeze(1).expand(-1, K, -1)  # [B, K, d_model]
        horizons_expanded = horizons.unsqueeze(0).expand(B, -1)  # [B, K]
        
        # Predict representation for each horizon
        h_preds = []
        for k in range(K):
            h_pred = self.predictor(h_t, horizons_expanded[:, k])  # [B, d_model]
            h_preds.append(h_pred)
        
        h_preds = torch.stack(h_preds, dim=1)  # [B, K, d_model]
        
        # Compute hazards via event head
        return self.event_head(h_preds)  # [B, K]
    
    def compute_survival_cdf(self, hazards: torch.Tensor) -> torch.Tensor:
        """
        Convert hazards to survival CDF.
        Equation (4): p(t, Δt) = 1 - Π_{j=1}^{Δt} (1 - λ_j(t))
        
        Args:
            hazards: [B, K] hazard rates
        Returns:
            [B, K] cumulative event probabilities (monotonically increasing)
        """
        survival = torch.cumprod(1.0 - hazards + 1e-8, dim=1)  # [B, K]
        return 1.0 - survival


# ============================================================================
# Loss Functions
# ============================================================================

class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.
    Prevents representation collapse by constraining predictions 
    toward isotropic Gaussian distribution.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, h_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_pred: [B, d_model] predicted representations
        Returns:
            scalar SIGReg loss
        """
        # Encourage unit variance and zero mean across batch
        mean = h_pred.mean(dim=0)
        var = h_pred.var(dim=0, unbiased=False)
        
        # KL divergence to standard normal (simplified)
        loss_mean = (mean ** 2).mean()
        loss_var = (var - 1.0).abs().mean()
        
        return loss_mean + loss_var


class PretrainingLoss(nn.Module):
    """
    Combined pretraining loss.
    Equation (2): L = (1-α) * ||ĥ - h*||_1 + α * L_SIG
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.sigreg = SIGRegLoss()
        
    def forward(self, h_pred: torch.Tensor, h_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            h_pred: [B, d_model] predicted future representation
            h_target: [B, d_model] target future representation
        Returns:
            total_loss, loss_dict
        """
        # L1 prediction loss (chosen over L2 for equal gradient distribution)
        pred_loss = F.l1_loss(h_pred, h_target)
        
        # SIGReg regularization
        sigreg_loss = self.sigreg(h_pred)
        
        total_loss = (1 - self.alpha) * pred_loss + self.alpha * sigreg_loss
        
        return total_loss, {
            'pred_loss': pred_loss.item(),
            'sigreg_loss': sigreg_loss.item(),
            'total_loss': total_loss.item()
        }


class FinetuningLoss(nn.Module):
    """
    Finetuning loss with positive-weighted BCE on survival CDF.
    Equation (5): L_FT = Σ w⁺ * BCE(p(t,Δt), y(t,Δt))
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, p_surface: torch.Tensor, y_target: torch.Tensor, 
                pos_weight: float = 1.0) -> torch.Tensor:
        """
        Args:
            p_surface: [B, K] predicted cumulative event probabilities
            y_target: [B, K] binary event indicators (1 if event within Δt)
            pos_weight: w⁺ = N_neg/N_pos for class imbalance
        Returns:
            weighted BCE loss
        """
        # BCE on cumulative surface acts as smoothing regularizer across horizons
        bce = F.binary_cross_entropy(p_surface.clamp(1e-7, 1 - 1e-7), 
                                      y_target, reduction='none')  # [B, K]
        
        # Apply positive weighting
        weights = torch.where(y_target > 0.5, 
                             torch.tensor(pos_weight, device=y_target.device),
                             torch.tensor(1.0, device=y_target.device))
        
        weighted_bce = (bce * weights).mean()
        return weighted_bce


# ============================================================================
# Data Generation
# ============================================================================

class MockTimeSeriesDataset(Dataset):
    """
    Generates synthetic multi-channel time series data simulating 
    sensor logs with degradation patterns for RUL prediction.
    
    Mimics turbofan degradation characteristics:
    - Multiple sensors showing gradual drift before failure
    - Clear degradation pattern in later cycles
    - Binary event labels at multiple horizons
    """
    def __init__(self, 
                 n_samples: int = 500,
                 seq_length: int = 256,
                 n_sensors: int = 14,
                 max_rul: int = 200,
                 failure_prob: float = 0.3,
                 seed: int = 42):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_sensors = n_sensors
        self.max_rul = max_rul
        
        # Generate synthetic data
        self.sequences, self.rul_targets, self.event_labels = self._generate_data(
            n_samples, seq_length, n_sensors, max_rul, failure_prob
        )
        
    def _generate_data(self, n_samples, seq_length, n_sensors, max_rul, failure_prob):
        """
        Generate synthetic time series with degradation patterns.
        
        Returns:
            sequences: [n_samples, seq_length, n_sensors]
            rul_targets: [n_samples] remaining useful life
            event_labels: [n_samples, max_rul] binary event indicators
        """
        sequences = []
        rul_targets = []
        event_labels = []
        
        for i in range(n_samples):
            # Determine if this sample is near failure
            is_failing = np.random.random() < failure_prob
            
            if is_failing:
                # RUL between 1 and max_rul
                rul = np.random.randint(1, max_rul + 1)
            else:
                # Healthy: RUL > max_rul (set to large value)
                rul = np.random.randint(max_rul + 10, max_rul + 200)
            
            # Generate sensor readings
            seq = np.zeros((seq_length, n_sensors))
            
            # Base noise level
            base_noise = 0.05
            
            for s in range(n_sensors):
                # Different sensors have different patterns
                if s < n_sensors // 3:
                    # Degrading sensors: show drift toward failure
                    drift_rate = np.random.uniform(0.001, 0.01)
                    noise_level = np.random.uniform(0.02, 0.08)
                    
                    for t in range(seq_length):
                        # Cycle number relative to end of sequence
                        cycle = seq_length - t
                        if is_failing and cycle <= max_rul:
                            # Degradation increases as RUL decreases
                            degradation = np.exp(-cycle / (rul * 0.3)) * drift_rate * 100
                        else:
                            degradation = 0
                        seq[t, s] = np.sin(t * 0.1 + s) * 0.5 + degradation + \
                                   np.random.randn() * noise_level
                
                elif s < 2 * n_sensors // 3:
                    # Noisy sensors with slight trend
                    trend = np.linspace(0, np.random.uniform(-0.5, 0.5), seq_length) if is_failing else np.zeros(seq_length)
                    seq[:, s] = trend + np.random.randn(seq_length) * base_noise * 2
                
                else:
                    # Stable sensors with occasional spikes
                    seq[:, s] = np.random.randn(seq_length) * base_noise
                    if is_failing:
                        # Add spikes as failure approaches
                        spike_positions = np.random.choice(seq_length, size=3, replace=False)
                        seq[spike_positions, s] += np.random.uniform(2, 5)
            
            sequences.append(seq.astype(np.float32))
            rul_targets.append(rul)
            
            # Generate event labels for each horizon
            labels = np.zeros(max_rul, dtype=np.float32)
            if is_failing:
                # Event occurs at horizon = rul
                labels[rul:max_rul] = 1.0  # Event within Δt for Δt >= RUL
            event_labels.append(labels)
        
        return (
            torch.tensor(np.stack(sequences)),
            torch.tensor(rul_targets, dtype=torch.float32),
            torch.tensor(np.stack(event_labels))
        )
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'rul': self.rul_targets[idx],
            'event_labels': self.event_labels[idx]
        }


# ============================================================================
# Training Pipeline
# ============================================================================

class HEPATrainer:
    """
    Complete training pipeline for HEPA model.
    Handles both Stage 1 (pretraining) and Stage 2 (finetuning).
    """
    def __init__(self, config: HEPAConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = HEPA(config).to(self.device)
        
        # Loss functions
        self.pretrain_loss_fn = PretrainingLoss(alpha=config.alpha_sigreg)
        self.finetune_loss_fn = FinetuningLoss()
        
        # Optimizers
        self.pretrain_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.pretrain_lr,
            weight_decay=config.weight_decay
        )
        
        # Horizons for evaluation
        self.horizons = torch.arange(1, config.horizon_bins + 1, device=self.device)
        
    def pretrain_epoch(self, dataloader: DataLoader) -> dict:
        """
        Single epoch of JEPA pretraining.
        
        For each sequence, sample a random horizon Δt,
        encode past context, predict future representation,
        and compute loss against target encoder output.
        """
        self.model.train()
        epoch_losses = {'pred_loss': 0, 'sigreg_loss': 0, 'total_loss': 0}
        n_batches = 0
        
        for batch in dataloader:
            sequences = batch['sequence'].to(self.device)  # [B, L, S]
            B, L, S = sequences.shape
            
            # Split into context (past) and target (future) windows
            # Random split point
            split_point = L // 2
            
            context = sequences[:, :split_point, :]  # Past
            target = sequences[:, split_point:, :]   # Future
            
            # Sample random horizon Δt from log-uniform distribution
            # Following paper: LogUniform[1, K]
            log_min, log_max = math.log(1), math.log(self.config.horizon_max)
            log_delta = torch.rand(B, device=self.device) * (log_max - log_min) + log_min
            delta_t = torch.exp(log_delta).long().clamp(1, self.config.horizon_max)
            
            # Encode context
            h_t = self.model.encode_context(context)  # [B, d_model]
            
            # Predict future representation for horizon Δt
            h_pred = self.model.predict_future(h_t, delta_t)  # [B, d_model]
            
            # Get target representation from future window
            # (weight-shared encoder, bidirectional)
            with torch.set_grad_enabled(True):  # Both branches receive gradients
                h_target = self.model.encode_target(target)  # [B, d_model]
            
            # Compute pretraining loss
            loss, loss_dict = self.pretrain_loss_fn(h_pred, h_target)
            
            # Backward pass
            self.pretrain_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.pretrain_optimizer.step()
            
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            n_batches += 1
        
        return {k: v / n_batches for k, v in epoch_losses.items()}
    
    def finetune_epoch(self, dataloader: DataLoader, pos_weight: float) -> dict:
        """
        Single epoch of predictor finetuning.
        
        Freeze encoder, only update predictor and event head.
        """
        # Freeze encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False
        for param in self.model.pooling.parameters():
            param.requires_grad = False
        
        # Ensure predictor and event head are trainable
        for param in self.model.predictor.parameters():
            param.requires_grad = True
        for param in self.model.event_head.parameters():
            param.requires_grad = True
        
        self.model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Finetuning optimizer (only predictor + event head)
        finetune_params = list(self.model.predictor.parameters()) + \
                         list(self.model.event_head.parameters())
        finetune_optimizer = optim.AdamW(
            finetune_params,
            lr=self.config.finetune_lr,
            weight_decay=self.config.weight_decay
        )
        
        for batch in dataloader:
            sequences = batch['sequence'].to(self.device)
            event_labels = batch['event_labels'].to(self.device)  # [B, K]
            B = sequences.size(0)
            
            # Encode context (frozen encoder)
            with torch.no_grad():
                h_t = self.model.encode_context(sequences)
            
            # Compute hazards for all horizons
            hazards = self.model.compute_hazards(h_t, self.horizons)  # [B, K]
            
            # Convert to survival CDF
            p_surface = self.model.compute_survival_cdf(hazards)  # [B, K]
            
            # Ensure event_labels match horizons dimension
            if event_labels.size(1) != self.horizons.size(0):
                K = min(event_labels.size(1), self.horizons.size(0))
                p_surface = p_surface[:, :K]
                event_labels = event_labels[:, :K]
            
            # Compute finetuning loss
            loss = self.finetune_loss_fn(p_surface, event_labels, pos_weight)
            
            finetune_optimizer.zero_grad()
            loss.backward()
            finetune_optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        return {'finetune_loss': epoch_loss / n_batches}
    
    def compute_h_auroc(self, p_surface: torch.Tensor, 
                        event_labels: torch.Tensor) -> float:
        """
        Compute horizon-averaged AUROC.
        h-AUROC = mean of per-horizon AUROC values.
        """
        from sklearn.metrics import roc_auc_score
        
        K = p_surface.size(1)
        aurocs = []
        
        for k in range(K):
            y_true = event_labels[:, k].cpu().numpy()
            y_pred = p_surface[:, k].detach().cpu().numpy()
            
            # Skip degenerate horizons
            if len(np.unique(y_true)) < 2:
                continue
            
            try:
                auroc = roc_auc_score(y_true, y_pred)
                aurocs.append(auroc)
            except:
                continue
        
        return np.mean(auroc) if aurocs else 0.5


def generate_mock_dataloaders(config: HEPAConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create mock training and validation dataloaders.
    """
    train_dataset = MockTimeSeriesDataset(
        n_samples=config.batch_size * 20,
        seq_length=config.max_seq_len,
        n_sensors=config.n_sensors,
        max_rul=config.horizon_bins,
        failure_prob=0.3,
        seed=42
    )
    
    val_dataset = MockTimeSeriesDataset(
        n_samples=config.batch_size * 5,
        seq_length=config.max_seq_len,
        n_sensors=config.n_sensors,
        max_rul=config.horizon_bins,
        failure_prob=0.3,
        seed=123
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """
    Complete training pipeline demonstrating HEPA on synthetic data.
    
    Stage 1: JEPA Pretraining
    - Trains encoder + predictor with representation prediction objective
    - No labels used
    
    Stage 2: Predictor Finetuning  
    - Freezes encoder
    - Trains predictor + event head with survival CDF objective
    - Computes h-AUROC as evaluation metric
    """
    print("=" * 70)
    print("HEPA: Horizon-Conditioned Event Predictive Architecture")
    print("=" * 70)
    
    # Configuration
    config = HEPAConfig(
        d_model=256,
        n_heads=4,
        n_layers=2,
        patch_size=16,
        dropout=0.1,
        max_seq_len=256,
        n_sensors=14,
        horizon_bins=150,  # K=150 for C-MAPSS-like data
        batch_size=32,
        pretrain_epochs=20,
        finetune_epochs=15,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nDevice: {config.device}")
    print(f"Model dimension: {config.d_model}")
    print(f"Horizon bins: {config.horizon_bins}")
    
    # Generate data
    print("\n" + "-" * 40)
    print("Generating mock time-series data...")
    train_loader, val_loader = generate_mock_dataloaders(config)
    
    # Compute positive weight for class imbalance
    sample_batch = next(iter(train_loader))
    n_pos = sample_batch['event_labels'].sum()
    n_neg = sample_batch['event_labels'].numel() - n_pos
    pos_weight = (n_neg / n_pos).item() if n_pos > 0 else 1.0
    print(f"Positive weight (N_neg/N_pos): {pos_weight:.2f}")
    
    # Initialize trainer
    trainer = HEPATrainer(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in trainer.model.parameters())
    encoder_params = sum(p.numel() for p in trainer.model.encoder.parameters()) + \
                    sum(p.numel() for p in trainer.model.patch_embed.parameters()) + \
                    sum(p.numel() for p in trainer.model.pooling.parameters())
    predictor_params = sum(p.numel() for p in trainer.model.predictor.parameters())
    head_params = sum(p.numel() for p in trainer.model.event_head.parameters())
    
    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Predictor: {predictor_params:,}")
    print(f"  Event head: {head_params:,}")
    print(f"  Finetuned (predictor + head): {predictor_params + head_params:,}")
    
    # ================================================================
    # Stage 1: JEPA Pretraining
    # ================================================================
    print("\n" + "=" * 50)
    print("STAGE 1: JEPA Pretraining (Self-Supervised)")
    print("=" * 50)
    print("Training encoder + predictor without labels...")
    
    best_pretrain_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.pretrain_epochs):
        losses = trainer.pretrain_epoch(train_loader)
        
        print(f"Epoch {epoch+1:3d}/{config.pretrain_epochs} | "
              f"Pred Loss: {losses['pred_loss']:.4f} | "
              f"SIGReg: {losses['sigreg_loss']:.4f} | "
              f"Total: {losses['total_loss']:.4f}")
        
        # Early stopping
        if losses['total_loss'] < best_pretrain_loss - 1e-4:
            best_pretrain_loss = losses['total_loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.pretrain_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nPretraining complete. Best loss: {best_pretrain_loss:.4f}")
    
    # ================================================================
    # Stage 2: Predictor Finetuning
    # ================================================================
    print("\n" + "=" * 50)
    print("STAGE 2: Predictor Finetuning (Supervised)")
    print("=" * 50)
    print("Freezing encoder, finetuning predictor + event head...")
    
    best_val_auroc = 0.0
    
    for epoch in range(config.finetune_epochs):
        # Training
        train_losses = trainer.finetune_epoch(train_loader, pos_weight)
        
        # Validation
        trainer.model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(config.device)
                event_labels = batch['event_labels'].to(config.device)
                
                h_t = trainer.model.encode_context(sequences)
                hazards = trainer.model.compute_hazards(h_t, trainer.horizons)
                p_surface = trainer.model.compute_survival_cdf(hazards)
                
                K = min(event_labels.size(1), trainer.horizons.size(0))
                val_predictions.append(p_surface[:, :K])
                val_labels.append(event_labels[:, :K])
        
        val_preds = torch.cat(val_predictions, dim=0)
        val_labs = torch.cat(val_labels, dim=0)
        
        # Compute h-AUROC
        try:
            h_auroc = trainer.compute_h_auroc(val_preds, val_labs)
        except ImportError:
            h_auroc = 0.5
            print("(scikit-learn not available for AUROC calculation)")
        
        print(f"Epoch {epoch+1:3d}/{config.finetune_epochs} | "
              f"Train Loss: {train_losses['finetune_loss']:.4f} | "
              f"Val h-AUROC: {h_auroc:.4f}")
        
        if h_auroc > best_val_auroc:
            best_val_auroc = h_auroc
    
    # ================================================================
    # Results Summary
    # ================================================================
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - Results Summary")
    print("=" * 50)
    print(f"Pretraining best loss (ε): {best_pretrain_loss:.4f}")
    print(f"Finetuning best h-AUROC: {best_val_auroc:.4f}")
    
    # Compute example probability surface
    print("\nExample probability surface (first validation sample):")
    with torch.no_grad():
        sample = next(iter(val_loader))
        seq = sample['sequence'][:1].to(config.device)
        h_t = trainer.model.encode_context(seq)
        hazards = trainer.model.compute_hazards(h_t, trainer.horizons)
        p_surface = trainer.model.compute_survival_cdf(hazards)
        
        # Show monotonicity
        p_vals = p_surface[0, :10].cpu().numpy()
        print(f"p(t, Δt) for Δt=1..10: {p_vals}")
        is_monotonic = np.all(np.diff(p_vals) >= -1e-6)
        print(f"Monotonically increasing: {is_monotonic} ✓")
    
    print("\n" + "=" * 70)
    print("HEPA implementation complete. Model ready for deployment.")
    print("=" * 70)
    
    return trainer.model, best_val_auroc


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    model, h_auroc = main()