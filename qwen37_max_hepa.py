import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Architectural Components
# ==========================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (per-context instance normalisation).
    Normalizes over the time dimension for each channel independently.
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Time, Sensors)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps
        x_norm = (x - mean) / std
        return x_norm, mean, std

class PatchEmbedding(nn.Module):
    """Tokenizes time series into non-overlapping patches."""
    def __init__(self, patch_size, n_sensors, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * n_sensors, d_model)

    def forward(self, x):
        # x: (B, T, S)
        B, T, S = x.shape
        # Ensure T is divisible by patch_size
        x = x[:, : (T // self.patch_size) * self.patch_size, :]
        # Reshape to (B, num_patches, patch_size * S)
        x = x.reshape(B, -1, self.patch_size * S)
        return self.proj(x)

class HEPA_Encoder(nn.Module):
    """
    Causal Transformer Encoder. 
    Can also act as the bidirectional Target Encoder by disabling the causal mask.
    """
    def __init__(self, d_model, n_heads, n_layers, patch_size, n_sensors, max_patches=64):
        super().__init__()
        self.patch_emb = PatchEmbedding(patch_size, n_sensors, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, causal=True):
        B, T, S = x.shape
        x, _, _ = RevIN(S)(x)
        x = self.patch_emb(x)
        N = x.size(1)
        
        x = x + self.pos_enc[:, :N, :]
        
        mask = None
        if causal:
            # Causal mask: prevent attending to future patches
            mask = torch.triu(torch.ones(N, N, device=x.device) * float('-inf'), diagonal=1)
            
        out = self.transformer(x, mask=mask)
        return self.norm(out)

class AttentionPool(nn.Module):
    """Attention pooling for the Target Encoder to summarize the future window."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        out, _ = self.attn(query, x, x)
        return out.squeeze(1)

class Predictor(nn.Module):
    """
    Horizon-conditioned Predictor MLP.
    Maps encoder output and horizon \Delta t to predicted future representation.
    """
    def __init__(self, d_model, max_horizon_patches):
        super().__init__()
        self.horizon_emb = nn.Embedding(max_horizon_patches + 1, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, h_t, delta_t):
        # h_t: (B, d_model), delta_t: (B,)
        h_emb = self.horizon_emb(delta_t)
        inp = torch.cat([h_t, h_emb], dim=-1)
        return self.mlp(inp)

class EventHead(nn.Module):
    """Lightweight linear event head outputting per-interval conditional hazards."""
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, h):
        return self.fc(h) # Returns logits, sigmoid applied in loss/CDF calculation

# ==========================================
# 2. SIGReg Loss Proxy
# ==========================================

def sigreg_loss(h_hat):
    """
    Sketched Isotropic Gaussian Regularisation (SIGReg).
    We approximate the isotropic Gaussian constraint by forcing the batch 
    covariance matrix to approach the identity matrix (similar to VICReg/Barlow Twins).
    """
    # Center the representations
    h_hat = h_hat - h_hat.mean(dim=0, keepdim=True)
    # Compute covariance matrix
    N, D = h_hat.shape
    cov = (h_hat.T @ h_hat) / (N - 1)
    target = torch.eye(D, device=h_hat.device)
    # Frobenius norm of the difference
    return torch.norm(cov - target, p='fro') ** 2

# ==========================================
# 3. HEPA Architecture Wrapper
# ==========================================

class HEPA(nn.Module):
    def __init__(self, n_sensors, patch_size=16, d_model=256, n_heads=4, n_layers=2, max_horizon_patches=16):
        super().__init__()
        # Online Encoder (Causal) & Target Encoder (Bidirectional) share weights
        self.encoder = HEPA_Encoder(d_model, n_heads, n_layers, patch_size, n_sensors)
        self.attn_pool = AttentionPool(d_model, n_heads)
        
        self.predictor = Predictor(d_model, max_horizon_patches)
        self.event_head = EventHead(d_model)
        
        self.d_model = d_model
        self.patch_size = patch_size

    def pretrain_step(self, x_past, x_future, delta_t_patches, alpha=0.1):
        """
        Stage 1: Self-Supervised JEPA Pretraining
        """
        # 1. Online Encoder (Causal) on past
        h_seq_past = self.encoder(x_past, causal=True)
        h_t = h_seq_past[:, -1, :] # Summary embedding of the past
        
        # 2. Predictor maps h_t and horizon to future representation
        h_hat = self.predictor(h_t, delta_t_patches)
        
        # 3. Target Encoder (Bidirectional, weight-shared) on future
        h_seq_future = self.encoder(x_future, causal=False)
        h_star = self.attn_pool(h_seq_future)
        
        # 4. Loss Calculation
        # L1 loss (chosen over L2 to avoid outlier domination, per paper Sec 3.1)
        loss_l1 = F.l1_loss(h_hat, h_star)
        # SIGReg prevents representation collapse
        loss_sig = sigreg_loss(h_hat)
        
        loss = (1 - alpha) * loss_l1 + alpha * loss_sig
        return loss

    def finetune_step(self, x_past, K_horizons):
        """
        Stage 2: Predictor Finetuning (Encoder Frozen)
        Outputs discrete-time survival CDF.
        """
        with torch.no_grad():
            h_seq_past = self.encoder(x_past, causal=True)
            h_t = h_seq_past[:, -1, :]
            
        B = h_t.size(0)
        hazards = []
        
        # Run predictor for each discrete horizon \Delta t = 1 ... K
        for dt in range(1, K_horizons + 1):
            dt_tensor = torch.full((B,), dt, dtype=torch.long, device=h_t.device)
            h_pred = self.predictor(h_t, dt_tensor)
            logit = self.event_head(h_pred)
            hazards.append(torch.sigmoid(logit))
            
        hazards = torch.cat(hazards, dim=1) # Shape: (B, K)
        
        # Compose into Survival CDF: p(t, \Delta t) = 1 - \prod (1 - \lambda_j)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        cdf = 1.0 - survival
        
        return cdf, hazards

# ==========================================
# 4. Mock Data Generation
# ==========================================

class MockEventDataset(Dataset):
    """
    Generates mock multivariate time series with rare critical events.
    """
    def __init__(self, num_samples=500, seq_len=512, n_sensors=14, patch_size=16):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.n_sensors = n_sensors
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        # Generate random noise data
        self.data = torch.randn(num_samples, seq_len, n_sensors)
        
        # Generate random failure times (in patch indices)
        # Failures occur somewhere in the second half of the sequence
        self.failure_patch_idx = torch.randint(self.num_patches // 2, self.num_patches, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        fail_idx = self.failure_patch_idx[idx]
        
        # Context: first half of patches
        ctx_patches = self.num_patches // 2
        x_past = x[: ctx_patches * self.patch_size, :]
        
        # Future: second half of patches
        x_future = x[ctx_patches * self.patch_size :, :]
        
        # Horizon for pretraining (relative to future window)
        delta_t_patches = fail_idx - ctx_patches
        if delta_t_patches <= 0: delta_t_patches = 1 # Ensure positive horizon
        
        # Target CDF for finetuning (over the future window patches)
        K = self.num_patches // 2
        y_cdf = torch.zeros(K)
        if delta_t_patches < K:
            y_cdf[delta_t_patches:] = 1.0 # Event occurs, CDF jumps to 1 and stays 1
            
        return x_past, x_future, delta_t_patches, y_cdf

# ==========================================
# 5. Training Loops
# ==========================================

def train_stage1(model, dataloader, optimizer, epochs=5):
    print("--- Stage 1: Self-Supervised Pretraining (JEPA) ---")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_past, x_future, delta_t, _ in dataloader:
            optimizer.zero_grad()
            loss = model.pretrain_step(x_past, x_future, delta_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Pretrain Loss: {total_loss/len(dataloader):.4f}")

def train_stage2(model, dataloader, optimizer, epochs=5, K_horizons=16):
    print("\n--- Stage 2: Predictor Finetuning (Survival CDF) ---")
    # Freeze Encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    model.predictor.train()
    model.event_head.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for x_past, _, _, y_cdf in dataloader:
            optimizer.zero_grad()
            cdf, _ = model.finetune_step(x_past, K_horizons)
            
            # Positive-weighted BCE (Sec 3.2)
            # Calculate pos_weight dynamically for the batch
            pos_count = (y_cdf == 1).sum().clamp(min=1)
            neg_count = (y_cdf == 0).sum().clamp(min=1)
            pos_weight = (neg_count / pos_count).to(cdf.device)
            
            loss = F.binary_cross_entropy(cdf, y_cdf, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Finetune BCE Loss: {total_loss/len(dataloader):.4f}")

# ==========================================
# 6. Main Execution
# ==========================================

if __name__ == "__main__":
    # Hyperparameters matching paper where applicable
    N_SENSORS = 14
    PATCH_SIZE = 16
    SEQ_LEN = 512
    D_MODEL = 256
    N_HEADS = 4
    N_LAYERS = 2
    MAX_HORIZON_PATCHES = 16
    
    # 1. Setup Mock Data
    dataset = MockEventDataset(num_samples=200, seq_len=SEQ_LEN, n_sensors=N_SENSORS, patch_size=PATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize HEPA
    model = HEPA(
        n_sensors=N_SENSORS, 
        patch_size=PATCH_SIZE, 
        d_model=D_MODEL, 
        n_heads=N_HEADS, 
        n_layers=N_LAYERS,
        max_horizon_patches=MAX_HORIZON_PATCHES
    )
    
    # 3. Stage 1: Pretraining
    # Paper notes: "Pretraining takes under one minute per dataset... both encoders are trained jointly"
    optimizer_pretrain = torch.optim.AdamW(
        list(model.encoder.parameters()) + 
        list(model.predictor.parameters()) + 
        list(model.attn_pool.parameters()), 
        lr=3e-4, weight_decay=1e-2
    )
    train_stage1(model, dataloader, optimizer_pretrain, epochs=3)
    
    # 4. Stage 2: Finetuning
    # Paper notes: "freeze the encoder and finetune only the predictor alongside a lightweight event head"
    optimizer_finetune = torch.optim.AdamW(
        list(model.predictor.parameters()) + 
        list(model.event_head.parameters()), 
        lr=1e-3, weight_decay=1e-2
    )
    train_stage2(model, dataloader, optimizer_finetune, epochs=3, K_horizons=MAX_HORIZON_PATCHES)
    
    print("\nTraining pipeline completed successfully.")