import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. Mock Data Generation
# ==============================================================================
class MockTimeSeriesDataset(Dataset):
    """
    Generates mock multivariate time series with rare events.
    Context length = 128, Max Horizon = 32.
    """
    def __init__(self, num_samples=512, context_len=128, max_horizon=32, num_features=14):
        self.num_samples = num_samples
        self.context_len = context_len
        self.max_horizon = max_horizon
        self.num_features = num_features
        
        # Generate random walk mock data
        self.data = torch.cumsum(torch.randn(num_samples, context_len + max_horizon, num_features) * 0.1, dim=1)
        
        # Generate cumulative event labels: y(t, \Delta t) = 1 if event in (t, t+\Delta t]
        self.labels = torch.zeros(num_samples, max_horizon)
        for i in range(num_samples):
            if np.random.rand() < 0.20:  # 20% chance of an event occurring in the future window
                t_event = np.random.randint(1, max_horizon + 1)
                self.labels[i, t_event-1:] = 1.0  # Cumulative indicator

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ==============================================================================
# 2. Architecture Components
# ==============================================================================
def per_context_instance_norm(x, eps=1e-5):
    """Per-context instance normalisation (Section 3.1)"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + eps
    return (x - mean) / std

class PatchEmbedding(nn.Module):
    def __init__(self, num_features, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * num_features, d_model)
        
    def forward(self, x):
        B, L, C = x.shape
        # Pad if necessary
        if L % self.patch_size != 0:
            pad_len = self.patch_size - (L % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            L = L + pad_len
        x = x.reshape(B, L // self.patch_size, self.patch_size * C)
        return self.proj(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class TimeSeriesEncoder(nn.Module):
    """
    Causal Transformer for Context Encoder.
    Bidirectional Transformer + Attention Pooling for Target Encoder (Weight-shared).
    """
    def __init__(self, num_features, patch_size, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(num_features, patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x, causal=False):
        x = per_context_instance_norm(x)
        x = self.patch_embed(x)
        x = self.pos_encoder(x)
        
        if causal:
            sz = x.size(1)
            mask = torch.triu(torch.ones(sz, sz, device=x.device) * float('-inf'), diagonal=1)
            x = self.transformer(x, mask=mask)
            h = x[:, -1, :]  # Summary embedding h_t
        else:
            x = self.transformer(x)
            h = x.mean(dim=1)  # Attention pooling (approximated via mean pooling for mock)
        return h

class Predictor(nn.Module):
    """Horizon-conditioned predictor g_\phi (Section 3.1)"""
    def __init__(self, d_model=256, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
    def forward(self, h_t, delta_t_norm):
        # delta_t_norm is normalized to [0, 1]
        x = torch.cat([h_t, delta_t_norm], dim=-1)
        return self.mlp(x)

def sigreg_loss(h_hat):
    """
    Sketched Isotropic Gaussian Regularisation (SIGReg) (Section 3.1).
    Prevents representation collapse by pushing batch embeddings toward N(0, I).
    """
    B, D = h_hat.shape
    h_hat = h_hat - h_hat.mean(dim=0)
    
    # Variance penalty (push std to 1)
    std = h_hat.std(dim=0)
    var_loss = F.relu(1 - std).mean()
    
    # Covariance penalty (push off-diagonal to 0)
    cov = (h_hat.T @ h_hat) / (B - 1)
    off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
    cov_loss = off_diag.pow(2).mean()
    
    return var_loss + cov_loss

# ==============================================================================
# 3. HEPA Model Wrapper
# ==============================================================================
class HEPA(nn.Module):
    def __init__(self, num_features, patch_size, d_model=64, max_horizon=32):
        super().__init__()
        self.encoder = TimeSeriesEncoder(num_features, patch_size, d_model)
        self.predictor = Predictor(d_model, hidden_dim=d_model*2)
        self.event_head = nn.Linear(d_model, 1)
        self.max_horizon = max_horizon
        
    def pretrain_step(self, x_context, x_target, delta_t_norm):
        # Joint-Embedding Predictive Architecture (JEPA) forward pass
        h_t = self.encoder(x_context, causal=True)
        h_star = self.encoder(x_target, causal=False) # Target encoder (weight-shared)
        
        h_hat = self.predictor(h_t, delta_t_norm)
        
        # L1 prediction objective
        l1_loss = F.l1_loss(h_hat, h_star)
        # SIGReg regulariser
        l_sig = sigreg_loss(h_hat)
        
        return l1_loss, l_sig
        
    def finetune_step(self, x_context):
        B = x_context.size(0)
        h_t = self.encoder(x_context, causal=True)
        
        hazards = []
        # Sweep over discrete horizons \Delta t = 1, ..., K
        for j in range(1, self.max_horizon + 1):
            delta_t_norm = torch.full((B, 1), j / self.max_horizon, device=x_context.device)
            h_hat_j = self.predictor(h_t, delta_t_norm)
            lambda_j = torch.sigmoid(self.event_head(h_hat_j)) # Per-interval conditional hazard
            hazards.append(lambda_j)
            
        hazards = torch.cat(hazards, dim=-1) # (B, K)
        
        # Discrete-time survival CDF: p(t, \Delta t) = 1 - \prod (1 - \lambda_j)
        survival = torch.cumprod(1 - hazards, dim=-1)
        cdf = 1 - survival
        return cdf

# ==============================================================================
# 4. Training Loops
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters (scaled down for fast mock testing)
    BATCH_SIZE = 32
    CONTEXT_LEN = 128
    MAX_HORIZON = 32
    NUM_FEATURES = 14
    PATCH_SIZE = 16
    D_MODEL = 64 
    ALPHA = 0.1  # Mixing weight for SIGReg
    
    dataset = MockTimeSeriesDataset(num_samples=256, context_len=CONTEXT_LEN, max_horizon=MAX_HORIZON, num_features=NUM_FEATURES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = HEPA(num_features=NUM_FEATURES, patch_size=PATCH_SIZE, d_model=D_MODEL, max_horizon=MAX_HORIZON).to(device)
    
    # ==========================================
    # STAGE 1: Self-Supervised Pretraining
    # ==========================================
    print("\n--- STAGE 1: Self-Supervised Pretraining (JEPA) ---")
    optimizer_pre = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    
    model.train()
    for epoch in range(3):
        epoch_l1, epoch_sig = 0, 0
        for data, _ in loader:
            data = data.to(device)
            x_context = data[:, :CONTEXT_LEN, :]
            
            # Sample \Delta t from log-uniform distribution (Section 3.1)
            log_dt = np.random.uniform(np.log(1), np.log(MAX_HORIZON))
            dt_val = int(np.clip(np.round(np.exp(log_dt)), 1, MAX_HORIZON))
            
            x_target = data[:, CONTEXT_LEN:CONTEXT_LEN+dt_val, :]
            delta_t_norm = torch.full((BATCH_SIZE, 1), dt_val / MAX_HORIZON, device=device)
            
            l1, l_sig = model.pretrain_step(x_context, x_target, delta_t_norm)
            loss = (1 - ALPHA) * l1 + ALPHA * l_sig
            
            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()
            
            epoch_l1 += l1.item()
            epoch_sig += l_sig.item()
            
        print(f"Epoch {epoch+1} | L1 Loss: {epoch_l1/len(loader):.4f} | SIGReg Loss: {epoch_sig/len(loader):.4f}")

    # ==========================================
    # STAGE 2: Predictor Finetuning
    # ==========================================
    print("\n--- STAGE 2: Predictor Finetuning ---")
    # Freeze the encoder (Section 3.2)
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # Finetune only the predictor and event head
    optimizer_ft = torch.optim.AdamW(
        list(model.predictor.parameters()) + list(model.event_head.parameters()), 
        lr=1e-3, weight_decay=1e-2
    )
    
    # Calculate positive weight for class imbalance (w+ = N_neg / N_pos)
    all_labels = dataset.labels.flatten()
    pos_weight = (all_labels == 0).sum() / ((all_labels == 1).sum() + 1e-5)
    pos_weight = torch.tensor(pos_weight, device=device)
    
    model.train()
    for epoch in range(3):
        epoch_loss = 0
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            x_context = data[:, :CONTEXT_LEN, :]
            
            cdf = model.finetune_step(x_context)
            
            # Positive-weighted BCE over horizons (Section 3.2)
            bce = F.binary_cross_entropy(cdf, labels, reduction='none')
            weight = torch.where(labels == 1, pos_weight, 1.0)
            loss = (bce * weight).mean()
            
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Finetune BCE Loss: {epoch_loss/len(loader):.4f}")

    # ==========================================
    # INFERENCE TEST
    # ==========================================
    print("\n--- Inference Test ---")
    model.eval()
    with torch.no_grad():
        sample_data = dataset[0][0].unsqueeze(0).to(device)
        x_context = sample_data[:, :CONTEXT_LEN, :]
        prob_surface = model.finetune_step(x_context)
        
        print(f"Output Probability Surface p(t, \Delta t) shape: {prob_surface.shape}")
        print(f"Sample cumulative probabilities across horizons (first 10 steps):\n{prob_surface[0, :10].cpu().numpy()}")
        print("Note: Probabilities are monotonically non-decreasing by construction (Survival CDF).")

if __name__ == "__main__":
    main()