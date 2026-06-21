import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =============================================================================
# 1. Architektur-Komponenten (basierend auf Paper Spezifikationen)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encodings (Paper Abschnitt 3.1)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, SeqLen, d_model)
        return x + self.pe[:, :x.size(1), :]

class HEPAEncoder(nn.Module):
    """
    Causal Transformer Encoder f_θ (Paper Abschnitt 3.1 & Table 7)
    - d = 256, 2 Layers, 4 Heads
    - Patch size P = 16
    - Per-context instance normalisation
    """
    def __init__(self, in_channels, d_model=256, num_layers=2, nhead=4, patch_size=16, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Linear projection of non-overlapping patches
        self.patch_proj = nn.Linear(in_channels * patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling (Für die Target-Branch beschrieben in Abschnitt 3.1)
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention_pool = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)

    def forward(self, x, causal=True, pool_output=False):
        """
        x: (Batch, Channels, SeqLen)
        causal: Boolean, True für Online-Encoder, False für Target-Encoder (bidirektional)
        """
        B, C, L = x.shape
        
        # 1. Per-context instance normalisation
        # Instanzen normalisieren über die Zeitdimension
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
        
        # 2. Tokenise into non-overlapping patches
        num_patches = L // self.patch_size
        # Shape: (B, C, NumPatches, PatchSize)
        x = x.view(B, C, num_patches, self.patch_size)
        # Shape: (B, NumPatches, C * PatchSize)
        x = x.permute(0, 2, 1, 3).reshape(B, num_patches, C * self.patch_size)
        
        # Projection & Positional Encoding
        x = self.patch_proj(x)
        x = self.pos_encoder(x)
        
        # Causal Mask generieren, falls online branch
        mask = None
        if causal:
            mask = nn.Transformer.generate_square_subsequent_mask(num_patches).to(x.device)
            
        # Transformer anwenden
        out = self.transformer(x, mask=mask, is_causal=causal)
        
        if pool_output:
            # Bidirectional mit Attention Pooling (Abschnitt 3.1: "with attention pooling")
            q = self.pooling_query.repeat(B, 1, 1)
            # Query the transformer outputs
            pooled, _ = self.attention_pool(q, out, out)
            return pooled.squeeze(1) # Shape: (B, d_model)
        else:
            # Rückgabe des letzten Zeitschritts h_t für den kausalen Online-Encoder
            return out[:, -1, :] # Shape: (B, d_model)

class HEPAPredictor(nn.Module):
    """
    Predictor g_φ (Paper Abschnitt 3.1 & Anhang C)
    - 3-layer MLP mapping [h_t; Δt] -> h_hat
    """
    def __init__(self, d_model=256, max_horizon=200):
        super().__init__()
        # Learned horizon embedding (aus Anhang C)
        self.horizon_embed = nn.Embedding(max_horizon + 1, d_model)
        
        # 3-Layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, h_t, delta_t):
        # h_t: (B, d_model), delta_t: (B,)
        dt_emb = self.horizon_embed(delta_t)
        # Concatenate [h_t; Δt]
        x = torch.cat([h_t, dt_emb], dim=-1)
        return self.mlp(x)

class HEPAEventHead(nn.Module):
    """
    Lightweight Event Head (Paper Abschnitt 3.2 & Anhang C)
    - LayerNorm + Linear logit
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h_hat):
        # h_hat: (B, d_model)
        # Produces discrete hazard λ = σ(w^T * h_hat + b)
        return torch.sigmoid(self.linear(self.norm(h_hat)))


# =============================================================================
# 2. Verlustfunktionen
# =============================================================================

def sigreg_loss(h_pred):
    """
    Sketched Isotropic Gaussian Regularisation (SIGReg) (Abschnitt 3.1)
    Verhindert Representation Collapse. Constrains toward isotropic Gaussian.
    """
    B, D = h_pred.shape
    if B <= 1:
        return torch.tensor(0.0, device=h_pred.device)
        
    h_centered = h_pred - h_pred.mean(dim=0, keepdim=True)
    cov = (h_centered.T @ h_centered) / (B - 1)
    eye = torch.eye(D, device=h_pred.device)
    
    # MSE zwischen Kovarianz und Identitätsmatrix
    return F.mse_loss(cov, eye)

def pretraining_loss(h_pred, h_target, alpha=0.1):
    """
    JEPA Objective + SIGReg (Gleichung 2)
    L = (1 - α) ||h - h*||_1 + α L_SIG
    L1 on L2-normalised representations (Table 7)
    """
    # L2-Normalisierung
    h_pred_norm = F.normalize(h_pred, p=2, dim=-1)
    h_target_norm = F.normalize(h_target, p=2, dim=-1)
    
    l1_loss = F.l1_loss(h_pred_norm, h_target_norm)
    sig_loss = sigreg_loss(h_pred)
    
    return (1 - alpha) * l1_loss + alpha * sig_loss


# =============================================================================
# 3. Haupt-Trainingslogik
# =============================================================================

class HEPAModel(nn.Module):
    def __init__(self, in_channels, max_horizon=200):
        super().__init__()
        self.encoder = HEPAEncoder(in_channels=in_channels, patch_size=16)
        self.predictor = HEPAPredictor(d_model=256, max_horizon=max_horizon)
        self.event_head = HEPAEventHead(d_model=256)
        self.max_horizon = max_horizon

    def forward_pretrain(self, x_past, x_future, delta_t):
        """Phase 1: Self-Supervised JEPA"""
        # Online Branch (Kausal)
        h_t = self.encoder(x_past, causal=True, pool_output=False)
        
        # Target Branch (Gleiche Gewichte, bidirektional, Attention Pooling)
        with torch.no_grad(): # Obwohl das Paper Joint Training sagt, erfordern einige Implementierungen No-Grad für Targets. 
                              # Paper Abschnitt 3.1 sagt: "Both encoders are trained jointly via the optimizer... no stop-gradient is needed".
                              pass
                              
        h_target = self.encoder(x_future, causal=False, pool_output=True)
        
        # Predictor
        h_pred = self.predictor(h_t, delta_t)
        return h_pred, h_target

    def forward_finetune(self, x_past):
        """Phase 2: Supervised Predictor Finetuning - Generiert Survival CDF"""
        # Encoder ist eingefroren
        with torch.no_grad():
            h_t = self.encoder(x_past, causal=True, pool_output=False)
            
        hazards = []
        # Evaluierung über alle K Horizonte (Abschnitt 3.2)
        for dt in range(1, self.max_horizon + 1):
            dt_tensor = torch.full((x_past.size(0),), dt, device=x_past.device, dtype=torch.long)
            h_hat_dt = self.predictor(h_t, dt_tensor)
            lambda_dt = self.event_head(h_hat_dt) # Hazard Rate
            hazards.append(lambda_dt)
            
        hazards = torch.cat(hazards, dim=1) # Shape: (B, K)
        
        # Survival CDF berechnen: p(t, Δt) = 1 - ∏(1 - λ_j) (Gleichung 4)
        # Für numerische Stabilität nutzen wir log(1-λ)
        log_survival = torch.log(1.0 - hazards + 1e-8)
        cumulative_log_survival = torch.cumsum(log_survival, dim=1)
        p_cdf = 1.0 - torch.exp(cumulative_log_survival) # Shape: (B, K)
        
        return p_cdf


# =============================================================================
# 4. Dummy Daten & Trainingsausführung
# =============================================================================

def train_hepa():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benutze Device: {device}")
    
    # Hyperparameter aus dem Paper (Table 7)
    batch_size = 64
    in_channels = 14 # Angelehnt an C-MAPSS FD001
    seq_len_past = 512 # 32 Patches à 16 (Table 3 / L)
    seq_len_future = 64 # Future Window (z.B. 4 Patches)
    max_horizon = 150 # K=150 für C-MAPSS
    
    model = HEPAModel(in_channels=in_channels, max_horizon=max_horizon).to(device)
    
    print("\n--- Start Stage 1: Self-Supervised Pretraining ---")
    model.train()
    # Optimizer (Pretraining: AdamW, LR 3e-4, WD 1e-2)
    optimizer_pre = optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()), 
        lr=3e-4, weight_decay=1e-2
    )
    
    # Mock Data für Stage 1
    for epoch in range(1, 4): # Dummy Loop
        optimizer_pre.zero_grad()
        
        x_past = torch.randn(batch_size, in_channels, seq_len_past).to(device)
        x_future = torch.randn(batch_size, in_channels, seq_len_future).to(device)
        
        # Sample Δt Log-Uniform [1, max_horizon]
        u = torch.rand(batch_size).to(device)
        log_min, log_max = math.log(1), math.log(max_horizon)
        delta_t = torch.exp(log_min + u * (log_max - log_min)).long().clamp(1, max_horizon)
        
        h_pred, h_target = model.forward_pretrain(x_past, x_future, delta_t)
        loss_pre = pretraining_loss(h_pred, h_target, alpha=0.1)
        
        loss_pre.backward()
        optimizer_pre.step()
        
        print(f"Pretraining Epoche {epoch} | Loss: {loss_pre.item():.4f}")


    print("\n--- Start Stage 2: Supervised Predictor Finetuning ---")
    # Encoder einfrieren (Abschnitt 3.2: "freeze the encoder and finetune only the predictor...")
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # Optimizer (Finetuning: AdamW, LR 1e-3, WD 1e-2)
    optimizer_ft = optim.AdamW(
        list(model.predictor.parameters()) + list(model.event_head.parameters()), 
        lr=1e-3, weight_decay=1e-2
    )
    
    # Mock Data für Stage 2
    w_plus = 5.0 # Kompensation für Klassen-Ungleichgewicht N_neg / N_pos (Abschnitt 3.2)
    
    for epoch in range(1, 4): # Dummy Loop
        optimizer_ft.zero_grad()
        
        x_past = torch.randn(batch_size, in_channels, seq_len_past).to(device)
        
        # Zufällige Event-Indikatoren (Ground Truth Survival Function y(t, Δt))
        y_true = torch.randint(0, 2, (batch_size, max_horizon)).float().to(device)
        # Mache y_true monoton steigend (wie eine echte CDF)
        y_true, _ = torch.cummax(y_true, dim=1) 
        
        p_cdf = model.forward_finetune(x_past)
        
        # Positive-weighted BCE Loss (Gleichung 5)
        # Implementierung mit manuellen Gewichten (w_plus auf den positiven Samples)
        bce = F.binary_cross_entropy(p_cdf, y_true, reduction='none')
        weight_mask = torch.where(y_true == 1, w_plus, 1.0)
        loss_ft = (bce * weight_mask).sum(dim=1).mean() # Sum over horizons, mean over batch
        
        loss_ft.backward()
        optimizer_ft.step()
        
        print(f"Finetuning Epoche {epoch} | Weighted BCE Loss: {loss_ft.item():.4f}")

if __name__ == "__main__":
    train_hepa()