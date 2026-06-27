import torch
import torch.nn as nn
from models.layers import SinusoidalEmbedding

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.GELU()
        
    def forward(self, x, t_emb):
        # Inject time embedding by simple summation before layers
        h = x + t_emb
        h = self.fc1(h)
        h = self.ln1(h)
        h = self.act1(h)
        h = self.fc2(h)
        h = self.ln2(h)
        h = self.act2(h)
        return x + h

class VelocityNet(nn.Module):
    def __init__(self, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # Spatial coordinate input projection
        self.x_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU()
        )
        
        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim)
        ])
        
        # Final prediction output projection
        self.out_proj = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, t):
        # x: (batch_size, 2)
        # t: (batch_size,) or (batch_size, 1)
        t_emb = self.time_embed(t)
        h = self.x_proj(x)
        
        # Forward through residual blocks
        for block in self.blocks:
            h = block(h, t_emb)
            
        # Output prediction (v_t)
        return self.out_proj(h)
