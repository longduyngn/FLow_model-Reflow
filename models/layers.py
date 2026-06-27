import math
import torch
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # Handle cases where t is (batch_size, 1) instead of (batch_size,)
        if len(t.shape) == 2:
            t = t.squeeze(-1)
            
        # Scale t from [0, 1] to [0, 1000] for standard sinusoidal embedding frequencies
        t = t * 1000.0
        
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # Pad with zeros if dim is odd
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
            
        return emb
