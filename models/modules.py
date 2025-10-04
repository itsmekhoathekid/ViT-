import torch 
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttentionBlock

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

    def forward(self, x):
        return x + self.pos_embedding

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        self.mlp = MLP(d_model, d_ff, dropout=dropout)
        self.residuals = nn.ModuleList([
            ResidualConnection( d_model, dropout),
            ResidualConnection(d_model, dropout)
        ])
    
    def forward(self, x, mask=None):
        x = self.residuals[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.residuals[1](x, self.mlp)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, n_layers, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, d_model, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (batch, d_model, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (batch, d_model, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, d_model)
        return x


    
