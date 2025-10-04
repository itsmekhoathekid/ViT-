from .attention import MultiHeadAttentionBlock
import torch
import torch.nn as nn
from .modules import *
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, d_model, n_heads, d_ff, n_layers, dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_embedding = PatchEmbedding(in_channels, d_model, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = PositionEmbedding(self.num_patches, d_model)
        self.transformer = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x, mask = None):
        x = self.patch_embedding(x)  # (batch, num_patches, d_model)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, num_patches + 1, d_model)
        x = self.pos_embedding(x)  # (batch, num_patches + 1, d_model)
        x = self.transformer(x, mask)  # (batch, num_patches + 1, d_model)
        cls_output = x[:, 0]  # (batch, d_model)
        x = self.mlp_head(cls_output)  # (batch, num_classes)
        return x

