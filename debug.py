from models import ViT
import torch 

img_size = 32
patch_size = 4
in_channels = 3
num_classes = 10
d_model = 128
n_heads = 8
d_ff = 256
n_layers = 1
dropout = 0.1

x = torch.randn(2, in_channels, img_size, img_size)  # Example input tensor
model = ViT(img_size, patch_size, in_channels, num_classes, d_model, n_heads, d_ff, n_layers, dropout)
out = model(x)
print("Output shape:", out.shape)  # Should be (2, num_classes)