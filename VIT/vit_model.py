import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        patch_size,
        num_patches,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()

        self.patch_size = patch_size
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        
    def forward(self, x):        
        B, T, _ = x.shape
        x = self.input_layer(x)
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        cls = x[0]        
        return cls
    
class ImageToPatches(nn.Module):
    def __init__(self, patch_size,flatten_channels=True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten_channels = flatten_channels
    
    def img_to_patch(self,x, patch_size, flatten_channels=True):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        if flatten_channels:
            x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
        return x
    
    def forward(self, x):
        return self.img_to_patch(x, self.patch_size, self.flatten_channels)
    
    
class PatchToImage(nn.Module):
    def __init__(self, patch_size, img_size, num_channels, flatten_channels=True):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_channels = num_channels
        self.flatten_channels = flatten_channels
    
    def patch_to_img(self, x, patch_size, img_size, num_channels, flatten_channels=True):
        B, N, C = x.shape
        p_H = p_W = patch_size
        H = W = img_size
        x = x.unflatten(1, (H // p_H, W // p_W, num_channels * p_H * p_W))
        x = x.permute(0, 3, 1, 4, 2)  # [B, C*p_H*p_W, H', p_W, W']
        x = x.reshape(B, num_channels, H // p_H, p_H, W // p_W, p_W)
        x = x.permute(0, 1, 2, 4, 3, 5)  # [B, C, H', W', p_H, p_W]
        x = x.reshape(B, num_channels, H, W)
        return x
    
    def forward(self, x):
        return self.patch_to_img(x, self.patch_size, self.img_size, self.num_channels, self.flatten_channels)