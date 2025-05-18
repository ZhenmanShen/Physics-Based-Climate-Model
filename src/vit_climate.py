import torch
import torch.nn as nn

class ViTClimate(nn.Module):
    def __init__(self, in_channels=5, out_channels=2, height=48, width=72, embed_dim=128, depth=4, n_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()

        self.num_tokens = height * width
        self.embed_dim = embed_dim

        # Input projection: 5 → embed_dim
        self.input_proj = nn.Linear(in_channels, embed_dim)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Output projection: embed_dim → 2 (tas, pr)
        self.output_proj = nn.Linear(embed_dim, out_channels)

        self.height = height
        self.width = width

    def forward(self, x):
        B, C, H, W = x.shape  # (B, 5, 48, 72)
        assert H == self.height and W == self.width, "Input size mismatch"

        # Reshape: (B, C, H, W) → (B, H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)

        # Embed + add position
        x = self.input_proj(x) + self.pos_embedding  # (B, num_tokens, embed_dim)

        # Transformer
        x = self.transformer(x)  # (B, num_tokens, embed_dim)

        # Predict outputs
        x = self.output_proj(x)  # (B, num_tokens, 2)

        # Reshape: (B, H*W, 2) → (B, 2, H, W)
        x = x.view(B, self.height, self.width, -1).permute(0, 3, 1, 2)

        return x
