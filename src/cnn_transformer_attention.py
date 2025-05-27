import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attention

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attention

        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CNNTransformerAttention(nn.Module):
    def __init__(self, in_channels=5, out_channels=2, embed_dim=128, depth=4, n_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),  # → 24x36
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),    # → 12x18
            nn.ReLU(),
        )

        self.cbam = CBAM(embed_dim)  # Replaces SEBlock

        self.height = 12
        self.width = 18
        self.num_tokens = self.height * self.width
        self.embed_dim = embed_dim

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)         # (B, embed_dim, 12, 18)
        x = self.cbam(x)            # CBAM after encoding

        x = x.flatten(2).transpose(1, 2)  # (B, 216, embed_dim)
  
        x = x + self.pos_embedding       # (B, 216, embed_dim)

        x = self.transformer(x)          # (B, 216, embed_dim)

        x = x.transpose(1, 2).view(B, self.embed_dim, self.height, self.width)
        x = self.decoder(x)

        return x
