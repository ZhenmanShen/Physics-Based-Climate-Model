import torch
import torch.nn as nn

class CNNTransformer(nn.Module):
    def __init__(self, in_channels=5, out_channels=2, embed_dim=128, depth=4, n_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()

        # CNN Encoder: reduce spatial size from (48, 72) → (12, 18)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),  # → 24x36
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),    # → 12x18
            nn.ReLU(),
        )

        self.height = 12
        self.width = 18
        self.num_tokens = self.height * self.width
        self.embed_dim = embed_dim

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # CNN Decoder: upsample to 48x72 and output 2 variables
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),  # → 24x36
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),  # → 48x72
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, out_channels, kernel_size=1)  # → output (tas, pr)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)  # → (B, embed_dim, 12, 18)

        x = x.flatten(2).transpose(1, 2)  # → (B, 216, embed_dim)
        x = x + self.pos_embedding

        x = self.transformer(x)  # → (B, 216, embed_dim)

        x = x.transpose(1, 2).view(B, self.embed_dim, self.height, self.width)  # → (B, embed_dim, 12, 18)
        x = self.decoder(x)  # → (B, 2, 48, 72)
        return x
