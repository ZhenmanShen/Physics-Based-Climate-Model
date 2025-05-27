import torch
import torch.nn as nn

class ClimateTransformer(nn.Module):
    def __init__(self, in_channels=5, out_channels=2, 
                 embed_dim=128, temp_layers=4, spatial_layers=4,
                 n_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        
        # Spatial encoder (per timestep)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim//2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        
        # Temporal processing components
        self.temporal_convs = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim*2, embed_dim, 3, padding=1),
        )
        
        # Factorized spatiotemporal attention
        self.spatial_attn = TransformerEncoder(
            dim=embed_dim,
            depth=spatial_layers,
            heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        self.temporal_attn = TransformerEncoder(
            dim=embed_dim,
            depth=temp_layers,
            heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # Decoder (per timestep)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim//2, embed_dim//4, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, out_channels, 1)
        )

        # Positional embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, 12*18, embed_dim))
        self.temp_pos = nn.Parameter(torch.randn(1, 1000, embed_dim))  # Max sequence length

    def forward(self, x):
        # Input shape: (Time, Channels, Y, X)
        T, C, H, W = x.size()
        
        # Process each timestep
        spatial_features = []
        for t in range(T):
            feat = self.encoder(x[t])  # (E, 12, 18)
            feat = feat.flatten(1).permute(1, 0)  # (H*W, E)
            spatial_features.append(feat)
        
        # Stack temporal features: (T, H*W, E)
        x = torch.stack(spatial_features, dim=0)
        
        # Add positional embeddings
        x = x + self.spatial_pos  # Spatial positions
        x = x.permute(1, 0, 2)  # (H*W, T, E)
        x = x + self.temp_pos[:, :T]  # Temporal positions
        
        # Spatial attention within timesteps
        x = self.spatial_attn(x)  # (H*W, T, E)
        
        # Temporal attention across timesteps
        x = x.permute(1, 0, 2)  # (T, H*W, E)
        x = self.temporal_attn(x)  # (T, H*W, E)
        
        # Reshape and decode
        outputs = []
        for t in range(T):
            feat = x[t].view(12, 18, -1).permute(2, 0, 1)  # (E, 12, 18)
            out = self.decoder(feat.unsqueeze(0)).squeeze(0)  # (C, 48, 72)
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)  # (T, C, Y, X)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x + self._sa_block(x)
        x = x + self._ff_block(x)
        return x
    
    def _sa_block(self, x):
        x = self.norm1(x)
        return self.attn(x, x, x, need_weights=False)[0]
    
    def _ff_block(self, x):
        return self.mlp(self.norm2(x))