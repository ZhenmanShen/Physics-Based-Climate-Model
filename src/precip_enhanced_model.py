import torch
import torch.nn as nn
import torch.nn.functional as F

class PrecipitationEnhancedModel(nn.Module):
    def __init__(self, in_channels=5, out_channels=2, embed_dim=128, 
                 depth=4, n_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        
        # Multi-scale encoder with skip connections
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down1 = nn.MaxPool2d(2)  # 48x72 → 24x36
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.MaxPool2d(2)  # 24x36 → 12x18
        
        # Precipitation-specific attention
        self.precip_attn = PrecipitationAttention(embed_dim)
        
        # Transformer module
        self.height = 12
        self.width = 18
        self.num_tokens = self.height * self.width
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.skip_conv1 = nn.Conv2d(embed_dim//2 + embed_dim, embed_dim, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(embed_dim//2 + embed_dim//2, embed_dim//2, kernel_size=1)
        
        # Multi-scale decoder with precipitation-specific head
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2),
            nn.ReLU()
        )  # 12x18 → 24x36
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2),
            nn.ReLU()
        )  # 24x36 → 48x72
        
        # Separate heads for temperature and precipitation
        self.temp_head = nn.Sequential(
            nn.Conv2d(embed_dim//2, embed_dim//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//4, 1, kernel_size=1)
        )
        
        self.precip_head = nn.Sequential(
            nn.Conv2d(embed_dim//2, embed_dim//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//4, 1, kernel_size=1),
            nn.Softplus()  # Ensure positive precipitation values
        )

    def forward(self, x):
        B = x.size(0)
        
        # Encoder pathway with skip connections
        enc1 = self.encoder1(x)       # Full resolution (48x72)
        x = self.down1(enc1)          # 24x36
        
        enc2 = self.encoder2(x)       # Mid resolution
        x = self.down2(enc2)          # 12x18
        
        # Precipitation attention
        x = self.precip_attn(x)       # Enhance precipitation features
        
        # Transformer processing
        x = x.flatten(2).transpose(1, 2)  # (B, 216, embed_dim)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.transpose(1, 2).view(B, -1, self.height, self.width)
        
        # Decoder pathway with skip connections
        x = self.decoder1(x)          # 12x18 → 24x36
        x = torch.cat([x, enc2], dim=1)  # Skip connection
        x = self.skip_conv1(x)
        
        x = self.decoder2(x)          # 24x36 → 48x72
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.skip_conv2(x)
        
        # Separate output heads
        temp = self.temp_head(x)
        precip = self.precip_head(x)
        
        return torch.cat([temp, precip], dim=1)

class PrecipitationAttention(nn.Module):
    """Attention mechanism focused on precipitation-relevant features"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()
        self.moisture_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Keep same channels
        
    def forward(self, x):
        # Channel attention
        residual = x
        x = self.channel_att(x)
        x = self.spatial_att(x)
        moisture = self.moisture_conv(x)
        return residual * torch.sigmoid(moisture)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return x * self.sigmoid(out.view(b, c, 1, 1))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(combined)
        return x * self.sigmoid(att)