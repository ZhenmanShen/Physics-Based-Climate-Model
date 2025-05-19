import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "unetAttention":
        model = UNet(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model

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

class AttentionGate(nn.Module):
    """Attention Gate for skip connections."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels):
        super().__init__()
        # Encoder
        self.enc1 = self._block(n_input_channels, 64)  # (B, 64, 48, 72)
        self.enc2 = self._block(64, 128)  # (B, 128, 24, 36)
        self.enc3 = self._block(128, 256)  # (B, 256, 12, 18)
        self.enc4 = self._block(256, 512)  # (B, 512, 6, 9)
        
        # Bottleneck with SE
        self.bottleneck = self._block(512, 1024)
        self.se = SEBlock(1024)
        
        # Fixed Attention Gates
        self.attn3 = AttentionGate(F_g=1024, F_l=512, F_int=512)  # Changed from (512,256,256)
        self.attn2 = AttentionGate(F_g=512, F_l=256, F_int=256)   # Changed from (256,128,128)
        self.attn1 = AttentionGate(F_g=256, F_l=128, F_int=128)   # Changed from (128,64,64)
        # Corrected Decoder Blocks
        self.dec3 = self._block(1024 + 512, 512)  # Was 1024+256
        self.dec2 = self._block(512 + 256, 256)    # Was 512+128
        self.dec1 = self._block(256 + 128, 128)     # Was 256+64
        
        # Output
        self.out = nn.Conv2d(128, n_output_channels, 1)  # (B, 2, 48, 72)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder (unchanged)
        e1 = self.enc1(x)        # (B,64,48,72)
        p1 = self.pool(e1)       # (B,64,24,36)
        e2 = self.enc2(p1)       # (B,128,24,36)
        p2 = self.pool(e2)       # (B,128,12,18)
        e3 = self.enc3(p2)       # (B,256,12,18)
        p3 = self.pool(e3)       # (B,256,6,9)
        e4 = self.enc4(p3)       # (B,512,6,9)
    
        # Bottleneck (unchanged)
        b = self.pool(e4)        # (B,512,3,4)
        b = self.bottleneck(b)   # (B,1024,3,4)
        b = self.se(b)
    
        # Decoder Stage 4->3 (FIXED ORDER)
        d4 = self.upsample(b)    # (B,1024,6,8)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear')  # (B,1024,6,9)
        a3 = self.attn3(d4, e4)  # Now both are (6,9)
        d4 = torch.cat([d4, a3], dim=1)  # (B,1536,6,9)
        d3 = self.dec3(d4)       # (B,512,6,9)
    
        # Decoder Stage 3->2 (FIXED ORDER)
        d3_up = self.upsample(d3)  # (B,512,12,18)
        d3_up = F.interpolate(d3_up, size=e3.shape[2:], mode='bilinear')  # Match e3
        a2 = self.attn2(d3_up, e3)  # Both (12,18)
        d3_up = torch.cat([d3_up, a2], dim=1)  # (B,768,12,18)
        d2 = self.dec2(d3_up)     # (B,256,12,18)
    
        # Decoder Stage 2->1 (FIXED ORDER)
        d2_up = self.upsample(d2)  # (B,256,24,36)
        d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode='bilinear')  # Match e2
        a1 = self.attn1(d2_up, e2)  # Both (24,36)
        d2_up = torch.cat([d2_up, a1], dim=1)  # (B,384,24,36)
        d1 = self.dec1(d2_up)     # (B,128,24,36)
    
        # Final output
        d1 = self.upsample(d1)    # (B,128,48,72)
        out = self.out(d1)        # (B,2,48,72)
        return out