import torch
import torch.nn as nn
from typing import Tuple

# ── tiny attention helpers ────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Channel-wise squeeze-and-excitation (ratio = 8)."""
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))

class SpatialGate(nn.Module):
    """1×1 conv on concatenated max- & avg-pool maps (CBAM style)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mxx = x.amax(1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mxx], dim=1)))
        return x * gate

# ── building blocks ───────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in,  c_out, 3, padding=1, bias=False),
            nn.GroupNorm(8, c_out), nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(8, c_out), nn.SiLU(inplace=True),
        )
        self.se   = SEBlock(c_out)
        self.spat = SpatialGate()

    def forward(self, x):
        x = self.body(x)
        x = self.se(x)
        x = self.spat(x)
        return x

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(c_in, c_out)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up   = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)
        self.conv = ConvBlock(c_out + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ── the network ───────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    Depth-4 UNet with attention.
        params ≈ 645 k for in_ch=5, out_ch=2, base=16
    """

    def __init__(self, in_ch: int = 5, out_ch: int = 2, base: int = 16):
        super().__init__()

        # Encoder --------------------------------------------------------------
        self.enc1 = ConvBlock(in_ch,        base)
        self.enc2 = Down(base,              base * 2)
        self.enc3 = Down(base * 2,          base * 4)
        self.enc4 = Down(base * 4,          base * 8)

        # Bottleneck (no pooling) ---------------------------------------------
        self.bott = ConvBlock(base * 8,     base * 8)

        # Decoder --------------------------------------------------------------
        self.up3  = Up(base * 8, base * 4, base * 4)
        self.up2  = Up(base * 4, base * 2, base * 2)
        self.up1  = Up(base * 2, base,     base)

        # Head -----------------------------------------------------------------
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)          # (b, 16, 48, 72)
        s2 = self.enc2(s1)         # (b, 32, 24, 36)
        s3 = self.enc3(s2)         # (b, 64, 12, 18)
        s4 = self.enc4(s3)         # (b,128,  6,  9)

        x  = self.bott(s4)         # (b,128,  6,  9)

        x  = self.up3(x, s3)       # (b, 64, 12, 18)
        x  = self.up2(x, s2)       # (b, 32, 24, 36)
        x  = self.up1(x, s1)       # (b, 16, 48, 72)
        return self.head(x)
