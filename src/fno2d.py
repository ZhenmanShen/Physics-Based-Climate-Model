import torch
import torch.nn as nn
import torch.fft

class SpectralConv2d(nn.Module):
    """2-D Fourier layer with complex weight modulation."""
    def __init__(self, in_ch, out_ch, modes_lat, modes_lon):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes_lat, self.modes_lon = modes_lat, modes_lon
        # complex weights for the retained modes
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes_lat, modes_lon, dtype=torch.cfloat)
        )

    def compl_mul(self, a, b):
        # (a+ib)(c+id) = (ac−bd) + i(ad+bc)
        return torch.stack([
            a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real
        ], dim=-1).sum(2)  # sum over in_ch

    def forward(self, x):                                # x: [B,C,H,W]  (float32)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")          # [B,C,H, W/2+1]
        # allocate output Fourier tensor
        out_ft = torch.zeros(B, self.out_ch, H, W//2 + 1, dtype=torch.cfloat, device=x.device)
        lat_slice = slice(0, self.modes_lat)
        lon_slice = slice(0, self.modes_lon)
        out_ft[:, :, lat_slice, lon_slice] = self.compl_mul(
            x_ft[:, :, lat_slice, lon_slice], self.weight
        )
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x_out

class FNOBlock(nn.Module):
    def __init__(self, ch, modes_lat, modes_lon):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes_lat, modes_lon)
        self.linear   = nn.Conv2d(ch, ch, 1)
        self.act      = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.linear(x))

class FNO2D(nn.Module):
    """Four-layer FNO for (48×72) climate grid."""
    def __init__(self, in_ch=5, out_ch=2, ch=64, modes_lat=16, modes_lon=16, layers=4):
        super().__init__()
        self.proj_in  = nn.Conv2d(in_ch, ch, 1)
        self.blocks   = nn.ModuleList([FNOBlock(ch, modes_lat, modes_lon) for _ in range(layers)])
        self.proj_out = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x):                    # x: [B,in_ch,48,72]
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj_out(x)
