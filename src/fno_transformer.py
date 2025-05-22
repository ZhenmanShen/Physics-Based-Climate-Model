# src/models/fno_transformer.py
import math, torch
import torch.nn as nn, torch.fft as fft


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 modes_y=12, modes_x=18):
        super().__init__()
        scale = 1 / math.sqrt(in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels,
                                modes_y, modes_x, 2))
        self.modes_y, self.modes_x = modes_y, modes_x

    def _cplx_mul(self, a, b):
        b = torch.view_as_complex(b)
        return torch.einsum("bihw,iohw->bohw", a, b)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.weight.size(1),
                             H, W//2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes_y, :self.modes_x] = \
            self._cplx_mul(x_ft[..., :self.modes_y, :self.modes_x],
                           self.weight)
        return fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, width, modes_y, modes_x):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_y, modes_x)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.bn = nn.BatchNorm2d(width)

    def forward(self, x):
        return self.bn(torch.relu(self.spectral(x) + self.pointwise(x)))


class FNOTransformer(nn.Module):
    def __init__(self,
                 in_channels=5, out_channels=2,
                 width=48, depth_fno=4,
                 modes_y=12, modes_x=18,
                 trans_layers=2, n_heads=4,
                 trans_mlp=192, dropout=0.1,
                 pool=2):                       # NEW: downsample factor
        super().__init__()
        H, W = 48, 72
        self.pool = pool
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.fno = nn.Sequential(
            *[FNOBlock(width, modes_y, modes_x)
              for _ in range(depth_fno)])

        # Optional pooling to save VRAM
        h_t, w_t = H // pool, W // pool
        self.token_dim = width
        self.pos = nn.Parameter(torch.randn(1, h_t*w_t, width))

        enc = nn.TransformerEncoderLayer(d_model=width, nhead=n_heads,
                                         dim_feedforward=trans_mlp,
                                         dropout=dropout, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=trans_layers)
        self.head = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):                       # (B,5,48,72)
        x = self.lift(x)
        x = self.fno(x)                         # (B,width,48,72)
        if self.pool > 1:
            x = torch.nn.functional.avg_pool2d(x, self.pool)   # (B,width,24,36)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B,T,C)  T=H*W
        x = self.trans(x + self.pos)
        x = x.transpose(1, 2).view(B, C, H, W)
        if self.pool > 1:
            x = torch.nn.functional.interpolate(
                    x, scale_factor=self.pool, mode="bilinear", align_corners=False)
        return self.head(x)                     # (B,2,48,72)
