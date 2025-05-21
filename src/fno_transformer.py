"""
Hybrid Fourier Neural Operator + Transformer
============================================
• No external “fno2d.py” dependency – everything lives here.
• Grid assumed to be (H=48, W=72).  Adjust `h`/`w` in the module if you crop/upsample.
• Torch ≥ 1.13.

Author: you ✨
"""
import math
import torch
import torch.nn as nn
import torch.fft

# --------------------------------------------------------------------------- #
#  Spectral 2-D convolution layer ------------------------------------------- #
# --------------------------------------------------------------------------- #
class SpectralConv2d(nn.Module):
    """
    2-D Fourier spectral conv: multiply low-frequency modes in Fourier space.
    For each forward pass:
        x ⇨ rFFT2  --mul--> inverse FFT2  ⇨ real output
    """
    def __init__(self, in_ch, out_ch, modes_height, modes_width):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes_h, self.modes_w = modes_height, modes_width

        # complex weights for the kept modes
        scale = 1 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, self.modes_h, self.modes_w, 2)
        )

    def compl_mul2d(self, x, w):
        """Complex multiplication (x: B, in, H, W, 2) (w: in, out, H, W, 2)"""
        # (a+ib)(c+id) = (ac-bd) + i(ad+bc)
        r1, i1 = x[..., 0], x[..., 1]
        r2, i2 = w[..., 0], w[..., 1]
        real = r1.unsqueeze(2) * r2 - i1.unsqueeze(2) * i2
        imag = r1.unsqueeze(2) * i2 + i1.unsqueeze(2) * r2
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        """
        x: (B, C, H, W) real tensor
        returns: (B, out_ch, H, W)
        """
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # ➜ (B,C,H,W//2+1) complex

        # manual complex view: (..., 2) last dim = (real, imag)
        x_ft = torch.view_as_real(x_ft)  # (B,C,H,Wc,2)

        out_ft = torch.zeros(
            B, self.out_ch, H, W // 2 + 1, 2, device=x.device, dtype=x.dtype
        )

        # 👉 top-left quadrant (low freqs)
        out_ft[:, :, : self.modes_h, : self.modes_w] = self.compl_mul2d(
            x_ft[:, :, : self.modes_h, : self.modes_w], self.weight
        )

        # inverse FFT back to physical domain
        out_ft = torch.view_as_complex(out_ft)
        y = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return y


# --------------------------------------------------------------------------- #
#  Vanilla 2-D FNO backbone (stack of spectral + point-wise conv) ---------- #
# --------------------------------------------------------------------------- #
class FNO2D(nn.Module):
    def __init__(
        self,
        in_ch=5,
        out_ch=2,
        modes: int = 12,
        width: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.project_in = nn.Conv2d(in_ch, width, 1)

        self.fno_layers = nn.ModuleList()
        for _ in range(depth):
            self.fno_layers.append(
                nn.ModuleDict({                        
                    "spectral": SpectralConv2d(width, width, modes, modes),
                    "pw":       nn.Conv2d(width, width, 1),
                })
            )
        self.norms = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(depth)])
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        # inject spatial grid (helps FNO generalize); optional
        B, C, H, W = x.shape
        grid_y, grid_x = torch.linspace(-1, 1, H, device=x.device), torch.linspace(
            -W / H, W / H, W, device=x.device
        )
        grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=0)
        grid = grid.expand(B, 2, H, W)
        x = torch.cat([x, grid], dim=1)  # (+2 channels)

        x = self.project_in(x)
        for s, bn in zip(self.fno_layers, self.norms):
            x = s["spectral"](x) + s["pw"](x)
            x = self.act(bn(x))
        return self.project_out(x)


# --------------------------------------------------------------------------- #
#  FNO + Transformer HYBRID ------------------------------------------------- #
# --------------------------------------------------------------------------- #
class FNOTransformer(nn.Module):
    """
    Stage-1  : FNO2D per-grid mixing (local+global spectral convolutions)  
    Stage-2  : ViT-style Transformer encoder (global token interactions)  
    Stage-3  : 1×1 conv head -> output vars
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # FNO hyper-params
        modes: int = 12,
        fno_width: int = 64,
        fno_depth: int = 4,
        # Transformer
        embed_dim: int = 128,
        trans_depth: int = 4,
        n_heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        # grid size
        h: int = 48,
        w: int = 72,
    ):
        super().__init__()
        self.h, self.w, self.embed_dim = h, w, embed_dim

        # ---- FNO backbone --------------------------------------------------
        self.fno = FNO2D(
            in_ch=in_channels + 2,          # +2 for injected coords
            out_ch=embed_dim,
            modes=modes,
            width=fno_width,
            depth=fno_depth,
        )

        # ---- Transformer ---------------------------------------------------
        num_tokens = h * w
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, trans_depth)

        # ---- Prediction head ----------------------------------------------
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, 1),
        )

        self._init_weights()

    # ----------------------------------------------------------------------
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----------------------------------------------------------------------
    @staticmethod
    def _flatten_hw(x: torch.Tensor):
        return x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H*W, C)

    def _unflatten_hw(self, x: torch.Tensor):
        return x.transpose(1, 2).view(-1, self.embed_dim, self.h, self.w)

    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        Args
        ----
        x : (B, in_channels, 48, 72)
        Returns
        -------
        (B, out_channels, 48, 72)
        """
        x = self.fno(x)                             # (B, embed_dim, H, W)
        x = self._flatten_hw(x) + self.pos_embed    # add learned pos enc
        x = self.transformer(x)                     # attention
        x = self._unflatten_hw(x)                   # (B, embed_dim, H, W)
        return self.head(x)
