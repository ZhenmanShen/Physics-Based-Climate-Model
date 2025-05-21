# sfno.py
import torch
import torch.nn as nn
import torch_harmonics as th   # pip install torch-harmonics


# -----------------------------------------------------------------------------#
#  Low-level spectral block
# -----------------------------------------------------------------------------#
class SFNOLayer(nn.Module):
    """
    One spectral block: SHT → learned spectral filter → iSHT → MLP → residual.
    """
    def __init__(self, nlat: int, nlon: int, ch: int, d_model: int, lmax: int):
        """
        Args:
            nlat, nlon   : grid resolution
            ch           : channel width of the block (input = output = ch)
            d_model      : hidden dimension of the MLP
            lmax         : maximum spherical-harmonic degree
        """
        super().__init__()
        self.sht  = th.RealSHT(nlat, nlon, lmax=lmax)
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=lmax)

        # Learned radial spectral filter: [C_in, C_out, L_max+1]
        self.filter = nn.Parameter(torch.zeros(ch, ch, lmax+1, dtype=torch.cfloat))
        with torch.no_grad():
            # make the DC (ℓ=0) pass through unchanged
            eye = torch.eye(ch, dtype=torch.cfloat)
            self.filter[:, :, 0] = eye

        # optional small random perturbation on higher modes
        nn.init.normal_(self.filter[:, :, 1:].real, mean=0.0, std=1e-2)
        nn.init.normal_(self.filter[:, :, 1:].imag, mean=0.0, std=1e-2)

        self.mlp = nn.Sequential(
            nn.Linear(ch, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : tensor of shape [B, ch, nlat, nlon]
        Returns:
            tensor of same shape
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)                              # safer FFTs

        coeffs = self.sht(x)                                 # [B, ch, L, M]
        filt   = self.filter[:, :, : coeffs.shape[2]]        # match L dimension
        coeffs = torch.einsum('bilm,icl->bclm', coeffs, filt)
        x_hat  = self.isht(coeffs).to(orig_dtype)            # back to grid

        # Channel-wise residual MLP (global pooling)
        delta = self.mlp(x_hat.mean(dim=(-1, -2)))           # [B, ch]
        return x + delta[:, :, None, None]


# -----------------------------------------------------------------------------#
#  SFNO spatial encoder (no time)
# -----------------------------------------------------------------------------#
class SFNOEncoder(nn.Module):
    """
    Stack of SFNO spectral blocks with 1×1 in/out projections.
    """
    def __init__(
        self,
        nlat: int = 48,
        nlon: int = 72,
        in_ch: int = 5,
        embed_dim: int = 256,
        lmax: int = 23,
        layers: int = 8,
    ):
        super().__init__()
        self.proj_in  = nn.Conv2d(in_ch, embed_dim, kernel_size=1)
        self.blocks   = nn.ModuleList(
            [SFNOLayer(nlat, nlon, embed_dim, embed_dim * 2, lmax) for _ in range(layers)]
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)          # [B, D, H, W]
        for blk in self.blocks:
            x = blk(x)
        return x                     # [B, D, H, W]


# -----------------------------------------------------------------------------#
#  Full model = SFNO spatial encoder  +  Transformer over grid tokens
# -----------------------------------------------------------------------------#
class SFNO(nn.Module):
    """
    SFNO-Transformer (spatial only).

    Input : [B, C_in, 48, 72]
    Output: [B, C_out, 48, 72]
    """
    def __init__(
        self,
        nlat: int = 48,
        nlon: int = 72,
        in_ch: int = 5,
        out_ch: int = 2,
        embed_dim: int = 256,
        lmax: int = 23,
        layers: int = 8,
        depth: int = 4,
        n_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = SFNOEncoder(
            nlat=nlat,
            nlon=nlon,
            in_ch=in_ch,
            embed_dim=embed_dim,
            lmax=lmax,
            layers=layers,
        )

        # (H, W) → flattened token sequence
        self.H = nlat
        self.W = nlon
        self.num_tokens = self.H * self.W

        # Positional embedding for each grid cell
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # Transformer encoder operating on grid tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, out_ch, kernel_size=1),
        )

    # ---------------------------------------------------------------------#
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C_in, 48, 72]
        Returns:
            y : [B, C_out, 48, 72]
        """
        B = x.size(0)

        # 1) SFNO spatial encoding
        x = self.encoder(x)                              # [B, D, H, W]

        # 2) Flatten spatial grid → tokens
        x = x.flatten(2).transpose(1, 2)                 # [B, H*W, D]
        x = x + self.pos_embed                           # add positional info
        x = self.transformer(x)                          # [B, H*W, D]

        # 3) Reshape back to (D, H, W)
        x = x.transpose(1, 2).view(B, -1, self.H, self.W)

        # 4) Final conv decoder
        return self.decoder(x)                           # [B, C_out, H, W]
