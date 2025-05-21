import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch_harmonics as th


# ------------------------------------------------------------
#  Spectral block (fp32 SHT to avoid half-precision FFT crash)
# ------------------------------------------------------------
class SFNOLayer(nn.Module):
    def __init__(self, nlat, nlon, ch, d_model, lmax, dropout=0.1):
        super().__init__()
        self.sht  = th.RealSHT(nlat, nlon, lmax=lmax)
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=lmax)

        # learned spectral weights: [C_in, C_out, L_max+1]
        self.filter = nn.Parameter(
            torch.randn(ch, d_model, lmax + 1, dtype=torch.cfloat)
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, ch),
        )

    def forward(self, x):                     # x: [B,C,H,W]
        dtype = x.dtype                       # keep original dtype
        x_fp32 = x.to(torch.float32)          # safe for cuFFT

        coeffs = self.sht(x_fp32)             # [B,C,L,M]
        filt   = self.filter[:, :, : coeffs.shape[2]]
        coeffs = torch.einsum('bilm,iol->bolm', coeffs, filt)

        x_hat  = self.isht(coeffs).to(dtype)  # back to grid
        delta  = self.mlp(x_hat.mean([-1, -2]))
        return x + delta[:, :, None, None]


# ------------------------------------------------------------
#  Probabilistic SFNO (µ, log σ)  —> draws samples on demand
# ------------------------------------------------------------
class ProbSFNO(nn.Module):
    def __init__(self,
                 nlat=48, nlon=72,
                 in_ch=5, out_ch=2,
                 embed_dim=256, layers=8, lmax=23, dropout=0.1):
        super().__init__()
        self.proj_in  = nn.Conv2d(in_ch, embed_dim, 1)
        self.blocks   = nn.ModuleList([
            SFNOLayer(nlat, nlon, embed_dim, embed_dim, lmax, dropout)
            for _ in range(layers)
        ])
        self.proj_out = nn.Conv2d(embed_dim, 2 * out_ch, 1)   # µ & log σ
        self.out_ch   = out_ch

    # sample=True during training / MC inference
    def forward(self, x, sample: bool = True):
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        mu, log_sigma = torch.chunk(self.proj_out(x), 2, dim=1)

        if not sample:
            return mu, log_sigma.exp()         # inference: return params

        eps = torch.randn_like(mu)
        return mu + eps * log_sigma.exp(), mu, log_sigma


# ------------------------------------------------------------
# Lightning wrapper
# ------------------------------------------------------------
class LitProbSFNO(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = ProbSFNO(
            nlat=cfg.model.nlat,
            nlon=cfg.model.nlon,
            in_ch=len(cfg.data.input_vars),
            out_ch=len(cfg.data.output_vars),
            embed_dim=cfg.model.embed_dim,
            layers=cfg.model.layers,
            lmax=cfg.model.lmax,
            dropout=0.1,
        )

        # latitude weights for area-weighted loss
        lat = torch.linspace(90, -90, cfg.model.nlat)
        self.register_buffer('w_lat', torch.cos(torch.deg2rad(lat)).view(1, 1, -1, 1))

    # ---------- training ----------
    def training_step(self, batch, _):
        x, y = batch                           # y: [B,C,H,W]
        y_samp, mu, log_sigma = self.model(x)  # one MC sample

        inv_var = torch.exp(-2 * log_sigma)
        nll = ((y - mu) ** 2 * inv_var + 2 * log_sigma)        # per-pixel NLL
        loss = (nll * self.w_lat).mean()                       # area weight

        self.log("train/nll", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ---------- validation ----------
    def validation_step(self, batch, _):
        x, y = batch
        mu, sigma = self.model(x, sample=False)
        # average 16 MC samples
        samples = [mu + sigma * torch.randn_like(mu) for _ in range(16)]
        y_hat = torch.stack(samples).mean(0)
        rmse = F.mse_loss(y_hat, y).sqrt()
        self.log("val/rmse", rmse, prog_bar=True)

    # ---------- optimiser ----------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.optim.lr,
                                weight_decay=self.hparams.optim.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.trainer.max_epochs
        )
        return [opt], [sched]
