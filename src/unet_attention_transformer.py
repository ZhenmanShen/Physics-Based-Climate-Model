"""
UNet-Attention-Transformer
--------------------------

• Encoder UNet blocks + SE channel re-weighting  
• ViT bottleneck + optional SE  
• Decoder skip-connections gated by AttentionGate  
• Output size kept at (B, out_ch, 48, 72)

Author: <you>
Date  : 2025-05-19
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------- #
def conv_bn_relu(in_ch: int, out_ch: int, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU()
    )


# Channel & spatial attention building blocks
class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, in_ch: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch,  in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class AttentionGate(nn.Module):
    """Spatial gate for a skip connection (as in Oktay et al., 2018)."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: decoder feature (gate) – x: encoder skip
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Up-block with optional AttentionGate + SE after fusion
class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 use_attn: bool = True, use_se: bool = False):
        super().__init__()
        self.use_attn = use_attn
        if use_attn:
            self.attn = AttentionGate(F_g=in_ch,
                                      F_l=skip_ch,
                                      F_int=max(in_ch, skip_ch) // 2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False)
        self.conv = conv_bn_relu(in_ch + skip_ch, out_ch)
        if use_se:
            self.se = SEBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad in case of odd spatial dims
        if x.shape[-2:] != skip.shape[-2:]:
            diffY = skip.size(-2) - x.size(-2)
            diffX = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        if self.use_attn:
            skip = self.attn(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        if hasattr(self, "se"):
            x = self.se(x)
        return x


# ViT bottleneck (unchanged from baseline)
class PatchEmbed(nn.Module):
    """Flatten (H,W) → N patches and embed."""
    def __init__(self, in_ch, emb_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, emb_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                         # B,C,H',W'
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)         # B, N, C
        return x, (H, W)


class ViT_Bottleneck(nn.Module):
    def __init__(self, in_ch, emb_dim=512, num_layers=4,
                 num_heads=8, patch_size=4, mlp_ratio=4.):
        super().__init__()
        self.embed = PatchEmbed(in_ch, emb_dim, patch_size)
        num_patches = (48 // patch_size) * (72 // patch_size)   # fixed grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,
                                    num_patches + 1, emb_dim))
        self.pos_drop  = nn.Dropout(0.1)

        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            batch_first=True,
            dropout=0.1,
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer,
                                                 num_layers=num_layers)
        self.deproj = nn.Linear(emb_dim, in_ch)
        self.patch_size = patch_size
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):                         # B,C,48,72
        B = x.size(0)
        tokens, (H, W) = self.embed(x)            # B,N,C
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        tokens = self.pos_drop(tokens)
        tokens = self.transformer(tokens)
        tokens = tokens[:, 1:, :]                 # drop CLS
        tokens = self.deproj(tokens)              # B,N,in_ch
        x = tokens.transpose(1, 2).reshape(B, -1, H, W)
        # reverse patchify
        x = F.interpolate(x, scale_factor=self.patch_size,
                          mode="bilinear", align_corners=False)
        return x                                  # B,C,48,72


# Full UNet-Attention-Transformer
class UNetAttentionTransformer(nn.Module):
    """
    Encoder-Decoder UNet with ViT bottleneck.
    Added SE in encoder & ViT; AttentionGate+SE in decoder.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 base_ch: int = 64, depth: int = 4,
                 vit_dim: int = 512, vit_layers: int = 4,
                 vit_heads: int = 8, patch_size: int = 4,
                 se_reduction: int = 8):
        super().__init__()

        # 1. Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for d in range(depth):
            out = base_ch * 2 ** d
            block = conv_bn_relu(ch, out)
            block = nn.Sequential(block, SEBlock(out, se_reduction))
            self.downs.append(block)
            self.pools.append(nn.MaxPool2d(2))
            ch = out

        # 2. Bottleneck
        self.vit = ViT_Bottleneck(ch,
                                  emb_dim=vit_dim,
                                  num_layers=vit_layers,
                                  num_heads=vit_heads,
                                  patch_size=patch_size)
        self.se_vit = SEBlock(ch, se_reduction)

        # 3. Decoder
        self.ups = nn.ModuleList()
        for d in reversed(range(depth)):
            skip_ch = base_ch * 2 ** d
            self.ups.append(
                UpBlock(ch, skip_ch, skip_ch,
                        use_attn=True, use_se=True)
            )
            ch = skip_ch

        # 4. Prediction head
        self.head = nn.Conv2d(ch, out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        # encoder
        for enc, pool in zip(self.downs, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # bottleneck
        x = self.vit(x)
        x = self.se_vit(x)

        # decoder
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.head(x)


def get_model(cfg):
    """
    Hydra-style factory.
    cfg.model.type should be "unetAttentionTransformer".
    """
    model_kwargs = {k: v for k, v in cfg.model.items()
                    if k not in ("type",)}
    model_kwargs["in_ch"]  = len(cfg.data.input_vars)
    model_kwargs["out_ch"] = len(cfg.data.output_vars)

    if cfg.model.type == "unetAttentionTransformer":
        return UNetAttentionTransformer(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")


# sanity test
if __name__ == "__main__":
    dummy = torch.randn(2, 5, 48, 72)   # 5 input channels
    net   = UNetAttentionTransformer(in_ch=5, out_ch=2)
    out   = net(dummy)
    print("output shape:", out.shape)   # should be (2, 2, 48, 72)
