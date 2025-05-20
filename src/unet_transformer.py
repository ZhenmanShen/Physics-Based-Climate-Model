import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------  Small helpers ----------
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU()
    )

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv_bn_relu(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed (odd spatial dims)
        if x.shape[-2:] != skip.shape[-2:]:
            diffY = skip.size(-2) - x.size(-2)
            diffX = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ----------  ViT bottleneck ----------
class PatchEmbed(nn.Module):
    """Flatten (H, W) into N patches and embed"""
    def __init__(self, in_ch, emb_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, emb_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                                     # B,C,H',W'
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                     # B, N, C
        return x, (H, W)


class ViT_Bottleneck(nn.Module):
    def __init__(self, in_ch, emb_dim=512, num_layers=4,
                 num_heads=8, patch_size=4, mlp_ratio=4.):
        super().__init__()
        self.embed = PatchEmbed(in_ch, emb_dim, patch_size)
        num_patches = (48 // patch_size) * (72 // patch_size)   # fixed spatial size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, emb_dim))
        self.pos_drop = nn.Dropout(0.1)

        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.deproj = nn.Linear(emb_dim, in_ch)

        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):                                     # B,C,48,72
        B = x.size(0)
        tokens, spatial_hw = self.embed(x)                    # B,N,C
        cls_token = self.cls_token.expand(B, -1, -1)          # B,1,C
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        tokens = self.pos_drop(tokens)

        tokens = self.transformer(tokens)                     # B,N+1,C
        tokens = tokens[:, 1:, :]                             # drop CLS

        tokens = self.deproj(tokens)                          # B,N,in_ch
        H, W = spatial_hw
        x = tokens.transpose(1, 2).reshape(B, -1, H, W)       # B,C,H',W'
        # reverse patchify
        x = F.interpolate(x, scale_factor=self.patch_size,
                          mode="bilinear", align_corners=False)
        return x                                              # B,C,48,72

# ----------  Full UNet-Transformer ----------
class UNetTransformer(nn.Module):
    """
    Encoder-Decoder UNet with a ViT bottleneck.
    Input : (B, in_ch, 48, 72)
    Output: (B, out_ch, 48, 72)
    """

    def __init__(self, in_ch, out_ch,
                 base_ch=64, depth=4,
                 vit_dim=512, vit_layers=4,
                 vit_heads=8, patch_size=4):
        super().__init__()

        # 1. Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for d in range(depth):
            out = base_ch * 2**d
            self.downs.append(conv_bn_relu(ch, out))
            self.pools.append(nn.MaxPool2d(2))
            ch = out

        # 2. Bottleneck ViT (global context)
        self.vit = ViT_Bottleneck(ch,
                                  emb_dim=vit_dim,
                                  num_layers=vit_layers,
                                  num_heads=vit_heads,
                                  patch_size=patch_size)

        # 3. Decoder
        self.ups = nn.ModuleList()
        for d in reversed(range(depth)):
            skip_ch = base_ch * 2**d
            self.ups.append(
                UpBlock(ch, skip_ch, skip_ch))
            ch = skip_ch

        # 4. Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        skips = []
        # ------ encoder -------
        for enc, pool in zip(self.downs, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # ------ bottleneck ----
        x = self.vit(x)

        # ------ decoder -------
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.head(x)
