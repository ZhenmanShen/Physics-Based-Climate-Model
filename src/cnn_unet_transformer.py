# src/models/cnn_unet_transformer.py
import torch, torch.nn as nn
import torch.nn.functional as F
from .unet_transformer import PatchEmbed, ViT_Bottleneck, UpBlock

class CNN_UNet_Transformer(nn.Module):
    """
    1) Initial CNN stem
    2) UNet encoder/decoder
    3) Transformer bottleneck
    """

    def __init__(
        self,
        in_ch, out_ch,
        stem_ch=32,   # extra CNN before UNet
        base_ch=64, depth=4,
        vit_dim=256, vit_layers=4, vit_heads=4, patch_size=1
    ):
        super().__init__()

        # 1) CNN stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch), nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch), nn.GELU(),
        )

        # 2) UNet encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = stem_ch
        for d in range(depth):
            out = base_ch * 2**d
            self.downs.append(nn.Sequential(
                nn.Conv2d(ch, out, 3, padding=1, bias=False),
                nn.BatchNorm2d(out), nn.GELU(),
                nn.Conv2d(out, out, 3, padding=1, bias=False),
                nn.BatchNorm2d(out), nn.GELU(),
            ))
            self.pools.append(nn.MaxPool2d(2))
            ch = out

        # 3) Transformer bottleneck
        self.vit = ViT_Bottleneck(
            in_ch=ch, emb_dim=vit_dim,
            num_layers=vit_layers, num_heads=vit_heads,
            patch_size=patch_size
        )

        # 4) UNet decoder
        self.ups = nn.ModuleList()
        for d in reversed(range(depth)):
            skip_ch = base_ch * 2**d
            self.ups.append(UpBlock(ch, skip_ch, skip_ch))
            ch = skip_ch

        # 5) Final CNN head
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, out_ch, 1)
        )

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Encoder
        skips = []
        for enc, pool in zip(self.downs, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Transformer bottleneck
        x = self.vit(x)

        # Decoder
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        # Head
        return self.head(x)
