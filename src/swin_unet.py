import torch, torch.nn as nn
from einops import rearrange, repeat

# ---------- Swin building blocks ----------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.drop(x)

class WindowAttention(nn.Module):
    """W-MHA with relative bias, no qkv bias for simplicity."""
    def __init__(self, dim, heads=4, window_size=6, qkv_bias=False, drop=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.ws = window_size
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

        # relative-position bias table
        self.register_parameter(
            "rel_bias", nn.Parameter(torch.zeros(
                (2*window_size-1) * (2*window_size-1), heads)))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flat = coords.flatten(1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords += window_size - 1
        rel_coords = rel_coords[0]* (2*window_size-1) + rel_coords[1]
        self.register_buffer("rel_index", rel_coords)

    def forward(self, x):                               # (B*nW, ws*ws, dim)
        B_, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B_, N, self.heads, C//self.heads).transpose(1,2) * self.scale
        k = k.view(B_, N, self.heads, C//self.heads).transpose(1,2)
        v = v.view(B_, N, self.heads, C//self.heads).transpose(1,2)

        attn = q @ k.transpose(-2,-1)
        attn += self.rel_bias[self.rel_index.view(-1)].view(
                    self.ws*self.ws, self.ws*self.ws, -1).permute(2,0,1)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        return self.proj(self.drop(x))

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window=6, shift=False, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window)
        self.shift = shift
        self.window = window
        self.norm2 = nn.LayerNorm(dim)
        self.mlp  = MLP(dim, mlp_ratio)
    def forward(self, x):                               # (B, H, W, C)
        H, W = x.shape[1:3]
        if self.shift:
            x = torch.roll(x, shifts=(-self.window//2, -self.window//2), dims=(1,2))
        # partition
        x_windows = rearrange(x, 'b (h w1) (w w2) c -> (b h w) (w1 w2) c',
                              w1=self.window, w2=self.window)
        x_windows = self.attn(self.norm1(x_windows))
        # merge
        x = rearrange(x_windows, '(b h w) (w1 w2) c -> b (h w1) (w w2) c',
                      h=H//self.window, w=W//self.window,
                      w1=self.window, w2=self.window, c=x.shape[-1])
        if self.shift:
            x = torch.roll(x, shifts=(self.window//2, self.window//2), dims=(1,2))
        x = x + self.mlp(self.norm2(x))
        return x
# ---------- Encoder-decoder skeleton ----------
def conv3x3(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU())

class PatchEmbed(nn.Module):
    """4×4 patch embed with conv stride=4."""
    def __init__(self, in_ch, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, 4, 4)
    def forward(self, x):
        return self.proj(x)                             # B, C, 12, 18

class SwinEncoder(nn.Module):
    def __init__(self, in_ch, dims=(64, 128, 256)):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, dims[0])
        self.stage1 = nn.Sequential(
            SwinBlock(dims[0], heads=2, window=6, shift=False),
            SwinBlock(dims[0], heads=2, window=6, shift=True))
        self.down1  = nn.Conv2d(dims[0], dims[1], 2, 2)           # 6×9
        self.stage2 = nn.Sequential(
            SwinBlock(dims[1], heads=4, window=3, shift=False),
            SwinBlock(dims[1], heads=4, window=3, shift=True))
        self.down2  = nn.Conv2d(dims[1], dims[2], 3, 3)           # 2×3
        self.stage3 = nn.Sequential(
            SwinBlock(dims[2], heads=8, window=2, shift=False),
            SwinBlock(dims[2], heads=8, window=2, shift=True))
    @staticmethod
    def _stage(x, stage):
        """helper: BCHW → BHWC → SwinBlocks → BCHW"""
        x = x.permute(0, 2, 3, 1)     # BCHW → BHWC
        x = stage(x)                  # Swin blocks
        return x.permute(0, 3, 1, 2)  # back to BCHW

    def forward(self, x):
        # 1) patch embed → [B, 64, 12, 18]
        x = self.patch_embed(x)
        # 2) stage1 on 12×18
        s1 = self._stage(x, self.stage1)

        # 3) down to 6×9, then stage2
        x = self.down1(s1)
        s2 = self._stage(x, self.stage2)

        # 4) down to 2×3, pad so both dims % window == 0, then stage3
        x = self.down2(s2)  # [B, 256, 2, 3]

        # — pad to multiples of the window (window=2 for stage3) —
        import torch.nn.functional as F
        ws = self.stage3[0].window        # should be 2
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            # F.pad takes (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h))

        s3 = self._stage(x, self.stage3)

        return s1, s2, s3
    
class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = conv3x3(in_c+skip_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear')
        x = torch.cat([x, skip], 1)
        return self.conv(x)

class SwinUNet(nn.Module):
    def __init__(self, in_ch, out_ch, dims=(64,128,256)):
        super().__init__()
        self.encoder = SwinEncoder(in_ch, dims)
        self.mid_conv = conv3x3(dims[2], dims[2])
        self.up1 = Up(dims[2], dims[1], dims[1])
        self.up2 = Up(dims[1], dims[0], dims[0])
        self.out_conv = nn.Conv2d(dims[0], out_ch, 1)
    def forward(self, x):
        s1,s2,s3 = self.encoder(x)
        x = self.mid_conv(s3)
        x = self.up1(x, s2)
        x = self.up2(x, s1)
        x = nn.functional.interpolate(x, size=(48,72), mode='bilinear')
        return self.out_conv(x)
