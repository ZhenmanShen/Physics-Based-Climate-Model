import torch, torch.nn as nn, torch.fft
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_c, out_c, modes=(20,20)):
        super().__init__()
        self.modes = modes
        scale = 1/(in_c*out_c)
        self.weight = nn.Parameter(scale * torch.randn(in_c, out_c, *modes, 2))
    def compl_mul2d(self, input, weights):
        # (B,in,H,W) , (in,out,kx,ky,2)
        w = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, w)
    def forward(self, x):                              # x B C H W
        B, C, H, W = x.shape
        # 1) Always do the spectral multiply in FP32
        x_fp32 = x.float()
        x_ft   = torch.fft.rfftn(x_fp32, dim=(2,3))

        # 2) Prepare an FP32-complex output buffer
        out_ft = torch.zeros(
            B, self.weight.shape[1], H, W//2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # 3) Complex multiply: weight is float→complex
        kx, ky = self.modes
        w_complex = torch.view_as_complex(self.weight)  # ComplexFloat
        out_ft[:, :, :kx, :ky] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :kx, :ky],
            w_complex
        )

        # 4) Inverse FFT in FP32
        x_ifft = torch.fft.irfftn(out_ft, s=(H, W), dim=(2,3))

        # 5) Cast back to the input dtype (e.g. fp16) before returning
        return x_ifft.to(x.dtype)

class FNOBlock(nn.Module):
    def __init__(self, width, modes=(20,20)):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.w = nn.Conv2d(width, width, 1)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.spectral(x) + self.w(x)
        return self.act(x)

class FNO2D(nn.Module):
    def __init__(self, in_ch, out_ch, width=64, depth=4, modes=(20,20)):
        super().__init__()
        self.fc0 = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.Sequential(
            *[FNOBlock(width, modes) for _ in range(depth)])
        self.fc1 = nn.Conv2d(width, 128, 1)
        self.fc2 = nn.Conv2d(128, out_ch, 1)
    def forward(self, x):
        x = self.fc0(x)
        x = self.blocks(x)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)
