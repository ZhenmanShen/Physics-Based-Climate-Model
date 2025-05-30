# src/unetpp_pro.py
# -------------------------------------------------------
# UNet++ “Pro”  – deeper nest + SE attention + supervision
# ~9 M params @ base_ch=64 on a 48×72 grid
# -------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Squeeze-and-Excitation ----------
class SEBlock(nn.Module):
    def __init__(self, ch, r: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


# ---------- 2-conv block (+SE) ----------
def conv_block(in_ch, out_ch, k=3, act=nn.ReLU, se=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        act(inplace=True),
        nn.Conv2d(out_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        act(inplace=True),
    ]
    if se:
        layers.append(SEBlock(out_ch))
    return nn.Sequential(*layers)


# ---------- UNet++ Pro ----------
class UNetPlusPlusPro(nn.Module):
    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        base_ch: int = 64,               # 64 => ~9 M params
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        chs = [base_ch * 2 ** i for i in range(4)]  # 64, 128, 256, 512

        # -------- Encoder --------
        self.conv00 = conv_block(n_input_channels, chs[0])
        self.pool0 = nn.MaxPool2d(2)            # 48×72 -> 24×36

        self.conv10 = conv_block(chs[0], chs[1])
        self.pool1 = nn.MaxPool2d(2)            # 24×36 -> 12×18

        self.conv20 = conv_block(chs[1], chs[2])
        self.pool2 = nn.MaxPool2d(2)            # 12×18 -> 6×9

        self.conv30 = conv_block(chs[2], chs[3])

        # -------- Decoder (nested) --------
        self.conv01 = conv_block(chs[0] + chs[1], chs[0])
        self.conv02 = conv_block(chs[0] * 2 + chs[1], chs[0])

        self.conv11 = conv_block(chs[1] + chs[2], chs[1])
        self.conv12 = conv_block(chs[1] * 2 + chs[2], chs[1])

        self.conv21 = conv_block(chs[2] + chs[3], chs[2])

        # -------- Output heads --------
        self.head0 = nn.Conv2d(chs[0], n_output_channels, 1)
        if deep_supervision:
            self.head1 = nn.Conv2d(chs[0], n_output_channels, 1)
            self.head2 = nn.Conv2d(chs[0], n_output_channels, 1)

    # util
    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    # forward
    def forward(self, x):
        # ---- enc ----
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))

        # ---- dec ----
        x01 = self.conv01(torch.cat([x00, self._up(x10, x00)], 1))
        x11 = self.conv11(torch.cat([x10, self._up(x20, x10)], 1))
        x21 = self.conv21(torch.cat([x20, self._up(x30, x20)], 1))

        x02 = self.conv02(torch.cat([x00, x01, self._up(x11, x00)], 1))
        x12 = self.conv12(torch.cat([x10, x11, self._up(x21, x10)], 1))

        if self.deep_supervision:
            out0 = self.head0(x02)
            out1 = self._up(self.head1(x01), out0)
            out2 = self._up(self.head2(x00), out0)
            return (out0 + out1 + out2) / 3.0
        else:
            return self.head0(x02)
