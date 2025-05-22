import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, k=3, act=nn.ReLU):
    """(Conv → BN → Act) × 2"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2),
        nn.BatchNorm2d(out_ch),
        act(inplace=True),
        nn.Conv2d(out_ch, out_ch, k, padding=k // 2),
        nn.BatchNorm2d(out_ch),
        act(inplace=True),
    )


class UNetPlusPlus(nn.Module):
    """
    A very small UNet++ (nested UNet) with 3 encoder depths
    to stay parameter-efficient for limited data.
    """

    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        base_ch: int = 32,           # 32 → ~2.7 M params
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # ---------- Encoder ----------
        self.conv00 = conv_block(n_input_channels, base_ch)
        self.pool0 = nn.MaxPool2d(2)   # 48×72 → 24×36

        self.conv10 = conv_block(base_ch, base_ch * 2)
        self.pool1 = nn.MaxPool2d(2)   # 24×36 → 12×18

        self.conv20 = conv_block(base_ch * 2, base_ch * 4)

        # ---------- Decoder (nested) ----------
        # level 0
        self.conv01 = conv_block(base_ch + base_ch * 2, base_ch)
        self.conv02 = conv_block(base_ch * 4, base_ch)

        # level 1
        self.conv11 = conv_block(base_ch * 2 + base_ch * 4, base_ch * 2)

        # ---------- Output heads ----------
        self.final0 = nn.Conv2d(base_ch, n_output_channels, 1)
        if deep_supervision:
            self.final1 = nn.Conv2d(base_ch, n_output_channels, 1)
            self.final2 = nn.Conv2d(base_ch, n_output_channels, 1)

    # ----- helper -----
    def _upsample(self, x, target):
        # bilinear up to target spatial size
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    # ----- forward -----
    def forward(self, x):
        # Encoder
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))

        # Decoder – level 1
        x01 = self.conv01(torch.cat([x00, self._upsample(x10, x00)], dim=1))
        x11 = self.conv11(torch.cat([x10, self._upsample(x20, x10)], dim=1))

        # Decoder – level 2 (top nest)
        x02 = self.conv02(torch.cat([x00, x01, self._upsample(x11, x00)], dim=1))

        if self.deep_supervision:
            out0 = self.final0(x02)
            out1 = self.final1(x01)
            out2 = self.final2(x00)
            return (out0 + self._upsample(out1, out0) + self._upsample(out2, out0)) / 3.0
        else:
            return self.final0(x02)
