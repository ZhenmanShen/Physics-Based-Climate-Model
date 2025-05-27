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


# ---------- ConvLSTM Cell ----------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ---------- Temporal UNet++ Pro ----------
class TemporalUNetPlusPlus(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, base_ch=64, deep_supervision=True, time_steps=12):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.time_steps = time_steps
        chs = [base_ch * 2 ** i for i in range(4)]

        # Encoder (shared across time steps)
        self.conv00 = conv_block(n_input_channels, chs[0])
        self.pool0 = nn.MaxPool2d(2)
        self.conv10 = conv_block(chs[0], chs[1])
        self.pool1 = nn.MaxPool2d(2)
        self.conv20 = conv_block(chs[1], chs[2])
        self.pool2 = nn.MaxPool2d(2)
        self.conv30 = conv_block(chs[2], chs[3])

        # ConvLSTM layer on top encoder output (x30)
        self.convlstm = ConvLSTMCell(chs[3], chs[3], kernel_size=3)

        # Decoder
        self.conv21 = conv_block(chs[2] + chs[3], chs[2])
        self.conv11 = conv_block(chs[1] + chs[2], chs[1])
        self.conv01 = conv_block(chs[0] + chs[1], chs[0])
        self.conv02 = conv_block(chs[0] * 2 + chs[1], chs[0])
        self.conv12 = conv_block(chs[1] * 2 + chs[2], chs[1])

        # Output heads
        self.head0 = nn.Conv2d(chs[0], n_output_channels, 1)
        if deep_supervision:
            self.head1 = nn.Conv2d(chs[0], n_output_channels, 1)
            self.head2 = nn.Conv2d(chs[0], n_output_channels, 1)

    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # Encoder through time
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))

        x30 = x30.view(B, T, -1, x30.shape[-2], x30.shape[-1])

        # ConvLSTM on encoded features
        h = torch.zeros_like(x30[:, 0])
        c = torch.zeros_like(x30[:, 0])
        for t in range(T):
            h, c = self.convlstm(x30[:, t], h, c)

        # Decoder
        x21 = self.conv21(torch.cat([x20[:B], self._up(h, x20[:B])], dim=1))
        x11 = self.conv11(torch.cat([x10[:B], self._up(x21, x10[:B])], dim=1))
        x01 = self.conv01(torch.cat([x00[:B], self._up(x11, x00[:B])], dim=1))
        x02 = self.conv02(torch.cat([x00[:B], x01, self._up(x11, x00[:B])], dim=1))

        if self.deep_supervision:
            out0 = self.head0(x02)
            out1 = self._up(self.head1(x01), out0)
            out2 = self._up(self.head2(x00[:B]), out0)
            return (out0 + out1 + out2) / 3.0
        else:
            return self.head0(x02)