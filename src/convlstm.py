# src/models/convlstm.py -------------------------------------------------------
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, c_in, c_hid, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, kernel_size, padding=pad)

    def forward(self, x, h_c):
        h, c = h_c
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    """Temporal depth T is handled outside (loop or be fed a 5-D tensor)."""
    def __init__(self, c_in, c_hid, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(c_in, c_hid, kernel_size)

    def forward(self, x_seq):
        # x_seq: (T, B, C, H, W)
        h = torch.zeros_like(x_seq[0, :, :self.cell.conv.out_channels//4])
        c = torch.zeros_like(h)
        outs = []
        for t in range(x_seq.size(0)):
            h, c = self.cell(x_seq[t], (h, c))
            outs.append(h)
        return torch.stack(outs)          # (T, B, C_hid, H, W)
