# src/models/att_unet_convlstm.py ---------------------------------------------
from .unet import ConvBlock, Up, SEBlock, SpatialGate
from .convlstm import ConvLSTM
import torch.nn as nn
import torch

class AttUNetConvLSTM(nn.Module):
    def __init__(self, in_ch=5, out_ch=2, base=16, seq_len=3):
        super().__init__()
        self.seq_len = seq_len

        # encoder (shared for each frame)
        self.enc1 = ConvBlock(in_ch,        base)
        self.enc2 = ConvBlock(base,         base * 2)
        self.enc3 = ConvBlock(base * 2,     base * 4)
        self.pool = nn.MaxPool2d(2)

        # temporal bottleneck
        self.convlstm = ConvLSTM(base * 4, base * 4)

        # decoder
        self.up2  = Up(base * 4, base * 2, base * 2)
        self.up1  = Up(base * 2, base,     base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x_seq):
        """
        x_seq : (B, T, C_in, H, W)   with T == self.seq_len
        returns predictions for the *last* frame (B, C_out, H, W)
        """
        B, T, C, H, W = x_seq.shape
        skips2, skips1, feats = [], [], []

        # encode every frame
        for t in range(T):
            x = x_seq[:, t]          # (B,C,H,W)
            s1 = self.enc1(x)        # (B,16,H,W)
            s2 = self.enc2(self.pool(s1))
            f3 = self.enc3(self.pool(s2))
            skips1.append(s1); skips2.append(s2); feats.append(f3)

        # aggregate through ConvLSTM
        f3_seq = torch.stack(feats, dim=0)     # (T,B,C,H',W')
        f3_time = self.convlstm(f3_seq)        # (T,B,C,H',W')
        bott = f3_time[-1]                     # take last hidden state

        # decode only last frame
        d2 = self.up2(bott, skips2[-1])
        d1 = self.up1(d2,   skips1[-1])
        return self.head(d1)
