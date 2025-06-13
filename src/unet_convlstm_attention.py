# You would place this in a new file or update your att_unet_convlstm.py
# Ensure ConvBlock, Up, etc. are correctly imported from your unet.py
# from .unet import ConvBlock, Up # Assuming unet.py has these
# from .convlstm import ConvLSTM

import torch
import torch.nn as nn
# Make sure these imports point to your actual unet.py and convlstm.py
# For example, if they are in the same directory 'models':
from .unet import ConvBlock, Up 
from .convlstm import ConvLSTM

# If unet.py is like the one you provided:
# It has SEBlock and SpatialGate inside ConvBlock.
# It has Down (MaxPool + ConvBlock) and Up (ConvTranspose + ConvBlock)

# Let's redefine Down here if it's not directly usable or to ensure clarity
class DownPoolEnc(nn.Module): # Similar to Down in your UNet
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(c_in, c_out) # Your ConvBlock with SE and SpatialGate

    def forward(self, x):
        return self.conv(self.pool(x))

class AttUNetConvLSTM(nn.Module):
    def __init__(self, in_ch: int = 5, out_ch: int = 2, base: int = 16, seq_len: int = 3):
        super().__init__()
        self.seq_len = seq_len

        # Per-frame Encoder (deeper: 4 stages)
        self.enc1 = ConvBlock(in_ch+6, base)          # H, W, base
        self.enc2 = DownPoolEnc(base, base * 2)     # H/2, W/2, base*2
        self.enc3 = DownPoolEnc(base * 2, base * 4) # H/4, W/4, base*4
        self.enc4 = DownPoolEnc(base * 4, base * 8) # H/8, W/8, base*8 (features for ConvLSTM)

        # Temporal Bottleneck with ConvLSTM
        # Input to ConvLSTM: base*8 channels
        # Keep ConvLSTM hidden channels manageable, e.g., base*4 or base*8
        # Let's try c_hid = base*4 to keep params lower initially.
        # If base=16, c_in=128, c_hid=64. Conv2d(128+64, 4*64) = Conv2d(192, 256) -> ~442k params
        # If c_hid=base*8 (e.g., 128 for base=16), Conv2d(128+128, 4*128)=Conv2d(256,512) -> ~1.18M params (too much for LSTM part alone)
        # So, c_hid = base*4 seems like a good compromise.
        self.convlstm = ConvLSTM(c_in=base * 8, c_hid=base * 4, kernel_size=3)
        self.post_conv = nn.Sequential(
            nn.Conv2d(base*4, base*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        # ConvLSTM output has base*4 channels. Skips from enc3, enc2, enc1.
        self.up3 = Up(c_in=base * 4, c_skip=base * 4, c_out=base * 4) # Input from LSTM, Skip from enc3
        self.up2 = Up(c_in=base * 4, c_skip=base * 2, c_out=base * 2) # Skip from enc2
        self.up1 = Up(c_in=base * 2, c_skip=base, c_out=base)         # Skip from enc1
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)



    def forward(self, x_seq):
        """
        x_seq : (B, T, C_in, H, W)  with T == self.seq_len
        returns predictions for the *last* frame (B, C_out, H, W)
        """
        B, T, C, H, W = x_seq.shape
        
        all_s1, all_s2, all_s3 = [], [], []
        encoded_features_for_lstm = []

        # Encode every frame
        for t in range(T):
            x_t = x_seq[:, t]               # (B, C_in, H, W)
            
            s1_t = self.enc1(x_t)           # (B, base, H, W)
            s2_t = self.enc2(s1_t)          # (B, base*2, H/2, W/2)
            s3_t = self.enc3(s2_t)          # (B, base*4, H/4, W/4)
            s4_t = self.enc4(s3_t)          # (B, base*8, H/8, W/8) -> Input to ConvLSTM

            encoded_features_for_lstm.append(s4_t)
            all_s1.append(s1_t)
            all_s2.append(s2_t)
            all_s3.append(s3_t)
        
        # Aggregate features for ConvLSTM
        lstm_input_seq = torch.stack(encoded_features_for_lstm, dim=0) # (T, B, base*8, H/8, W/8)
        
        lstm_out_seq = self.convlstm(lstm_input_seq) # (T, B, base*4, H/8, W/8) (since c_hid = base*4)
        bottleneck_features = lstm_out_seq[-1]       # Use last hidden state (B, base*4, H/8, W/8)

        # Aggregate skip connections (e.g., by averaging over time)
        s1_skip = torch.stack(all_s1, dim=0).mean(dim=0)
        s2_skip = torch.stack(all_s2, dim=0).mean(dim=0)
        s3_skip = torch.stack(all_s3, dim=0).mean(dim=0)
        # Alternatively, use skips from the last frame:
        # s1_skip = all_s1[-1]
        # s2_skip = all_s2[-1]
        # s3_skip = all_s3[-1]

        # Decoder
        d3 = self.up3(bottleneck_features, s3_skip) # Skip from enc3 features
        d2 = self.up2(d3, s2_skip)                  # Skip from enc2 features
        d1 = self.up1(d2, s1_skip)                  # Skip from enc1 features
        
        return self.head(d1)