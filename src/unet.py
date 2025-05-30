import torch
import torch.nn as nn

# ---------- tiny attention ---------------------------------------------------
class CBAM(nn.Module):                         # ~2 lines / channel
    def __init__(self, c, r=8):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1, bias=False), nn.GELU(),
            nn.Conv2d(c // r, c, 1, bias=False), nn.Sigmoid()
        )
        self.sa = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        x = x * self.ca(x)                                           # channel
        avg, mx = x.mean(1, keepdim=True), x.amax(1, keepdim=True)   # spatial
        x = x * torch.sigmoid(self.sa(torch.cat([avg, mx], 1)))
        return x

def CBR(i, o, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(i, o, k, s, k // 2, bias=False),
        nn.GroupNorm(4, o),
        nn.GELU()
    )

# ---------- encoder / decoder -----------------------------------------------
class Down(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(CBR(i, o), CBR(o, o), CBAM(o))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.block(x)                # skip
        return self.pool(y), y

class Up(nn.Module):
    """
    Correct order:   up-conv (inC ➜ outC)  THEN  concat skip  THEN  convs
    """
    def __init__(self, inC, outC):
        super().__init__()
        self.up = nn.ConvTranspose2d(inC, outC, 2, stride=2)
        self.conv = nn.Sequential(
            CBR(outC * 2, outC),         # after concat
            CBR(outC, outC)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:          # pad for odd dims
            diffY = skip.size(-2) - x.size(-2)
            diffX = skip.size(-1) - x.size(-1)
            x = nn.functional.pad(x, (0, diffX, 0, diffY))
        x = torch.cat([skip, x], 1)
        return self.conv(x)

# ---------- optional ConvLSTM bottleneck ------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, inC, hidC, k=3):
        super().__init__()
        self.conv = nn.Conv2d(inC + hidC, 4 * hidC, k, padding=k // 2)
        self.hidC = hidC

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], 1))
        i, f, o, g = torch.chunk(gates, 4, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, b, h, w, device):
        z = torch.zeros(b, self.hidC, h, w, device=device)
        return z, z

# ---------- full tiny-UNet ---------------------------------------------------
class UNet(nn.Module):
    """
    * With seq_len == 1 ➜ plain UNet (~230 k params).
    * With seq_len > 1 ➜ ConvLSTM bottleneck adds ~70 k (total ~300 k).
    """
    def __init__(self,
                 n_input_channels: int,
                 n_output_channels: int,
                 base_channels: int = 8,
                 depth: int = 3,
                 convlstm_hidden: int = 0):          # 0 = disable LSTM
        super().__init__()
        self.use_lstm = convlstm_hidden > 0

        chs = [base_channels * 2 ** i for i in range(depth)]
        self.stem = CBR(n_input_channels, base_channels)

        self.downs = nn.ModuleList(
            [Down(chs[i - 1] if i else base_channels, chs[i])
             for i in range(depth)]
        )

        bott_in = chs[-1]
        bott_out = convlstm_hidden if self.use_lstm else chs[-1]
        self.bottleneck = CBR(bott_in, bott_out)
        if self.use_lstm:
            self.lstm = ConvLSTMCell(bott_out, bott_out)

        # decoder
        rev = list(reversed(chs))
        in_channels = bott_out
        self.ups = nn.ModuleList()
        for skipC in rev:
            self.ups.append(Up(in_channels, skipC))
            in_channels = skipC

        self.head = nn.Conv2d(base_channels, n_output_channels, 1)

    # accepts (B,C,H,W) or (B,T,C,H,W)
    def forward(self, x):
        if x.dim() == 4:          # (B,C,H,W)  -> (B,1,C,H,W)
            x = x.unsqueeze(1)

        B, T, C, H, W = x.shape
        device = x.device
        outs = []

        if self.use_lstm:         # initialise (h, c)
            h_l = H // 2 ** len(self.downs)
            w_l = W // 2 ** len(self.downs)
            state = self.lstm.init_state(B, h_l, w_l, device)

        for t in range(T):
            f = self.stem(x[:, t])
            skips = []
            for down in self.downs:
                f, s = down(f)
                skips.append(s)

            f = self.bottleneck(f)

            if self.use_lstm:
                h_state, c_state = self.lstm(f, state)  # <-- fix
                state = (h_state, c_state)
                f = h_state

            for up in self.ups:
                f = up(f, skips.pop())

            outs.append(self.head(f))

        return outs[-1] if T > 1 else outs[0]   # (B,C_out,H,W)