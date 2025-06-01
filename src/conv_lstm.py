import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, bias=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels[i - 1]
            self.layers.append(
                ConvLSTMCell(in_ch, hidden_channels[i], kernel_size, bias)
            )

    def forward(self, x, hidden=None):
        B, T, C, H, W = x.size()
        device = x.device

        if hidden is None:
            hidden = [
                layer.init_hidden(B, H, W, device) for layer in self.layers
            ]

        current_input = x
        for layer_idx, layer in enumerate(self.layers):
            h, c = hidden[layer_idx]
            output_inner = []
            for t in range(T):
                h, c = layer(current_input[:, t], h, c)
                output_inner.append(h)
            current_input = torch.stack(output_inner, dim=1)

        return current_input  # (B, T, hidden_channels[-1], H, W)


class ClimateConvLSTM(nn.Module):
    def __init__(self, in_channels=5, out_channels=2, hidden_dim=128, num_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(hidden_dim//4, hidden_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # /4
        )
        
        self.convlstm = ConvLSTM(
            input_channels=hidden_dim//2,
            hidden_channels=[16, 32],
            kernel_size=3,
            num_layers=num_layers
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, hidden_dim//2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim//2, out_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):  # x shape: (T, C, H, W)
        T, C, H, W = x.shape
        device = x.device
        x = x.unsqueeze(0)  # Add batch dimension: (1, T, C, H, W)

        # Encode each timestep
        encoded = []
        for t in range(T):
            frame = self.encoder(x[:, t])  # (1, hidden_dim//2, H//4, W//4)
            encoded.append(frame)
        encoded = torch.stack(encoded, dim=1)  # (1, T, hidden_dim//2, H//4, W//4)

        # ConvLSTM
        convlstm_out = self.convlstm(encoded)  # (1, T, 32, H//4, W//4)

        # Decode each timestep
        decoded = []
        for t in range(T):
            out = self.decoder(convlstm_out[:, t])  # (1, C_out, H, W)
            decoded.append(out.squeeze(0))  # remove batch dim → (C_out, H, W)
        return torch.stack(decoded, dim=0)  # (T, C_out, H, W)

