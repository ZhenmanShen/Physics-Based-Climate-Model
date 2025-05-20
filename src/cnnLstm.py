import torch.nn as nn
import torch

class CNN_LSTM(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, hidden_dim=128, lstm_layers=1):
        super().__init__()

        # CNN layer (예시)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Flatten spatial dims and feed into LSTM
        self.lstm = nn.LSTM(
            input_size=32 * 48 * 72,  
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

      
        self.fc = nn.Linear(hidden_dim, n_output_channels * 48 * 72)

        self.out_channels = n_output_channels

    def forward(self, x):
        B, C, H, W = x.shape  # (B, 5, 48, 72)
        x = self.cnn(x)       # (B, 32, H, W)
        x = x.view(B, 1, -1)  # (B, T=1, Features)
        x, _ = self.lstm(x)   # (B, 1, hidden_dim)
        x = self.fc(x[:, -1, :])  # (B, out_dim)
        x = x.view(B, self.out_channels, H, W)  # (B, 2, 48, 72)
        return x
