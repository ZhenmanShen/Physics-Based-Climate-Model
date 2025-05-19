import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "cnn_rnn":
        model = CNNtoRNN(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model

class CNNtoRNN(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, num_layers=5, hidden_dims=128, base_channels=64):
        super().__init__()

        # CNN to extract features per spatial row (height)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # RNN treats width dimension as time steps, input_size = channels * height
        self.rnn = nn.LSTM(input_size=64 * 48, hidden_size=hidden_dims, num_layers=num_layers, batch_first=True)

        # Final layer projects hidden size back to output channels * height
        self.fc = nn.Linear(hidden_dims, n_output_channels * 48)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        # Pass through CNN
        features = self.cnn(x)  # (32, 64, 48, 72)

        # Permute to (batch, width, channels * height)
        # Each width position is a time step; RNN models sequence along width
        features = features.permute(0, 3, 1, 2)  # (32, 72, 64, 48)
        features = features.reshape(batch_size, W, 64 * H)  # (32, 72, 64*48)

        # Pass through RNN along width dimension
        rnn_out, _ = self.rnn(features)  # (32, 72, hidden_size)

        # Take output at last time step (width=72)
        last_out = rnn_out[:, -1, :]  # (32, hidden_size)

        # Fully connected to project back to (channels * height)
        fc_out = self.fc(last_out)  # (32, 2*48)

        # Reshape to (batch, channels, height)
        fc_out = fc_out.view(batch_size, 2, 48)  # (32, 2, 48)

        # Now expand width dimension to match input width = 72 (e.g., replicate or interpolate)
        # Here just expand by repeating width dimension
        output = fc_out.unsqueeze(-1).repeat(1, 1, 1, 72)  # (32, 2, 48, 72)

        return output

