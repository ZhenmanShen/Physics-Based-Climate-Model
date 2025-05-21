import torch.nn as nn
from .unet import UNet
from omegaconf import DictConfig
from .vit_climate import ViTClimate
from .cnn_transformer import CNNTransformer
from  .cnn_transformer_attention import CNNTransformerAttention
from .unet_transformer import UNetTransformer
from .swin_unet import SwinUNet
from .fno2d import FNO2D
from .unet_attention_transformer import UNetAttentionTransformer
from .cnn_unet_transformer import CNN_UNet_Transformer
from .sfno import SFNO
from .psfno import ProbSFNO
from .fno_transformer import FNOTransformer

def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif cfg.model.type == "unet": # Added unet here
        return UNet(
            in_channels=len(cfg.data.input_vars),
            out_channels=len(cfg.data.output_vars),
            base_channels=cfg.model.base_channels,
        )
    elif cfg.model.type == "fno_transformer":
        return FNOTransformer(
            in_channels=len(cfg.data.input_vars),
            out_channels=len(cfg.data.output_vars),
            modes=cfg.model.modes,
            fno_width=cfg.model.fno_width,
            fno_depth=cfg.model.fno_depth,
            embed_dim=cfg.model.embed_dim,
            trans_depth=cfg.model.trans_depth,
            n_heads=cfg.model.n_heads,
            mlp_dim=cfg.model.mlp_dim,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.type == "vit":
        return ViTClimate(
            in_channels=len(cfg.data.input_vars),
            out_channels=len(cfg.data.output_vars)
        )
    elif cfg.model.type == "cnn_transformer":
        return CNNTransformer(
            in_channels=len(cfg.data.input_vars),
            out_channels=len(cfg.data.output_vars),
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            n_heads=cfg.model.n_heads,
            mlp_dim=cfg.model.mlp_dim,
            dropout=cfg.model.dropout
        )
    elif cfg.model.type == "cnn_transformer_attention":
        return CNNTransformerAttention(
            in_channels=len(cfg.data.input_vars),
            out_channels=len(cfg.data.output_vars),
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            n_heads=cfg.model.n_heads,
            mlp_dim=cfg.model.mlp_dim,
            dropout=cfg.model.dropout
        )
    elif cfg.model.type == "unet_transformer":
        return UNetTransformer(
            in_ch=len(cfg.data.input_vars),
            out_ch=len(cfg.data.output_vars),
            base_ch=cfg.model.base_channels,
            depth=cfg.model.depth,
            vit_dim=cfg.model.vit_dim,
            vit_layers=cfg.model.vit_layers,
            vit_heads=cfg.model.vit_heads,
            patch_size=cfg.model.patch_size,
        )
    elif cfg.model.type == "swin_unet":
        return SwinUNet(
            in_ch=len(cfg.data.input_vars),
            out_ch=len(cfg.data.output_vars),
        )
    elif cfg.model.type == "fno_small":
        return FNO2D(
            in_ch=len(cfg.data.input_vars),
            out_ch=len(cfg.data.output_vars),
            width=cfg.model.width,
            depth=cfg.model.depth,
            modes=(cfg.model.modes, cfg.model.modes)
        )
    elif cfg.model.type == "unet_attention_transformer":
        return UNetAttentionTransformer(
            in_ch=len(cfg.data.input_vars),
            out_ch=len(cfg.data.output_vars),
            base_ch=cfg.model.base_channels,
            depth=cfg.model.depth,
            vit_dim=cfg.model.vit_dim,
            vit_layers=cfg.model.vit_layers,
            vit_heads=cfg.model.vit_heads,
            patch_size=cfg.model.patch_size,
        )
    elif cfg.model.type == "cnn_unet_transformer":
        return CNN_UNet_Transformer(
            in_ch=len(cfg.data.input_vars),
            out_ch=len(cfg.data.output_vars),
            stem_ch=cfg.model.stem_channels,
            base_ch=cfg.model.base_channels,
            depth=cfg.model.depth,
            vit_dim=cfg.model.vit_dim,
            vit_layers=cfg.model.vit_layers,
            vit_heads=cfg.model.vit_heads,
            patch_size=cfg.model.patch_size,
        )
    elif cfg.model.type == "sfno":
        return SFNO(
            nlat=cfg.model.nlat,
            nlon=cfg.model.nlon,
            in_ch=len(cfg.data.input_vars),
            out_ch=len(cfg.data.output_vars),
            embed_dim=cfg.model.embed_dim,
            lmax=cfg.model.lmax,
            layers=cfg.model.layers,
            depth=cfg.model.depth,
            n_heads=cfg.model.n_heads,
            mlp_dim=cfg.model.mlp_dim,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.type == "psfno":
        from src.psfno import ProbSFNO  # if you need the bare model
        return ProbSFNO(
            nlat=cfg.model.nlat,
            nlon=cfg.model.nlon,
            in_ch=len(cfg.data.input_vars),
            out_ch=len(cfg.data.output_vars),
            embed_dim=cfg.model.embed_dim,
            layers=cfg.model.layers,
            lmax=cfg.model.lmax,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
        out = self.relu(out)

        return out


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        depth=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Initial convolution to expand channels
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing feature dimensions
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim

        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:  # Don't double the final layer
                current_dim *= 2

        # Final prediction layers
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.dropout(x)
        x = self.final(x)

        return x
