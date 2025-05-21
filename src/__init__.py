from .models import SimpleCNN

def get_model(cfg):
    if cfg.model.type == "simple_cnn":
        return SimpleCNN(
            in_channels=len(cfg.data.input_vars),
            out_channels=len(cfg.data.output_vars),
            kernel_size=cfg.model.kernel_size,
            init_dim=cfg.model.init_dim,
            depth=cfg.model.depth,
            dropout_rate=cfg.model.dropout_rate,
        )
    elif cfg.model.type == "resnet_cnn":
        return ResNetCNN(
            in_channels=len(cfg.data.input_vars),
            out_channels=len(cfg.data.output_vars),
            dim=cfg.model.init_dim,
            depth=cfg.model.depth,
        )
    elif cfg.model.type == "unet":
        return UNet(
            in_channels=len(cfg.data.input_vars),
            out_channels=len(cfg.data.output_vars),
            base_channels=cfg.model.base_channels,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
