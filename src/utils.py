import logging
from typing import Any, Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for level_name in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level_name, rank_zero_only(getattr(logger, level_name)))
    return logger


log = get_logger(__name__)

# --- Data Handling Utilities ---

class Normalizer:
    def __init__(self):
        self.input_methods = {}
        self.output_methods = {}
        self.mean_in = {}
        self.std_in = {}
        self.mean_out = {}
        self.std_out = {}

    def set_input_statistics(self, mean: np.ndarray, std: np.ndarray, method: str = "zscore", var: str = "default"):
        self.input_methods[var] = method
        self.mean_in[var] = mean
        self.std_in[var] = std

    def set_output_statistics(self, mean: np.ndarray, std: np.ndarray, method: str = "zscore", var: str = "default"):
        self.output_methods[var] = method
        self.mean_out[var] = mean
        self.std_out[var] = std

    def normalize(self, data, data_type: str = "input", var: str = "default"):
        method = (self.input_methods if data_type == "input" else self.output_methods).get(var, "zscore")
        mean = (self.mean_in if data_type == "input" else self.mean_out).get(var)
        std = (self.std_in if data_type == "input" else self.std_out).get(var)
        return self._forward(data, method, mean, std)

    def inverse_transform_output(self, data_norm: np.ndarray, var: str = "default") -> np.ndarray:
        method = self.output_methods.get(var, "zscore")
        mean = self.mean_out.get(var)
        std = self.std_out.get(var)
        return self._inverse(data_norm, method, mean, std)

    def _forward(self, data, method, mean, std):
        if method == "zscore":
            return (data - mean) / (std + 1e-6)
        elif method == "minmax":
            return (data - mean) / (std - mean + 1e-6)
        elif method == "none" or mean is None or std is None:
            return data
        else:
            raise NotImplementedError(f"Unsupported normalization method: {method}")

    def _inverse(self, data_norm, method, mean, std):
        if method == "zscore":
            return data_norm * (std + 1e-6) + mean
        elif method == "minmax":
            return data_norm * (std - mean + 1e-6) + mean
        elif method == "none" or mean is None or std is None:
            return data_norm
        else:
            raise NotImplementedError(f"Unsupported inverse method: {method}")


def get_trainer_config(cfg: DictConfig, model=None) -> Dict[str, Any]:
    if cfg.use_wandb:
        if not cfg.wandb_entity or not cfg.wandb_project:
            raise ValueError("wandb_entity and wandb_project required if use_wandb is true.")
        logger = WandbLogger(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.run_name, log_model=False)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        if model is not None:
            logger.watch(model, log="all")
    else:
        logger = None

    trainer_config = OmegaConf.to_container(cfg.trainer)
    trainer_config["logger"] = logger

    if trainer_config.get("accelerator") == "gpu" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.warning("GPU requested but CUDA not available. Using MPS (Apple Silicon) instead.")
            trainer_config["accelerator"] = "mps"
        else:
            log.warning("GPU requested but no GPU available. Falling back to CPU.")
            trainer_config["accelerator"] = "cpu"

    callbacks_list = trainer_config.pop("callbacks", []) or []
    trainer_config["callbacks"] = []
    for callback_config in callbacks_list:
        trainer_config["callbacks"].append(hydra.utils.instantiate(callback_config))

    return trainer_config


# --- Evaluation and Visualization Utilities ---

def create_climate_data_array(data, time_coords, lat_coords, lon_coords, var_name=None, var_unit=None):
    if len(data.shape) == 3:
        dims = ("time", "y", "x")
        coords = {"time": time_coords, "y": lat_coords, "x": lon_coords}
    else:
        dims = ("y", "x")
        coords = {"y": lat_coords, "x": lon_coords}

    attrs = {}
    if var_name:
        attrs["long_name"] = var_name
    if var_unit:
        attrs["units"] = var_unit

    return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)


def calculate_weighted_metric(data_array, weights, dims, metric_type="rmse"):
    weighted_mean = data_array.weighted(weights).mean(dim=dims).values
    return np.sqrt(weighted_mean) if metric_type == "rmse" else weighted_mean


DEFAULT_VIZ_PARAMS = {
    "standard_cmap": "viridis",
    "diff_cmap": "RdBu_r",
    "variance_cmap": "plasma",
    "colorbar_kwargs": {"fraction": 0.046, "pad": 0.04},
    "figure_size": (18, 6),
}


def create_comparison_plots(
    true_data,
    pred_data,
    title_prefix,
    metric_value=None,
    metric_name=None,
    cmap=None,
    diff_cmap=None,
    fig_size=None,
    colorbar_kwargs=None,
):
    cmap = cmap or DEFAULT_VIZ_PARAMS["standard_cmap"]
    diff_cmap = diff_cmap or DEFAULT_VIZ_PARAMS["diff_cmap"]
    fig_size = fig_size or DEFAULT_VIZ_PARAMS["figure_size"]
    colorbar_kwargs = colorbar_kwargs or DEFAULT_VIZ_PARAMS["colorbar_kwargs"]
    fig, axes = plt.subplots(1, 3, figsize=fig_size)

    vmin = min(true_data.min().item(), pred_data.min().item())
    vmax = max(true_data.max().item(), pred_data.max().item())
    plot_params = {"vmin": vmin, "vmax": vmax, "add_colorbar": True, "cbar_kwargs": colorbar_kwargs}

    true_data.plot(ax=axes[0], cmap=cmap, **plot_params)
    axes[0].set_title(f"{title_prefix} (Ground Truth)")

    pred_data.plot(ax=axes[1], cmap=cmap, **plot_params)
    axes[1].set_title(f"{title_prefix} (Prediction)")

    diff = pred_data - true_data
    diff_max = max(abs(diff.min().item()), abs(diff.max().item()))
    diff_plot_params = plot_params.copy()
    diff_plot_params.update({"vmin": -diff_max, "vmax": diff_max, "cmap": diff_cmap})
    diff.plot(ax=axes[2], **diff_plot_params)

    if metric_value is not None and metric_name is not None:
        metric_text = f" ({metric_name}: {metric_value:.4f})"
    else:
        metric_text = ""
    axes[2].set_title(f"Difference{metric_text}")

    plt.tight_layout()
    return fig


def get_lat_weights(latitude_values):
    lat_rad = np.deg2rad(latitude_values)
    weights = np.cos(lat_rad)
    return weights / np.mean(weights)


def convert_predictions_to_kaggle_format(predictions, time_coords, lat_coords, lon_coords, var_names):
    try:
        import pandas as pd
        rows = []
        for t_idx, t in enumerate(time_coords):
            for var_idx, var_name in enumerate(var_names):
                for y_idx, lat in enumerate(lat_coords):
                    for x_idx, lon in enumerate(lon_coords):
                        pred_value = predictions[t_idx, var_idx, y_idx, x_idx]
                        row_id = f"t{t_idx:03d}_{var_name}_{lat:.2f}_{lon:.2f}"
                        rows.append({"ID": row_id, "Prediction": pred_value})
        return pd.DataFrame(rows)
    except Exception as e:
        log.error(f"Failed to convert predictions to Kaggle format: {str(e)}")
        raise
