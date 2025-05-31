import logging
from typing import Any, Dict
import dask.array as da
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)

# --- Data Handling Utilities ---


class Normalizer:
    def __init__(self, transform_map: Dict[str, Dict[str, Any]]):
        """
        transform_map: e.g.
        {
          "CO2": {"method": "log1p"},
          "pr":  {"method": "pow", "lambda": 1/3},
          "rsdt": {"method": "minmax", "min": 0, "max": 550},
          ...
        }
        """
        self.transform_map = transform_map
        self.stats_map: Dict[str, Dict[str, Any]] = {}

    def _fit_zscore(self, array: da.Array):
        mean = da.nanmean(array, axis=(0, 2, 3), keepdims=True).compute()
        std = da.nanstd(array, axis=(0, 2, 3), keepdims=True).compute()
        return mean, std

    def _fit_minmax(self, array: da.Array):
        min_val = da.nanmin(array, axis=(0, 2, 3), keepdims=True).compute()
        max_val = da.nanmax(array, axis=(0, 2, 3), keepdims=True).compute()
        return min_val, max_val

    def fit(self, data_arrays: Dict[str, da.Array]):
        """
        Fit stats for each variable in data_arrays.
        data_arrays[var_name] must be shape (time,1,y,x) or (time,y,x).
        """
        for var_name, array in data_arrays.items():
            recipe = self.transform_map.get(var_name, {})
            method = recipe.get("method", "zscore")

            if method == "zscore":
                mean, std = self._fit_zscore(array)

                self.stats_map[var_name] = {
                    "method": "zscore",
                    "mean": mean,
                    "std": std,
                }

            elif method == "minmax":
                # if user hard-coded min/max, use that; otherwise compute
                if "min" in recipe and "max" in recipe:
                    min_val = recipe["min"]
                    max_val = recipe["max"]
                else:
                    min_val, max_val = self._fit_minmax(array)

                self.stats_map[var_name] = {
                    "method": "minmax",
                    "min": min_val,
                    "max": max_val,
                }

            elif method in ("log1p", "sqrt", "pow"):
                # apply the non-linear, then fit zscore on the transformed array
                if method == "log1p":
                    transformed = da.log1p(array)
                elif method == "sqrt":
                    transformed = da.sqrt(array)
                else:  # pow
                    exponent = recipe.get("lambda", 1.0)
                    transformed = array ** exponent

                mean, std = self._fit_zscore(transformed)
                entry: Dict[str, Any] = {
                    "method": method,
                    "mean": mean,
                    "std": std,
                }
                if method == "pow":
                    entry["lambda"] = exponent
                self.stats_map[var_name] = entry

            else:
                raise ValueError(
                    f"Unknown normalization method '{method}' for variable '{var_name}'"
                )

    def transform(self, array, var_name: str):
        """
        Apply normalization to `array` (dask or numpy) for variable `var_name`.
        Returns normalized array of same type.
        """
        info = self.stats_map[var_name]
        method = info["method"]

        if method == "zscore":
            return (array - info["mean"]) / info["std"]

        if method == "minmax":
            denom = info["max"] - info["min"] + 1e-9
            return (array - info["min"]) / denom

        if method == "log1p":
            return (da.log1p(array) - info["mean"]) / info["std"]

        if method == "sqrt":
            return (da.sqrt(array) - info["mean"]) / info["std"]

        if method == "pow":
            exponent = info["lambda"]
            return (array ** exponent - info["mean"]) / info["std"]

        raise ValueError(f"Unsupported method '{method}'")

    def inverse(self, array, var_name: str):
        """
        Inverse-transform a numpy array of normalized outputs back to the original scale.
        """
        info = self.stats_map[var_name]
        method = info["method"]

        if method == "zscore":
            return array * info["std"] + info["mean"]

        if method == "minmax":
            span = info["max"] - info["min"]
            return array * span + info["min"]

        if method == "log1p":
            denorm = array * info["std"] + info["mean"]
            return np.expm1(denorm)

        if method == "sqrt":
            denorm = array * info["std"] + info["mean"]
            return denorm ** 2

        if method == "pow":
            denorm = array * info["std"] + info["mean"]
            exponent = info["lambda"]
            return denorm ** (1 / exponent)

        raise ValueError(f"Unsupported method '{method}'")


def get_trainer_config(cfg: DictConfig, model=None) -> Dict[str, Any]:
    # Setup logger
    if cfg.use_wandb:
        if not cfg.wandb_entity or not cfg.wandb_project:
            raise ValueError("wandb_entity and wandb_project required if use_wandb is true.")
        logger = WandbLogger(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.run_name, log_model=False)
        # Log hyperparameters
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        if model is not None:
            # Watch model gradients etc.
            logger.watch(model, log="all")
    else:
        logger = None

    # Prepare trainer config - convert to dict to allow modification if needed
    trainer_config = OmegaConf.to_container(cfg.trainer)
    trainer_config["logger"] = logger

    # Check if GPU was requested but not available
    if trainer_config.get("accelerator") == "gpu" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.warning("GPU requested but CUDA not available. Using MPS (Apple Silicon) instead.")
            trainer_config["accelerator"] = "mps"
        else:
            log.warning("GPU requested but no GPU available. Falling back to CPU.")
            trainer_config["accelerator"] = "cpu"

    # Initialize callbacks using Hydra's instantiate
    callbacks_list = trainer_config.pop("callbacks", []) or []
    trainer_config["callbacks"] = []
    for callback_config in callbacks_list:
        trainer_config["callbacks"].append(hydra.utils.instantiate(callback_config))

    return trainer_config


# --- Evaluation and Visualization Utilities ---


def create_climate_data_array(data, time_coords, lat_coords, lon_coords, var_name=None, var_unit=None):
    """
    Create a standardized xarray DataArray for climate data.

    Args:
        data: numpy array with shape (time, y, x) or (y, x)
        time_coords: array of time coordinates (or None for 2D data)
        lat_coords: array of latitude coordinates
        lon_coords: array of longitude coordinates
        var_name: optional variable name
        var_unit: optional unit string

    Returns:
        xarray.DataArray with proper dimensions and coordinates
    """
    # Determine dimensions based on data shape
    if len(data.shape) == 3:
        dims = ("time", "y", "x")
        coords = {"time": time_coords, "y": lat_coords, "x": lon_coords}
    else:  # 2D data
        dims = ("y", "x")
        coords = {"y": lat_coords, "x": lon_coords}

    # Create attributes dictionary if name or unit is specified
    attrs = {}
    if var_name:
        attrs["long_name"] = var_name
    if var_unit:
        attrs["units"] = var_unit

    # Create and return the DataArray
    return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)


def calculate_weighted_metric(data_array, weights, dims, metric_type="rmse"):
    """
    Calculate area-weighted metrics for DataArrays.

    Args:
        data_array: xarray DataArray containing the data (typically squared differences or abs diff)
        weights: xarray DataArray with weights matching spatial dimensions
        dims: tuple of dimension names to average over
        metric_type: 'rmse' or 'mae' to determine final calculation

    Returns:
        float: The calculated metric value
    """
    # Apply weights and take mean over specified dimensions
    weighted_mean = data_array.weighted(weights).mean(dim=dims).values

    # Apply final calculation based on metric type
    if metric_type == "rmse":
        return np.sqrt(weighted_mean)
    else:  # mae or other metrics that don't need sqrt
        return weighted_mean


# Default visualization parameters
DEFAULT_VIZ_PARAMS = {
    "standard_cmap": "viridis",  # Default colormap for data
    "diff_cmap": "RdBu_r",  # Default colormap for differences
    "variance_cmap": "plasma",  # Colormap for variance plots
    "colorbar_kwargs": {"fraction": 0.046, "pad": 0.04},
    "figure_size": (18, 6),  # Standard figure size for comparison plots
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
    """
    Create standardized comparison plots between true and predicted data.

    Args:
        true_data: xarray DataArray of ground truth
        pred_data: xarray DataArray of predictions
        title_prefix: String prefix for plot titles
        metric_value: Optional metric value to show in difference plot title
        metric_name: Optional name of the metric to show in difference plot title
        cmap: Colormap for data plots (defaults to DEFAULT_VIZ_PARAMS['standard_cmap'])
        diff_cmap: Colormap for difference plot (defaults to DEFAULT_VIZ_PARAMS['diff_cmap'])
        fig_size: Figure size tuple (defaults to DEFAULT_VIZ_PARAMS['figure_size'])
        colorbar_kwargs: Dictionary of kwargs for colorbar (defaults to DEFAULT_VIZ_PARAMS['colorbar_kwargs'])

    Returns:
        matplotlib figure with 3 subplots (truth, prediction, difference)
    """
    # Use default parameters if not specified
    cmap = cmap or DEFAULT_VIZ_PARAMS["standard_cmap"]
    diff_cmap = diff_cmap or DEFAULT_VIZ_PARAMS["diff_cmap"]
    fig_size = fig_size or DEFAULT_VIZ_PARAMS["figure_size"]
    colorbar_kwargs = colorbar_kwargs or DEFAULT_VIZ_PARAMS["colorbar_kwargs"]
    fig, axes = plt.subplots(1, 3, figsize=fig_size)

    # Find global min/max for consistent color scaling
    vmin = min(true_data.min().item(), pred_data.min().item())
    vmax = max(true_data.max().item(), pred_data.max().item())

    # Common plotting parameters
    plot_params = {"vmin": vmin, "vmax": vmax, "add_colorbar": True, "cbar_kwargs": colorbar_kwargs}

    # Plot ground truth
    true_data.plot(ax=axes[0], cmap=cmap, **plot_params)
    axes[0].set_title(f"{title_prefix} (Ground Truth)")

    # Plot prediction
    pred_data.plot(ax=axes[1], cmap=cmap, **plot_params)
    axes[1].set_title(f"{title_prefix} (Prediction)")

    # Plot difference
    diff = pred_data - true_data
    diff_max = max(abs(diff.min().item()), abs(diff.max().item()))

    # Override min/max for difference plot to be centered at zero
    diff_plot_params = plot_params.copy()
    diff_plot_params.update({"vmin": -diff_max, "vmax": diff_max, "cmap": diff_cmap})

    diff.plot(ax=axes[2], **diff_plot_params)

    # Add metric to title if provided
    if metric_value is not None and metric_name is not None:
        metric_text = f" ({metric_name}: {metric_value:.4f})"
    else:
        metric_text = ""

    axes[2].set_title(f"Difference{metric_text}")

    plt.tight_layout()
    return fig


def get_lat_weights(latitude_values):
    """
    Compute area weights based on latitude values.

    Args:
        latitude_values: Array of latitude values

    Returns:
        Array of weights with the same shape as latitude_values
    """
    # Convert latitude values to radians
    lat_rad = np.deg2rad(latitude_values)

    # Calculate weights as cosine of latitude (proportional to grid cell area)
    weights = np.cos(lat_rad)

    # Normalize weights to mean=1.0
    weights = weights / np.mean(weights)

    return weights


def convert_predictions_to_kaggle_format(predictions, time_coords, lat_coords, lon_coords, var_names):
    """
    Convert climate model predictions to Kaggle submission format.

    Args:
        predictions (np.ndarray): Predicted values with shape (time, channels, y, x)
        time_coords (np.ndarray): Time coordinate values
        lat_coords (np.ndarray): Latitude coordinate values
        lon_coords (np.ndarray): Longitude coordinate values
        var_names (list): List of variable names corresponding to the channel dimension

    Returns:
        pandas.DataFrame: DataFrame with columns 'ID' and 'Prediction' in Kaggle submission format
    """
    try:
        import pandas as pd

        # Create a list to hold all data rows
        rows = []

        # Loop through all dimensions to create flattened data
        for t_idx, t in enumerate(time_coords):
            for var_idx, var_name in enumerate(var_names):
                for y_idx, lat in enumerate(lat_coords):
                    for x_idx, lon in enumerate(lon_coords):
                        # Get the predicted value
                        pred_value = predictions[t_idx, var_idx, y_idx, x_idx]

                        # Create row ID: format as time_variable_lat_lon
                        row_id = f"t{t_idx:03d}_{var_name}_{lat:.2f}_{lon:.2f}"

                        # Add to rows list
                        rows.append({"ID": row_id, "Prediction": pred_value})

        # Create DataFrame
        submission_df = pd.DataFrame(rows)
        return submission_df

    except Exception as e:
        log.error(f"Failed to convert predictions to Kaggle format: {str(e)}")
        raise