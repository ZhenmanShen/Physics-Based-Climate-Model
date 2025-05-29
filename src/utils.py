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
    """
    Helper class for Z-score normalization. Stores mean/std as NumPy arrays.

    Applies the standard normalization formula: (data - mean) / std
    """

    def __init__(self):
        """Initialize the normalizer with empty parameters."""
        self.mean_in, self.std_in = None, None
        self.mean_out, self.std_out = None, None

    def set_input_statistics(self, mean, std):
        """
        Set normalization parameters for input features.

        Args:
            mean: Mean values for input normalization, shape [num_channels, 1, 1]
            std: Standard deviation values for input normalization, shape [num_channels, 1, 1]
        """
        log.info(f"Setting input normalizer with mean shape: {mean.shape}, std shape: {std.shape}")
        self.mean_in = mean
        self.std_in = std

    def set_output_statistics(self, mean, std):
        """
        Set normalization parameters for output values.

        Args:
            mean: Mean value(s) for output normalization
            std: Standard deviation value(s) for output normalization
        """
        log.info(f"Setting output normalizer with mean shape: {mean.shape}, std shape: {std.shape}")
        self.mean_out = mean
        self.std_out = std

    def normalize(self, data, data_type="input"):
        """
        Normalize data using fitted mean and std values

        Args:
            data: Input data to normalize (numpy array or dask array)
                 Expected shapes:
                 - input: (time, channels, y, x)
                 - output: (time, C, y, x) - channel dimension should already be added
            data_type: Either 'input' or 'output' to specify which normalization to use

        Returns:
            Normalized data with same type as input
        """
        if data_type == "input":
            if self.mean_in is None or self.std_in is None:
                raise RuntimeError("Must fit input normalizer before normalizing input data")
            return (data - self.mean_in) / self.std_in
        elif data_type == "output":
            if self.mean_out is None or self.std_out is None:
                raise RuntimeError("Must fit output normalizer before normalizing output data")
            # Output data should already have channel dimension
            return (data - self.mean_out) / self.std_out
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'input' or 'output'")

    def inverse_transform_output(self, data_norm):
        """
        Denormalize output data back to original scale

        Args:
            data_norm: Normalized output data with shape (..., C, lat, lon)

        Returns:
            Denormalized data in same format as input
        """
        if self.mean_out is None or self.std_out is None:
            raise RuntimeError("Must fit output normalizer before inverse transforming")

        # Handles broadcasting correctly
        denormalized = data_norm * self.std_out + self.mean_out
        return denormalized

class AdvancedNormalizer:
    def __init__(self, input_vars, output_vars):
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.input_stats = {}
        self.output_stats = {}
    
    def fit(self, input_dask, output_dask):
        """Compute per-variable normalization statistics using Dask"""
        # Compute statistics for each input variable
        for i, var in enumerate(self.input_vars):
            var_data = input_dask[:, i, :, :]
            
            if var in ['SO2', 'BC']:
                # Sparse pollutants: log-normal for non-zero values
                # Compute statistics only for non-zero values
                non_zero_mask = (var_data > 1e-6)
                non_zero_count = da.sum(non_zero_mask).compute()
                
                if non_zero_count > 0:
                    non_zero_data = var_data[non_zero_mask]
                    log_mean = da.nanmean(da.log1p(non_zero_data)).compute()
                    log_std = da.nanstd(da.log1p(non_zero_data)).compute()
                else:
                    log_mean, log_std = 0.0, 1.0  # Defaults if no non-zero values
                
                self.input_stats[var] = {
                    'type': 'sparse',
                    'log_mean': log_mean,
                    'log_std': log_std
                }
            
            elif var == 'CO2':
                # Cube root transform for skewed distribution
                cube_root = da.sign(var_data) * da.power(da.abs(var_data), 1/3)
                mean = da.nanmean(cube_root).compute()
                std = da.nanstd(cube_root).compute()
                
                self.input_stats[var] = {
                    'type': 'skewed',
                    'transform': 'cube_root',
                    'mean': mean,
                    'std': std
                }
            
            elif var == 'rsdt':
                # Bimodal handling - compute statistics for values above threshold
                normal_mask = (var_data >= 50)
                normal_count = da.sum(normal_mask).compute()
                
                if normal_count > 0:
                    normal_data = var_data[normal_mask]
                    normal_mean = da.nanmean(normal_data).compute()
                    normal_std = da.nanstd(normal_data).compute()
                else:
                    normal_mean, normal_std = 300.0, 50.0  # Reasonable defaults
                
                self.input_stats[var] = {
                    'type': 'bimodal',
                    'threshold': 50,
                    'normal_mean': normal_mean,
                    'normal_std': normal_std
                }
            
            else:  # CH4 and others
                # Use robust statistics that Dask can compute efficiently
                mean = da.nanmean(var_data).compute()
                std = da.nanstd(var_data).compute()
                min_val = da.nanmin(var_data).compute()
                max_val = da.nanmax(var_data).compute()
                
                self.input_stats[var] = {
                    'type': 'standard',
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val
                }
        
        # Compute statistics for output variables
        for i, var in enumerate(self.output_vars):
            var_data = output_dask[:, i, :, :]
            
            # For outputs, use standard normalization that Dask can handle
            mean = da.nanmean(var_data).compute()
            std = da.nanstd(var_data).compute()
            
            if var == 'pr':
                # Additional precipitation-specific stats
                min_val = da.nanmin(var_data).compute()
                self.output_stats[var] = {
                    'type': 'log_standard',
                    'mean': mean,
                    'std': std,
                    'min': min_val
                }
            else:
                self.output_stats[var] = {
                    'type': 'standard',
                    'mean': mean,
                    'std': std
                }
    
    def normalize_input(self, input_dask):
        """Normalize input data per variable"""
        normalized_channels = []
        for i, var in enumerate(self.input_vars):
            var_data = input_dask[:, i, :, :]
            stats = self.input_stats[var]
            
            if stats['type'] == 'sparse':
                # Apply log transform to non-zero values
                log_data = da.log1p(var_data)
                normalized = da.where(
                    var_data > 1e-6,
                    (log_data - stats['log_mean']) / stats['log_std'],
                    0.0
                )
            
            elif stats['type'] == 'skewed':
                cube_root = da.sign(var_data) * da.power(da.abs(var_data), 1/3)
                normalized = (cube_root - stats['mean']) / stats['std']
            
            elif stats['type'] == 'bimodal':
                # Separate handling for polar regions
                normalized = da.where(
                    var_data < stats['threshold'],
                    var_data / stats['threshold'],  # Scale to [0,1]
                    (var_data - stats['normal_mean']) / stats['normal_std']
                )
            
            else:  # Standard z-score
                normalized = (var_data - stats['mean']) / stats['std']
            
            normalized_channels.append(normalized)
        
        return da.stack(normalized_channels, axis=1)
    
    def normalize_output(self, output_dask):
        """Normalize output data per variable"""
        normalized_channels = []
        for i, var in enumerate(self.output_vars):
            var_data = output_dask[:, i, :, :]
            stats = self.output_stats[var]
            
            if stats['type'] == 'log_standard':
                # Log transform for precipitation
                log_data = da.log1p(var_data - stats['min'])
                normalized = (log_data - stats['mean']) / stats['std']
            else:
                normalized = (var_data - stats['mean']) / stats['std']
            
            normalized_channels.append(normalized)
        
        return da.stack(normalized_channels, axis=1)
    
    def denormalize_output(self, output_norm_dask):
        """Convert normalized output back to physical units"""
        denormalized_channels = []
        for i, var in enumerate(self.output_vars):
            var_data = output_norm_dask[:, i, :, :]
            stats = self.output_stats[var]
            
            if stats['type'] == 'log_standard':
                log_data = var_data * stats['std'] + stats['mean']
                denormalized = da.expm1(log_data) + stats['min']
            else:
                denormalized = var_data * stats['std'] + stats['mean']
            
            denormalized_channels.append(denormalized)
        
        return da.stack(denormalized_channels, axis=1)

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

    true_data_np = true_data.values
    pred_data_np = pred_data.values

    # Find global min/max for consistent color scaling
    vmin = min(np.nanmin(true_data_np), np.nanmin(pred_data_np))
    vmax = max(np.nanmax(true_data_np), np.nanmax(pred_data_np))
    
    # # Find global min/max for consistent color scaling
    # vmin = min(true_data.min().item(), pred_data.min().item())
    # vmax = max(true_data.max().item(), pred_data.max().item())

    # Common plotting parameters
    plot_params = {"vmin": vmin, "vmax": vmax, "add_colorbar": True, "cbar_kwargs": colorbar_kwargs}

    # Plot ground truth
    true_data.plot(ax=axes[0], cmap=cmap, **plot_params)
    axes[0].set_title(f"{title_prefix} (Ground Truth)")

    # Plot prediction
    pred_data.plot(ax=axes[1], cmap=cmap, **plot_params)
    axes[1].set_title(f"{title_prefix} (Prediction)")

    # # Plot difference
    # diff = pred_data - true_data
    # diff_max = max(abs(diff.min().item()), abs(diff.max().item()))
    # Plot difference
    diff = pred_data - true_data
    diff_np = pred_data_np - true_data_np
    diff_max = max(abs(np.nanmin(diff_np)), abs(np.nanmax(diff_np)))

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