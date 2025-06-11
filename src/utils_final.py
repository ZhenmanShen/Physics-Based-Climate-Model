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
    def __init__(self):
        self.input_stats = {}  # Expects index-keyed map: {0: {'method': ..., 'params': {...}}, ...}
        self.output_stats = {} # Expects index-keyed map

    def set_input_statistics(self, transform_map_indexed):
        log.info(f"Normalizer A: Setting input stats with index-keyed map: {list(transform_map_indexed.keys())}")
        self.input_stats = transform_map_indexed

    def set_output_statistics(self, transform_map_indexed):
        log.info(f"Normalizer A: Setting output stats with index-keyed map: {list(transform_map_indexed.keys())}")
        self.output_stats = transform_map_indexed

    def normalize(self, data, data_type="input"):
        stats_collection = self.input_stats if data_type == "input" else self.output_stats
        if not stats_collection:
            raise RuntimeError(f"Statistics for '{data_type}' not set.")

        is_dask_array = isinstance(data, da.Array)
        if not (isinstance(data, np.ndarray) or is_dask_array):
            raise TypeError("Input 'data' must be a NumPy or Dask array.")

        num_variables = data.shape[1]
        output_slices = []
        epsilon = 1e-8

        for var_idx in range(num_variables):
            current_var_data_slice = data[:, var_idx, :, :]
            var_config = stats_collection.get(var_idx)

            if var_config is None:
                log.warning(f"No config for var index {var_idx} ({data_type}). Passing through.")
                output_slices.append(current_var_data_slice)
                continue

            method = var_config['method']
            params = var_config.get('params', {})
            transformed_slice = None

            if method == "zscore":
                mean = params.get('mean')
                std = params.get('std')
                if mean is None or std is None:
                    raise ValueError(f"Z-score params missing for var {var_idx}.")
                transformed_slice = (current_var_data_slice - mean) / (std + epsilon)
            elif method == "minimax":
                min_val = params.get('min_val')
                max_val = params.get('max_val')
                if min_val is None or max_val is None:
                    raise ValueError(f"Minimax params missing for var {var_idx}.")
                range_val = max_val - min_val
                current_scale = range_val if not np.isclose(range_val, 0) else 1.0 # Simplified for scalar/numpy
                if hasattr(range_val, "__iter__") and not isinstance(range_val, str): # Array-like stats
                    current_scale = (da.where(da.isclose(range_val, 0), 1.0, range_val) if is_dask_array 
                                     else np.where(np.isclose(range_val, 0), 1.0, range_val))
                transformed_slice = (current_var_data_slice - min_val) / current_scale
            
            # --- BEHAVIOR CHANGE TO MATCH NORMALIZER B ---
            elif method == "log1p":
                # Params 'mean' and 'std' are of the log1p-transformed data
                mean_of_log = params.get('mean')
                std_of_log = params.get('std')
                if mean_of_log is None or std_of_log is None:
                    raise ValueError(f"log1p method for var {var_idx} requires 'mean' and 'std' of log-transformed data in params.")
                
                # Apply log1p first
                data_after_log1p = da.log1p(current_var_data_slice) if is_dask_array else np.log1p(current_var_data_slice)
                # Then standardize
                transformed_slice = (data_after_log1p - mean_of_log) / (std_of_log + epsilon)

            elif method == "sqrt":
                # Params 'mean' and 'std' are of the sqrt-transformed data
                mean_of_sqrt = params.get('mean')
                std_of_sqrt = params.get('std')
                if mean_of_sqrt is None or std_of_sqrt is None:
                    raise ValueError(f"sqrt method for var {var_idx} requires 'mean' and 'std' of sqrt-transformed data in params.")

                data_after_sqrt = da.sqrt(current_var_data_slice) if is_dask_array else np.sqrt(current_var_data_slice)
                transformed_slice = (data_after_sqrt - mean_of_sqrt) / (std_of_sqrt + epsilon)

            elif method == "pow":
                # Params 'lambda', 'mean', 'std' (mean/std are of power-transformed data)
                exponent = params.get('lambda')
                mean_of_pow = params.get('mean')
                std_of_pow = params.get('std')
                if exponent is None or mean_of_pow is None or std_of_pow is None:
                     raise ValueError(f"pow method for var {var_idx} requires 'lambda', 'mean', and 'std' in params.")

                data_after_pow = current_var_data_slice ** exponent
                transformed_slice = (data_after_pow - mean_of_pow) / (std_of_pow + epsilon)
            # --- END OF BEHAVIOR CHANGE ---
            else:
                raise ValueError(f"Unknown method '{method}' for var {var_idx}.")
            
            output_slices.append(transformed_slice)

        return da.stack(output_slices, axis=1) if is_dask_array else np.stack(output_slices, axis=1)

    def inverse_transform_output(self, data_norm):
        stats_collection = self.output_stats
        if not stats_collection:
            raise RuntimeError("Output stats not set.")

        is_dask_array = isinstance(data_norm, da.Array)
        if not (isinstance(data_norm, np.ndarray) or is_dask_array):
            raise TypeError("Input 'data_norm' must be a NumPy or Dask array.")

        num_variables = data_norm.shape[1]
        output_slices = []
        epsilon = 1e-8 # Not typically needed for inverse unless std was 0

        for var_idx in range(num_variables):
            current_var_data_norm_slice = data_norm[:, var_idx, :, :]
            var_config = stats_collection.get(var_idx)

            if var_config is None:
                log.warning(f"No de-norm config for output var index {var_idx}. Passing through.")
                output_slices.append(current_var_data_norm_slice)
                continue
            
            method = var_config['method']
            params = var_config.get('params', {})
            denormalized_slice = None

            if method == "zscore":
                mean = params.get('mean')
                std = params.get('std')
                if mean is None or std is None:
                    raise ValueError(f"Z-score params missing for inverse for var {var_idx}.")
                denormalized_slice = current_var_data_norm_slice * std + mean
            elif method == "minimax":
                min_val = params.get('min_val')
                max_val = params.get('max_val')
                if min_val is None or max_val is None:
                    raise ValueError(f"Minimax params missing for inverse for var {var_idx}.")
                range_val = max_val - min_val
                denormalized_slice = current_var_data_norm_slice * range_val + min_val

            # --- BEHAVIOR CHANGE TO MATCH NORMALIZER B ---
            elif method == "log1p":
                mean_of_log = params.get('mean')
                std_of_log = params.get('std')
                if mean_of_log is None or std_of_log is None:
                    raise ValueError(f"log1p inverse params missing for var {var_idx}.")
                # De-standardize first
                de_standardized_slice = current_var_data_norm_slice * std_of_log + mean_of_log
                # Then apply inverse non-linear
                denormalized_slice = da.expm1(de_standardized_slice) if is_dask_array else np.expm1(de_standardized_slice)
            
            elif method == "sqrt":
                mean_of_sqrt = params.get('mean')
                std_of_sqrt = params.get('std')
                if mean_of_sqrt is None or std_of_sqrt is None:
                    raise ValueError(f"sqrt inverse params missing for var {var_idx}.")
                de_standardized_slice = current_var_data_norm_slice * std_of_sqrt + mean_of_sqrt
                denormalized_slice = de_standardized_slice ** 2
            
            elif method == "pow":
                exponent = params.get('lambda')
                mean_of_pow = params.get('mean')
                std_of_pow = params.get('std')
                if exponent is None or mean_of_pow is None or std_of_pow is None:
                    raise ValueError(f"pow inverse params missing for var {var_idx}.")
                de_standardized_slice = current_var_data_norm_slice * std_of_pow + mean_of_pow
                # Handle potential negative numbers if exponent is non-integer before raising to 1/exponent
                # This depends on the domain of your data after power transform.
                # For simplicity, assuming de_standardized_slice will be non-negative if 1/exponent requires it.
                denormalized_slice = de_standardized_slice ** (1.0 / exponent)
            # --- END OF BEHAVIOR CHANGE ---
            else:
                raise ValueError(f"Unknown inverse method '{method}' for var {var_idx}.")

            output_slices.append(denormalized_slice)
        
        return da.stack(output_slices, axis=1) if is_dask_array else np.stack(output_slices, axis=1)


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