import os
from datetime import datetime

import dask.array as da
import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from hydra.utils import to_absolute_path
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


try:
    import wandb  # Optional, for logging to Weights & Biases
except ImportError:
    wandb = None

from src.models import get_model
from src.utils import (
    Normalizer,
    calculate_weighted_metric,
    convert_predictions_to_kaggle_format,
    create_climate_data_array,
    create_comparison_plots,
    get_lat_weights,
    get_logger,
    get_trainer_config,
)


# Setup logging
log = get_logger(__name__)


# --- Data Handling ---


# Dataset to precompute all tensors during initialization
class ClimateDataset(Dataset):
    def __init__(self, inputs_norm_dask, outputs_dask, seq_len: int, output_is_normalized=True):
        log.info(
            f"Initializing ClimateDataset. Input dask shape: {inputs_norm_dask.shape}, "
            f"Output dask shape: {outputs_dask.shape}, seq_len: {seq_len}, output_normalized: {output_is_normalized}"
        )

        # Precompute all tensors in one go
        inputs_np = inputs_norm_dask.compute()
        outputs_np = outputs_dask.compute() # This is the target data

        self.input_tensors = torch.from_numpy(inputs_np).float()
        self.output_tensors = torch.from_numpy(outputs_np).float() # Targets
        self.seq_len = seq_len
        self.total_timesteps = inputs_np.shape[0] # Total individual months in this data chunk

        # --- MODIFIED FOR PADDING ---
        # self.size will be the total number of timesteps, as we aim to predict for each.
        self.size = self.total_timesteps
        # --- END OF MODIFICATION ---

        log.info(
            f"Dataset created. Input tensor shape: {self.input_tensors.shape}, "
            f"Output tensor shape: {self.output_tensors.shape}, "
            f"Effective dataset size (number of targets to predict): {self.size}"
        )

        # Determine padding shape once.
        # This assumes input_tensors[0] gives a single timestep (C, H, W).
        if self.total_timesteps > 0 and self.input_tensors.numel() > 0 :
            self.pad_tensor_template = torch.zeros_like(self.input_tensors[0], dtype=self.input_tensors.dtype, device=self.input_tensors.device)
            log.info(f"Padding template created with shape: {self.pad_tensor_template.shape}")
        elif self.seq_len > 0 : # If seq_len > 0 but no data to infer padding shape
             log.warning(
                f"Input tensors are empty or total_timesteps is 0, but seq_len is {self.seq_len}. "
                "Cannot create a padding template. This will cause errors if padding is needed."
             )
             self.pad_tensor_template = None # Will cause error if used later
        else: # seq_len is 0 or no data, no padding needed.
            self.pad_tensor_template = None


        # Handle NaN values
        if torch.isnan(self.input_tensors).any() or torch.isnan(self.output_tensors).any():
            log.warning("NaN values detected in dataset tensors after .compute(). This might lead to issues.")


    def __len__(self):
        # This now returns the total number of timesteps, as we'll predict for each one.
        return self.size

    def __getitem__(self, idx):
        # idx will range from 0 to self.total_timesteps - 1.
        # We want to predict the output for self.output_tensors[idx].
        # The UNetWithConvLSTM's last LSTM output is used for the decoder,
        # so it predicts the output corresponding to the last timestep of its input sequence,
        # OR the step immediately after, depending on its exact design and training.
        #
        # Let's assume the LSTM model, given an input sequence X_t, X_{t+1}, ..., X_{t+T-1},
        # is trained to predict Y_{t+T-1} (output at the end of the sequence window)
        # OR Y_{t+T} (output for the step after the sequence window).
        #
        # The provided UNetWithConvLSTM structure:
        #   lstm_out_seq = self.conv_lstm_bottleneck(bottleneck_seq)
        #   x_decoder_input = lstm_out_seq[-1, :, :, :, :]
        # This uses the LSTM's hidden state from the *last actual input time step* of the sequence.
        # This state is then used to make a prediction. This prediction usually corresponds to
        # the output at that *same last time step* or the *next time step*.
        #
        # If your model predicts Y_k given input sequence X_{k-T+1}...X_k:
        # Target is output_tensors[idx]
        # Input sequence needs to end at input_tensors[idx]
        # Input sequence indices: [idx - seq_len + 1, ..., idx]

        target = self.output_tensors[idx]
        input_sequence_parts = []

        for i in range(self.seq_len):
            # We need input from timestep (idx - seq_len + 1 + i) to predict for target at idx
            current_input_idx = idx - self.seq_len + 1 + i

            if current_input_idx < 0:
                # This part of the history is before the start of our available data, so pad.
                if self.pad_tensor_template is None:
                    raise RuntimeError("Padding template not available. Input data might be empty or seq_len is 0.")
                input_sequence_parts.append(self.pad_tensor_template)
            else:
                # Ensure we don't try to access beyond the actual input data if idx is very large
                # (though this shouldn't happen if idx < self.total_timesteps and logic is correct)
                if current_input_idx < self.total_timesteps:
                     input_sequence_parts.append(self.input_tensors[current_input_idx])
                else:
                    # This case implies an issue with indexing logic if idx is a valid target index.
                    # For input sequence X_{k-T+1}...X_k to predict Y_k, max current_input_idx is k.
                    # Since max k (idx) is total_timesteps-1, this should be fine.
                    # However, if something is off, pad as a fallback.
                    log.warning(f"Attempted to access input_idx {current_input_idx} beyond total_timesteps {self.total_timesteps} for target_idx {idx}. Padding instead.")
                    if self.pad_tensor_template is None:
                         raise RuntimeError("Padding template not available for fallback.")
                    input_sequence_parts.append(self.pad_tensor_template)


        if len(input_sequence_parts) != self.seq_len:
            # This should ideally not happen with the loop structure
            raise RuntimeError(f"Constructed input sequence has length {len(input_sequence_parts)}, expected {self.seq_len}")

        input_seq = torch.stack(input_sequence_parts, dim=0) # Stacks along new dim 0 -> (seq_len, C, H, W)
        
        return input_seq, target

def _load_process_ssp_data(
    ds,
    ssp: str,
    input_variables: list[str],
    output_variables: list[str],
    member_ids: list[int],
    spatial_template: xr.DataArray,
):
    """
    Enhanced with temporal features and variable interactions
    without requiring new data variables
    """
    input_members, output_members = [], []
    n_y, n_x = spatial_template.sizes['y'], spatial_template.sizes['x']
    
    for m in member_ids:
        ssp_input_dasks = []
        # PROCESS OUTPUT TO GET TIME DIMENSION
        da_out0 = ds[output_variables[0]].sel(ssp=ssp, member_id=m)
        if "latitude" in da_out0.dims:
            da_out0 = da_out0.rename({"latitude": "y", "longitude": "x"})
        time_coord = da_out0.time
        n_time = len(time_coord)
        year = time_coord.dt.year.data
        month_idx = (time_coord.dt.month - 1).data
        
        # ===== ENHANCED TEMPORAL FEATURES =====
        # 1. Seasonal signals (existing)
        sin_month = da.sin(2 * np.pi * month_idx / 12).reshape(n_time, 1, 1)
        cos_month = da.cos(2 * np.pi * month_idx / 12).reshape(n_time, 1, 1)
        
        # 2. Decadal progression (new)
        decade_progress = (year - year.min()) / 10  # Linear decade trend
        decade_progress = decade_progress.reshape(n_time, 1, 1)
        
        # Broadcast temporal features
        temporal_features = [
            da.broadcast_to(feat, (n_time, n_y, n_x))
            for feat in [sin_month, cos_month, decade_progress]
        ]
        
        # ===== INPUT PROCESSING =====
        input_arrays = {}  # Store for interaction calculations
        for var in input_variables:
            da_var = ds[var].sel(ssp=ssp)
            if "member_id" in da_var.dims:
                da_var = da_var.sel(member_id=m)
            if "latitude" in da_var.dims:
                da_var = da_var.rename({"latitude": "y", "longitude": "x"})
                
            # Handle different variable types
            if set(da_var.dims) == {"time"}:
                da_var = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
            elif set(da_var.dims) != {"time", "y", "x"}:
                raise ValueError(f"Unexpected dims {da_var.dims} for {var}")
                
            # Store for potential interactions
            input_arrays[var] = da_var.data
            ssp_input_dasks.append(da_var.data)
        
        # ===== ADD ENGINEERED FEATURES =====
        # 1. Temporal features
        ssp_input_dasks.extend(temporal_features)
        
        # 2. Variable interactions (using existing inputs)
        # Aerosol interaction
        if 'SO2' in input_arrays and 'BC' in input_arrays:
            aerosol_interaction = input_arrays['SO2'] * input_arrays['BC']
            ssp_input_dasks.append(aerosol_interaction)
            
        # Greenhouse gas ratio
        if 'CO2' in input_arrays and 'CH4' in input_arrays:
            # Use safe division to avoid NaN
            ghg_ratio = input_arrays['CO2'] / da.maximum(input_arrays['CH4'], 1e-6)
            ssp_input_dasks.append(ghg_ratio)
            
        # Radiation-adjusted CO2 (logarithmic forcing approximation)
        if 'CO2' in input_arrays and 'rsdt' in input_arrays:
            # Use log(1 + CO2) to approximate radiative forcing
            co2_rad = da.log(1 + input_arrays['CO2']) * input_arrays['rsdt']
            ssp_input_dasks.append(co2_rad)
        
        # Stack inputs (time × C × y × x)
        input_members.append(da.stack(ssp_input_dasks, axis=1))
        
        # ===== OUTPUT PROCESSING =====
        ssp_output_dasks = []
        for var in output_variables:
            da_out = ds[var].sel(ssp=ssp, member_id=m)
            if "latitude" in da_out.dims:
                da_out = da_out.rename({"latitude": "y", "longitude": "x"})
            ssp_output_dasks.append(da_out.data)
        output_members.append(da.stack(ssp_output_dasks, axis=1))

    # Concatenate members along time dimension
    stacked_input = da.concatenate(input_members, axis=0)
    stacked_output = da.concatenate(output_members, axis=0)
    return stacked_input, stacked_output
# def _load_process_ssp_data(
#     ds,
#     ssp: str,
#     input_variables: list[str],
#     output_variables: list[str],
#     member_ids: list[int],
#     spatial_template: xr.DataArray,
# ):
#     """
#     Returns inputs & outputs for **all requested ensemble members**,
#     concatenated on the *time* axis (time × channels × y × x).
#     """
#     input_members, output_members = [], []
    
#     # Precompute spatial dimensions from template
#     n_y, n_x = spatial_template.sizes['y'], spatial_template.sizes['x']
    
#     for m in member_ids:
#         ssp_input_dasks = []
#         time_coord = None

#         # ---------- PROCESS OUTPUT FIRST TO GET TIME DIMENSION ----------
#         # Get time coordinate from output variable (guaranteed to have full spatiotemporal dims)
#         da_out0 = ds[output_variables[0]].sel(ssp=ssp, member_id=m)
#         if "latitude" in da_out0.dims:
#             da_out0 = da_out0.rename({"latitude": "y", "longitude": "x"})
#         time_coord = da_out0.time
#         n_time = len(time_coord)

#         # ---------- SEASONAL SIGNAL GENERATION ----------
#         # Compute month indices (0-11) using dask
#         month_idx = (time_coord.dt.month - 1).data  # Dask array
        
#         # Compute seasonal signals with automatic broadcasting
#         sin_month = da.sin(2 * np.pi * month_idx / 12).reshape(n_time, 1, 1)
#         cos_month = da.cos(2 * np.pi * month_idx / 12).reshape(n_time, 1, 1)
        
#         # Broadcast to spatial dimensions
#         sin_broadcast = da.broadcast_to(sin_month, (n_time, n_y, n_x))
#         cos_broadcast = da.broadcast_to(cos_month, (n_time, n_y, n_x))
        
#         # ---------- INPUT PROCESSING WITH SEASONAL CHANNELS ----------
#         for var in input_variables:
#             da_var = ds[var].sel(ssp=ssp)
#             if "member_id" in da_var.dims:
#                 da_var = da_var.sel(member_id=m)
#             if "latitude" in da_var.dims:
#                 da_var = da_var.rename({"latitude": "y", "longitude": "x"})
                
#             # Handle global variables (time-only)
#             if set(da_var.dims) == {"time"}:
#                 da_var = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
#             elif set(da_var.dims) != {"time", "y", "x"}:
#                 raise ValueError(f"Unexpected dims {da_var.dims} for {var}")
                
#             ssp_input_dasks.append(da_var.data)
        
#         # Append seasonal signals as new channels
#         ssp_input_dasks.append(sin_broadcast)
#         ssp_input_dasks.append(cos_broadcast)
        
#         # Stack all input channels (time × C × y × x)
#         input_members.append(da.stack(ssp_input_dasks, axis=1))

#         # ---------- OUTPUT PROCESSING (unchanged) ----------
#         ssp_output_dasks = []
#         for var in output_variables:
#             da_out = ds[var].sel(ssp=ssp, member_id=m)
#             if "latitude" in da_out.dims:
#                 da_out = da_out.rename({"latitude": "y", "longitude": "x"})
#             ssp_output_dasks.append(da_out.data)
#         output_members.append(da.stack(ssp_output_dasks, axis=1))

#     # Concatenate members along time dimension
#     stacked_input = da.concatenate(input_members, axis=0)
#     stacked_output = da.concatenate(output_members, axis=0)
#     return stacked_input, stacked_output
    
    # input_members, output_members = [], []

    # for m in member_ids:
    #     ssp_input_dasks, ssp_output_dasks = [], []

    #     # ---------- INPUTS ----------
    #     for var in input_variables:
    #         da_var = ds[var].sel(ssp=ssp)
    #         if "latitude" in da_var.dims:   # rename spatial dims once
    #             da_var = da_var.rename({"latitude": "y", "longitude": "x"})
    #         if "member_id" in da_var.dims:
    #             da_var = da_var.sel(member_id=m)

    #         if set(da_var.dims) == {"time"}:        # global -> broadcast
    #             da_var = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
    #         elif set(da_var.dims) != {"time", "y", "x"}:
    #             raise ValueError(f"Unexpected dims {da_var.dims} for {var}")

    #         ssp_input_dasks.append(da_var.data)

    #     # time × C_in × y × x  (for one member)
    #     input_members.append(da.stack(ssp_input_dasks, axis=1))

    #     # ---------- OUTPUTS ----------
    #     for var in output_variables:
    #         da_out = ds[var].sel(ssp=ssp, member_id=m)
    #         if "latitude" in da_out.dims:
    #             da_out = da_out.rename({"latitude": "y", "longitude": "x"})
    #         ssp_output_dasks.append(da_out.data)

    #     # time × C_out × y × x  (for one member)
    #     output_members.append(da.stack(ssp_output_dasks, axis=1))

    # # concat the *members* along time, keeping chronology per member
    # stacked_input = da.concatenate(input_members, axis=0)
    # stacked_output = da.concatenate(output_members, axis=0)
    # return stacked_input, stacked_output


class ClimateEmulationDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        input_vars: list,
        output_vars: list,
        train_ssps: list,
        test_ssp: str,
        seq_len: int,
        transform_map: dict,
        member_ids: list[int] = (0,),
        test_months: int = 360,
        batch_size: int = 32,
        eval_batch_size: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.path = to_absolute_path(path)
        self.normalizer = Normalizer()
        self.hparams.member_ids = list(member_ids)
        self.hparams.transform_map = transform_map
        self.hparams.seq_len = seq_len

        # Set evaluation batch size to training batch size if not specified
        if eval_batch_size is None:
            self.hparams.eval_batch_size = batch_size

        # Placeholders
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.lat_coords, self.lon_coords, self._lat_weights_da = None, None, None

    def prepare_data(self):
        if not os.path.exists(self.hparams.path):
            raise FileNotFoundError(f"Data path not found: {self.hparams.path}")
        log.info(f"Data found at: {self.hparams.path}")

    def setup(self, stage: str | None = None):
        log.info(f"Setting up data module for stage: {stage} from {self.hparams.path}")

        with xr.open_zarr(self.hparams.path, consolidated=True, chunks={"time": 24}) as ds:
            spatial_template_da = ds["rsdt"].isel(time=0, ssp=0, drop=True)

            train_inputs_dask_list, train_outputs_dask_list = [], []
            val_input_dask, val_output_dask = None, None
            val_ssp = "ssp370" 
            val_months = 1080 

            log.info(f"Loading data from SSPs: {self.hparams.train_ssps}")
            for ssp in self.hparams.train_ssps:
                ssp_input_dask, ssp_output_dask = _load_process_ssp_data(
                    ds, ssp, self.hparams.input_vars, self.hparams.output_vars,
                    self.hparams.member_ids, spatial_template_da,
                )

                if ssp == val_ssp:
                    val_input_dask = ssp_input_dask[-val_months:]
                    val_output_dask = ssp_output_dask[-val_months:]
                    if ssp_input_dask.shape[0] > val_months: 
                        train_inputs_dask_list.append(ssp_input_dask[:-val_months])
                        train_outputs_dask_list.append(ssp_output_dask[:-val_months])
                else:
                    train_inputs_dask_list.append(ssp_input_dask)
                    train_outputs_dask_list.append(ssp_output_dask)
            
            if not train_inputs_dask_list: 
                raise ValueError("No training data available. Check SSP configuration and val_months.")

            train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
            train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)

            # --- Calculate overall statistics for normalization (used if not pre-defined or for specific methods) ---
            # These are calculated on the original training data for each channel.
            # Shape: (1, num_channels, 1, 1) -> then squeezed to scalar per channel for params.
            overall_input_mean = da.nanmean(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            overall_input_std = da.nanstd(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            overall_input_min = da.nanmin(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            overall_input_max = da.nanmax(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()

            overall_output_mean = da.nanmean(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            overall_output_std = da.nanstd(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            overall_output_min = da.nanmin(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            overall_output_max = da.nanmax(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()

            # --- Construct INDEX-KEYED transform_map for Normalizer A ---
            input_transform_map_indexed = {}
            for i, var_name in enumerate(self.hparams.input_vars):
                var_config_user = self.hparams.transform_map.get(var_name, {'method': 'zscore'})
                method = var_config_user.get('method', 'zscore')
                params = {}
                
                current_var_train_data_slice = train_input_dask[:, i, :, :] # Shape (time, y, x)

                if method == 'zscore':
                    params['mean'] = overall_input_mean[0, i, 0, 0]
                    params['std'] = overall_input_std[0, i, 0, 0]
                elif method == 'minimax':
                    params['min_val'] = var_config_user.get('min', overall_input_min[0, i, 0, 0])
                    params['max_val'] = var_config_user.get('max', overall_input_max[0, i, 0, 0])
                elif method == 'log1p':
                    data_after_log1p = da.log1p(current_var_train_data_slice)
                    params['mean'] = da.nanmean(data_after_log1p).compute() # Mean of log1p(data)
                    params['std'] = da.nanstd(data_after_log1p).compute()   # Std of log1p(data)
                elif method == 'sqrt':
                    data_after_sqrt = da.sqrt(current_var_train_data_slice)
                    params['mean'] = da.nanmean(data_after_sqrt).compute()  # Mean of sqrt(data)
                    params['std'] = da.nanstd(data_after_sqrt).compute()    # Std of sqrt(data)
                elif method == 'pow':
                    exponent = var_config_user.get('lambda')
                    if exponent is None:
                        raise ValueError(f"'lambda' (exponent) must be provided for 'pow' method for variable '{var_name}'.")
                    params['lambda'] = exponent
                    data_after_pow = current_var_train_data_slice ** exponent
                    params['mean'] = da.nanmean(data_after_pow).compute()  # Mean of data**lambda
                    params['std'] = da.nanstd(data_after_pow).compute()    # Std of data**lambda
                else:
                    log.warning(f"Method '{method}' for input var '{var_name}' might not have specific stat calculation logic here; using raw params if any from config.")
                    params = var_config_user.get('params', {})


                input_transform_map_indexed[i] = {'method': method, 'params': params}
                log.info(f"Input var '{var_name}' (idx {i}): method='{method}', params calculated/set.")
            
            self.normalizer.set_input_statistics(input_transform_map_indexed)

            # --- Construct transform_map for outputs (similar logic) ---
            output_transform_map_indexed = {}
            for i, var_name in enumerate(self.hparams.output_vars):
                var_config_user = self.hparams.transform_map.get(var_name, {'method': 'zscore'})
                method = var_config_user.get('method', 'zscore')
                params = {}

                current_var_train_data_slice = train_output_dask[:, i, :, :] # Shape (time, y, x)

                if method == 'zscore':
                    params['mean'] = overall_output_mean[0, i, 0, 0]
                    params['std'] = overall_output_std[0, i, 0, 0]
                elif method == 'minimax':
                    params['min_val'] = var_config_user.get('min', overall_output_min[0, i, 0, 0])
                    params['max_val'] = var_config_user.get('max', overall_output_max[0, i, 0, 0])
                elif method == 'log1p':
                    data_after_log1p = da.log1p(current_var_train_data_slice)
                    params['mean'] = da.nanmean(data_after_log1p).compute()
                    params['std'] = da.nanstd(data_after_log1p).compute()
                elif method == 'sqrt':
                    data_after_sqrt = da.sqrt(current_var_train_data_slice)
                    params['mean'] = da.nanmean(data_after_sqrt).compute()
                    params['std'] = da.nanstd(data_after_sqrt).compute()
                elif method == 'pow':
                    exponent = var_config_user.get('lambda')
                    if exponent is None:
                        raise ValueError(f"'lambda' (exponent) must be provided for 'pow' method for variable '{var_name}'.")
                    params['lambda'] = exponent
                    data_after_pow = current_var_train_data_slice ** exponent
                    params['mean'] = da.nanmean(data_after_pow).compute()
                    params['std'] = da.nanstd(data_after_pow).compute()
                else:
                    log.warning(f"Method '{method}' for output var '{var_name}' might not have specific stat calculation logic here; using raw params if any from config.")
                    params = var_config_user.get('params', {})
                
                output_transform_map_indexed[i] = {'method': method, 'params': params}
                log.info(f"Output var '{var_name}' (idx {i}): method='{method}', params calculated/set.")

            self.normalizer.set_output_statistics(output_transform_map_indexed)

            # --- Normalize data using the configured Normalizer A ---
            train_input_norm_dask = self.normalizer.normalize(train_input_dask, data_type="input")
            train_output_norm_dask = self.normalizer.normalize(train_output_dask, data_type="output")
            
            if val_input_dask is not None and val_output_dask is not None:
                val_input_norm_dask = self.normalizer.normalize(val_input_dask, data_type="input")
                val_output_norm_dask = self.normalizer.normalize(val_output_dask, data_type="output")
            else: 
                val_input_norm_dask, val_output_norm_dask = None, None

            # --- Prepare Test Data ---
            full_test_input_dask, full_test_output_dask = _load_process_ssp_data(
                ds, self.hparams.test_ssp, self.hparams.input_vars, self.hparams.output_vars,
                self.hparams.member_ids, spatial_template_da,
            )
            test_slice = slice(-self.hparams.test_months, None)
            sliced_test_input_dask = full_test_input_dask[test_slice]
            sliced_test_output_raw_dask = full_test_output_dask[test_slice] # Test outputs are raw
            
            test_input_norm_dask = self.normalizer.normalize(sliced_test_input_dask, data_type="input")
            # test_output_raw_dask is kept unnormalized for evaluation

        # Create datasets (passing seq_len)
        seq_len = self.hparams.seq_len # Get seq_len from hparams
        self.train_dataset = ClimateDataset(train_input_norm_dask, train_output_norm_dask, seq_len=seq_len, output_is_normalized=True)
        
        val_len = 0
        if val_input_norm_dask is not None and val_output_norm_dask is not None:
            self.val_dataset = ClimateDataset(val_input_norm_dask, val_output_norm_dask, seq_len=seq_len, output_is_normalized=True)
            val_len = len(self.val_dataset)
        else:
            self.val_dataset = None 
            log.warning("No validation data was separated. Validation steps will be skipped if val_dataset is None.")

        self.test_dataset = ClimateDataset(test_input_norm_dask, sliced_test_output_raw_dask, seq_len=seq_len, output_is_normalized=False)
        log.info(
            f"Datasets created. Train: {len(self.train_dataset)}, Val: {val_len}, Test: {len(self.test_dataset)}"
        )

    # Common DataLoader configuration
    def _get_dataloader_kwargs(self, is_train=False):
        """Return common DataLoader configuration as a dictionary"""
        return {
            "batch_size": self.hparams.batch_size if is_train else self.hparams.eval_batch_size,
            "shuffle": is_train,  # Only shuffle training data
            "num_workers": self.hparams.num_workers,
            "persistent_workers": self.hparams.num_workers > 0,
            "pin_memory": True,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._get_dataloader_kwargs(is_train=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._get_dataloader_kwargs(is_train=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._get_dataloader_kwargs(is_train=False))

    def get_lat_weights(self):
        """
        Returns area weights for the latitude dimension as an xarray DataArray.
        The weights can be used with xarray's weighted method for proper spatial averaging.
        """
        if self._lat_weights_da is None:
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0)
                y_coords = template.y.values

                # Calculate weights based on cosine of latitude
                weights = get_lat_weights(y_coords)

                # Create DataArray with proper dimensions
                self._lat_weights_da = xr.DataArray(weights, dims=["y"], coords={"y": y_coords}, name="area_weights")

        return self._lat_weights_da

    def get_coords(self):
        """
        Returns the y and x coordinates (representing latitude and longitude).

        Returns:
            tuple: (y array, x array)
        """
        if self.lat_coords is None or self.lon_coords is None:
            # Get coordinates if they haven't been stored yet
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0, drop=True)
                self.lat_coords = template.y.values
                self.lon_coords = template.x.values

        return self.lat_coords, self.lon_coords


# --- PyTorch Lightning Module ---
class ClimateEmulationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, weight_decay: float):
        super().__init__()
        self.model = model
        # Access hyperparams via self.hparams object after saving, e.g., self.hparams.learning_rate
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.MSELoss()
        self.normalizer = None
        # Store evaluation outputs for time-mean calculation
        self.test_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        self.normalizer = self.trainer.datamodule.normalizer  # Access the normalizer from the datamodule

    def training_step(self, batch, batch_idx):
        x, y_true_norm = batch
        y_pred_norm = self(x)
        loss = self.criterion(y_pred_norm, y_true_norm)
        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true_norm = batch
        y_pred_norm = self(x)
        loss = self.criterion(y_pred_norm, y_true_norm)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0), sync_dist=True)

        # Save unnormalized outputs for decadal mean/stddev calculation in validation_epoch_end
        y_pred_norm = self.normalizer.inverse_transform_output(y_pred_norm.cpu().numpy())
        y_true_norm = self.normalizer.inverse_transform_output(y_true_norm.cpu().numpy())
        self.validation_step_outputs.append((y_pred_norm, y_true_norm))

        return loss

    def _evaluate_predictions(self, predictions, targets, is_test=False):
        """
        Helper method to evaluate predictions against targets with climate metrics.

        Args:
            predictions (np.ndarray): Prediction array with shape (time, channels, y, x)
            targets (np.ndarray): Target array with shape (time, channels, y, x)
            is_test (bool): Whether this is being called from test phase (vs validation)
        """
        phase = "test" if is_test else "val"
        log_kwargs = {"prog_bar": not is_test, "sync_dist": not is_test}

        # Get number of evaluation timesteps
        n_timesteps = predictions.shape[0]

        # Get area weights for proper spatial averaging
        area_weights = self.trainer.datamodule.get_lat_weights()

        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        time_coords = np.arange(n_timesteps)
        output_vars = self.trainer.datamodule.hparams.output_vars

        # Process each output variable
        for i, var_name in enumerate(output_vars):
            # Extract channel data
            preds_var = predictions[:, i, :, :]
            trues_var = targets[:, i, :, :]

            var_unit = "mm/day" if var_name == "pr" else "K" if var_name == "tas" else "unknown"

            # Create xarray objects for weighted calculations
            preds_xr = create_climate_data_array(
                preds_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )
            trues_xr = create_climate_data_array(
                trues_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )

            preds_xr = preds_xr.astype(np.float64)
            trues_xr = trues_xr.astype(np.float64)

            # 1. Calculate weighted month-by-month RMSE over all samples
            diff_squared = (preds_xr - trues_xr) ** 2
            overall_rmse = calculate_weighted_metric(diff_squared, area_weights, ("time", "y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/avg/monthly_rmse", float(overall_rmse), **log_kwargs)

            # 2. Calculate time-mean (i.e. decadal, 120 months average) and calculate area-weighted RMSE for time means
            pred_time_mean = preds_xr.mean(dim="time")
            true_time_mean = trues_xr.mean(dim="time")
            mean_diff_squared = (pred_time_mean - true_time_mean) ** 2
            time_mean_rmse = calculate_weighted_metric(mean_diff_squared, area_weights, ("y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/time_mean_rmse", float(time_mean_rmse), **log_kwargs)

            # 3. Calculate time-stddev (temporal variability) and calculate area-weighted MAE for time stddevs
            pred_time_std = preds_xr.std(dim="time")
            true_time_std = trues_xr.std(dim="time")
            std_abs_diff = np.abs(pred_time_std - true_time_std)
            time_std_mae = calculate_weighted_metric(std_abs_diff, area_weights, ("y", "x"), "mae")
            self.log(f"{phase}/{var_name}/time_stddev_mae", float(time_std_mae), **log_kwargs)

            
            # Generate visualizations for test phase when using wandb
            if isinstance(self.logger, WandbLogger):
                # Time mean visualization
                fig = create_comparison_plots(
                    true_time_mean,
                    pred_time_mean,
                    title_prefix=f"{var_name} Mean",
                    metric_value=time_mean_rmse,
                    metric_name="Weighted RMSE",
                )
                self.logger.experiment.log({f"img/{var_name}/time_mean": wandb.Image(fig)})
                plt.close(fig)

                # Time standard deviation visualization
                fig = create_comparison_plots(
                    true_time_std,
                    pred_time_std,
                    title_prefix=f"{var_name} Stddev",
                    metric_value=time_std_mae,
                    metric_name="Weighted MAE",
                    cmap="plasma",
                )
                self.logger.experiment.log({f"img/{var_name}/time_Stddev": wandb.Image(fig)})
                plt.close(fig)

                # Sample timesteps visualization
                if n_timesteps > 10:
                    timesteps = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108]
                    for t in timesteps:
                        true_t = trues_xr.isel(time=t)
                        pred_t = preds_xr.isel(time=t)
                        fig = create_comparison_plots(true_t, pred_t, title_prefix=f"{var_name} Timestep {t}")
                        self.logger.experiment.log({f"img/{phase}/{var_name}/month_idx_{t}": wandb.Image(fig)})
                        plt.close(fig)

    def on_validation_epoch_end(self):
        # Compute time-mean and time-stddev errors using all validation months
        if not self.validation_step_outputs:
            return

        # Stack all predictions and ground truths
        all_preds_np = np.concatenate([pred for pred, _ in self.validation_step_outputs], axis=0)
        all_trues_np = np.concatenate([true for _, true in self.validation_step_outputs], axis=0)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_np, all_trues_np, is_test=False)

        self.validation_step_outputs.clear()  # Clear the outputs list for next epoch

    def test_step(self, batch, batch_idx):
        x, y_true_denorm = batch
        y_pred_norm = self(x)
        # Denormalize the predictions for evaluation back to original scale
        y_pred_denorm = self.normalizer.inverse_transform_output(y_pred_norm.cpu().numpy())
        y_true_denorm_np = y_true_denorm.cpu().numpy()
        self.test_step_outputs.append((y_pred_denorm, y_true_denorm_np))

    def on_test_epoch_end(self):
        # Concatenate all predictions and ground truths from each test step/batch into one array
        all_preds_denorm = np.concatenate([pred for pred, true in self.test_step_outputs], axis=0)
        all_trues_denorm = np.concatenate([true for pred, true in self.test_step_outputs], axis=0)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_denorm, all_trues_denorm, is_test=True)

        # Save predictions for Kaggle submission. This is the file that should be uploaded to Kaggle.
        log.info("Saving Kaggle submission...")
        self._save_kaggle_submission(all_preds_denorm)

        self.test_step_outputs.clear()  # Clear the outputs list

    def _save_kaggle_submission(self, predictions, suffix=""):
        """
        Create a Kaggle submission file from the model predictions.

        Args:
            predictions (np.ndarray): Predicted values with shape (time, channels, y, x)
        """
        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        output_vars = self.trainer.datamodule.hparams.output_vars
        n_times = predictions.shape[0]
        time_coords = np.arange(n_times)

        # Convert predictions to Kaggle format
        submission_df = convert_predictions_to_kaggle_format(
            predictions, time_coords, lat_coords, lon_coords, output_vars
        )

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = to_absolute_path(f"submissions/kaggle_submission{suffix}_{timestamp}.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
        submission_df.to_csv(filepath, index=False)

        if wandb is not None and isinstance(self.logger, WandbLogger):
            pass
            # Optionally, uncomment the following line to save the submission to the wandb cloud
            # self.logger.experiment.log_artifact(filepath)  # Log to wandb if available

        log.info(f"Kaggle submission saved to {filepath}")

    def configure_optimizers(self):

        learning_rate = self.hparams.learning_rate
        weight_decay_value = self.hparams.get("weight_decay", 0.0)
        log.info(f"Configuring Adam optimizer with LR: {learning_rate}, Weight Decay: {weight_decay_value}")
        optimizer = optim.Adam(
            self.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay_value  # Use the value that has a default
        )
        return optimizer


# --- Main Execution with Hydra ---
@hydra.main(version_base=None, config_path="configs", config_name="main_config.yaml")
def main(cfg: DictConfig):
    # Print resolved configs
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Create data module with parameters from configs
    datamodule = ClimateEmulationDataModule(seed=cfg.seed, **cfg.data)
    model = get_model(cfg)

    # Create lightning module
    lightning_module = ClimateEmulationModule(model, learning_rate=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    # Create lightning trainer
    trainer_config = get_trainer_config(cfg, model=model)
    trainer = pl.Trainer(**trainer_config)

    if cfg.ckpt_path and isinstance(cfg.ckpt_path, str):
        cfg.ckpt_path = to_absolute_path(cfg.ckpt_path)

    # Train model
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Training finished.")

    # Test model
    # IMPORTANT: Please note that the test metrics will be bad because the test targets have been corrupted on the public Kaggle dataset.
    # The purpose of testing below is to generate the Kaggle submission file based on your model's predictions.
    trainer_config["devices"] = 1  # Make sure you test on 1 GPU only to avoid synchronization issues with DDP
    eval_trainer = pl.Trainer(**trainer_config)
    eval_trainer.test(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.use_wandb and isinstance(trainer_config.get("logger"), WandbLogger):
        wandb.finish()  # Finish the run if using wandb


if __name__ == "__main__":
    main()