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

def create_time_encoding(month_idx, total_months, H=48, W=72, device="cpu"):
    import math
    angle = 2 * math.pi * (month_idx % 12) / 12.0
    sin_val = math.sin(angle)
    cos_val = math.cos(angle)
    trend_val = month_idx / total_months
    sin_channel = torch.full((1, H, W), sin_val, device=device)
    cos_channel = torch.full((1, H, W), cos_val, device=device)
    trend_channel = torch.full((1, H, W), trend_val, device=device)
    return torch.cat([sin_channel, cos_channel, trend_channel], dim=0)


# Dataset to precompute all tensors during initialization
class ClimateDataset(Dataset):
    def __init__(self, inputs_norm_dask, outputs_dask, months, total_months, output_is_normalized=True):
        self.size = inputs_norm_dask.shape[0]
        log.info(f"Creating dataset: {self.size} samples, input shape: {inputs_norm_dask.shape}, normalized output: {output_is_normalized}")
        self.input_np = inputs_norm_dask.compute()
        self.output_np = outputs_dask.compute()
        self.months = months
        self.total_months = total_months
        self.output_is_normalized = output_is_normalized

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.input_np[idx]).float()
        output_tensor = torch.from_numpy(self.output_np[idx]).float()
        month = self.months[idx]
        time_tensor = create_time_encoding(month_idx=month, total_months=self.total_months, H=input_tensor.shape[1], W=input_tensor.shape[2], device=input_tensor.device)
        input_tensor = torch.cat([input_tensor, time_tensor], dim=0)
        return input_tensor, output_tensor


def _load_process_ssp_data(
    ds,
    ssp: str,
    input_variables: list[str],
    output_variables: list[str],
    member_ids: list[int],
    spatial_template: xr.DataArray,
):
    """
    Returns inputs & outputs for **all requested ensemble members**,
    concatenated on the *time* axis (time × channels × y × x).
    """
    input_members, output_members = [], []

    for m in member_ids:
        ssp_input_dasks, ssp_output_dasks = [], []

        # ---------- INPUTS ----------
        for var in input_variables:
            da_var = ds[var].sel(ssp=ssp)
            if "latitude" in da_var.dims:   # rename spatial dims once
                da_var = da_var.rename({"latitude": "y", "longitude": "x"})
            if "member_id" in da_var.dims:
                da_var = da_var.sel(member_id=m)

            if set(da_var.dims) == {"time"}:        # global -> broadcast
                da_var = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
            elif set(da_var.dims) != {"time", "y", "x"}:
                raise ValueError(f"Unexpected dims {da_var.dims} for {var}")

            ssp_input_dasks.append(da_var.data)

        # time × C_in × y × x  (for one member)
        input_members.append(da.stack(ssp_input_dasks, axis=1))

        # ---------- OUTPUTS ----------
        for var in output_variables:
            da_out = ds[var].sel(ssp=ssp, member_id=m)
            if "latitude" in da_out.dims:
                da_out = da_out.rename({"latitude": "y", "longitude": "x"})
            ssp_output_dasks.append(da_out.data)

        # time × C_out × y × x  (for one member)
        output_members.append(da.stack(ssp_output_dasks, axis=1))

    # concat the *members* along time, keeping chronology per member
    stacked_input = da.concatenate(input_members, axis=0)
    stacked_output = da.concatenate(output_members, axis=0)
    return stacked_input, stacked_output


class ClimateEmulationDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        input_vars: list,
        output_vars: list,
        train_ssps: list,
        test_ssp: str,
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

        if eval_batch_size is None:
            self.hparams.eval_batch_size = batch_size

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

            for ssp in self.hparams.train_ssps:
                ssp_input_dask, ssp_output_dask = _load_process_ssp_data(
                    ds,
                    ssp,
                    self.hparams.input_vars,
                    self.hparams.output_vars,
                    self.hparams.member_ids,
                    spatial_template_da,
                )

                if ssp == val_ssp:
                    val_input_dask = ssp_input_dask[-val_months:]
                    val_output_dask = ssp_output_dask[-val_months:]
                    train_inputs_dask_list.append(ssp_input_dask[:-val_months])
                    train_outputs_dask_list.append(ssp_output_dask[:-val_months])
                else:
                    train_inputs_dask_list.append(ssp_input_dask)
                    train_outputs_dask_list.append(ssp_output_dask)

            train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
            train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)

            input_mean = da.nanmean(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            input_std = da.nanstd(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_mean = da.nanmean(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_std = da.nanstd(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()

            self.normalizer.set_input_statistics(mean=input_mean, std=input_std)
            self.normalizer.set_output_statistics(mean=output_mean, std=output_std)

            train_input_norm_dask = self.normalizer.normalize(train_input_dask, data_type="input")
            train_output_norm_dask = self.normalizer.normalize(train_output_dask, data_type="output")
            val_input_norm_dask = self.normalizer.normalize(val_input_dask, data_type="input")
            val_output_norm_dask = self.normalizer.normalize(val_output_dask, data_type="output")

            full_test_input_dask, full_test_output_dask = _load_process_ssp_data(
                ds,
                self.hparams.test_ssp,
                self.hparams.input_vars,
                self.hparams.output_vars,
                self.hparams.member_ids,
                spatial_template_da,
            )

            test_slice = slice(-self.hparams.test_months, None)
            sliced_test_input_dask = full_test_input_dask[test_slice]
            sliced_test_output_raw_dask = full_test_output_dask[test_slice]
            test_input_norm_dask = self.normalizer.normalize(sliced_test_input_dask, data_type="input")
            test_output_raw_dask = sliced_test_output_raw_dask

            train_months = list(range(train_input_norm_dask.shape[0]))
            val_months_list = list(range(val_input_norm_dask.shape[0]))
            test_months_list = list(range(test_input_norm_dask.shape[0]))
            total_months = max(train_months + val_months_list + test_months_list) + 1

        self.train_dataset = ClimateDataset(train_input_norm_dask, train_output_norm_dask, train_months, total_months, output_is_normalized=True)
        self.val_dataset = ClimateDataset(val_input_norm_dask, val_output_norm_dask, val_months_list, total_months, output_is_normalized=True)
        self.test_dataset = ClimateDataset(test_input_norm_dask, test_output_raw_dask, test_months_list, total_months, output_is_normalized=False)

        log.info(
            f"Datasets created. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)} (last months of {val_ssp}), Test: {len(self.test_dataset)}"
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
    def __init__(self, model: nn.Module, learning_rate: float):
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
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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
    lightning_module = ClimateEmulationModule(model, learning_rate=cfg.training.lr)

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