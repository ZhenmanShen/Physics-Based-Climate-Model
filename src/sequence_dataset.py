# src/sequence_dataset.py
import torch
from torch.utils.data import Dataset

class ClimateSequenceDataset(Dataset):
    """
    Produces (x_seq, y) pairs where
        x_seq  : (seq_len, C_in, 48, 72)
        y      : (C_out, 48, 72)  – the last month in the window
    The inputs must already be NORMALISED; that is exactly what
    ClimateEmulationDataModule gives us.
    """
    def __init__(self,
                 inputs_norm_dask,   # time × C_in × H × W
                 outputs_dask,       # time × C_out × H × W
                 seq_len: int = 4):
        assert inputs_norm_dask.shape[0] == outputs_dask.shape[0], "time dim mismatch"
        assert seq_len >= 1, "seq_len must be positive"

        # materialise into memory once – only ~ 3 000 months × (5+2) channels
        self.X = torch.from_numpy(inputs_norm_dask.compute()).float()
        self.Y = torch.from_numpy(outputs_dask.compute()).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]          # (T, C_in, H, W)
        y     = self.Y[idx + self.seq_len - 1]            # predict last month
        return x_seq, y
