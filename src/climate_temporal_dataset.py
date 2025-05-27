import torch
from torch.utils.data import Dataset

class ClimateTemporalDataset(Dataset):
    def __init__(self, inputs_norm_dask, outputs_dask, output_is_normalized=True, time_steps=12):
        self.input_tensors = torch.from_numpy(inputs_norm_dask.compute()).float()
        self.output_tensors = torch.from_numpy(outputs_dask.compute()).float()
        self.time_steps = time_steps
        self.output_is_normalized = output_is_normalized
        self.size = self.input_tensors.shape[0] - time_steps + 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x_seq = self.input_tensors[idx:idx + self.time_steps]  # [T, C, H, W]
        y = self.output_tensors[idx + self.time_steps - 1]     # 마지막 시점의 target 예측

        return x_seq, y
