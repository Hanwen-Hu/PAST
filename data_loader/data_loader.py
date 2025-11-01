"""Traffic Data Loader Module"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TrafficDataConfig:
    """Configuration for TrafficData"""
    dataset: str
    seq_len: int
    mode: str
    miss_rate: float
    miss_len: int
    miss_span: int
    device: torch.device


class TrafficData(Dataset):
    """Traffic Data Loader"""
    def __init__(self, config: TrafficDataConfig) -> None:
        self.seq_len = config.seq_len
        # Read Adj Matrix, Data and Mask
        adj = np.load(f'dataset/{config.dataset}/adj.npy')
        data = pd.read_csv(f'dataset/{config.dataset}/15min_values.csv', index_col='Timestamp')
        mask_file = f'mask_{(config.miss_rate * 10):.0f}_{config.miss_len}_{config.miss_span}.csv'
        mask = pd.read_csv(f'dataset/{config.dataset}/{mask_file}', index_col='Timestamp')
        timestamp = self._get_timestamp(data.index)
        # Transfer to Tensor
        self.adj = torch.from_numpy(adj).float().to(config.device)
        self._m = torch.from_numpy(mask.values).bool().to(config.device).unsqueeze(-1)
        self._t = torch.from_numpy(timestamp).long().to(config.device).unsqueeze(1)  # len, 1, 4
        data = torch.from_numpy(data.values).float().to(config.device).unsqueeze(-1)
        # Preprocess
        self._x, self._y = self._standardize(data)
        self._split(config.mode)

    @staticmethod
    def _get_timestamp(time_column: pd.Index) -> np.ndarray:
        timestamps = pd.to_datetime(time_column.tolist())
        days = timestamps.day.values - 1
        weekdays = timestamps.weekday.values
        hours = timestamps.hour.values
        minutes = timestamps.minute.values // 15
        return np.stack([days, weekdays, hours, minutes], axis=1, dtype=np.int64)

    def _standardize(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = y.clone()
        x[~self._m] = torch.nan
        x_np = x.cpu().numpy()
        mean = np.nanmean(x_np).item()
        std = np.nanstd(x_np).item()
        return (x - mean) / std, (y - mean) / std

    def _split(self, mode: str) -> None:
        assert mode in ['offline', 'online']
        length = self._y.shape[0]
        slice_point = int(0.8 * length)
        idx_range = slice(None, slice_point) if mode == 'offline' else slice(slice_point, None)
        self._x = self._x[idx_range]
        self._y = self._y[idx_range]
        self._m = self._m[idx_range]
        self._t = self._t[idx_range]

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x, y, m, t
        idx_range = slice(item, item + self.seq_len)
        return self._x[idx_range], self._y[idx_range], self._m[idx_range], self._t[idx_range]

    def __len__(self) -> int:
        return self._y.shape[0] - self.seq_len + 1
