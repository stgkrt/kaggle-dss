# pytorchでデーターローダーを作成する


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DSSDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = series_df
        self.mode = mode
        self.data_length = 17280

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(self, series_: np.ndarray) -> np.ndarray:
        if len(series_) < self.data_length:
            padding_length = self.data_length - len(series_)
            # TODO:計測できていないときのデータが0の場合は変更する必要がある
            padding_data = np.zeros(padding_length)
            series_data = np.concatenate([series_, padding_data])
        return series_data

    def _get_input_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        anglez = series_df_["anglez"].values
        enmo = series_df_["enmo"].values
        anglez = self._padding_data_to_same_length(anglez)
        enmo = self._padding_data_to_same_length(enmo)
        input_data = np.concatenate(
            [np.expand_dims(anglez, axis=0), np.expand_dims(enmo, axis=0)]
        )  # [channel, data_length]
        input_data = torch.tensor(input_data, dtype=torch.float32)
        return input_data

    def _get_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_["event"].values[0]
        target = torch.tensor(target, dtype=torch.long)
        return target

    def __getitem__(self, idx):
        data_key = self.key_df.iloc[idx]
        series_data = self.series_df[self.series_df["series_data_key"] == data_key]
        input = self._get_input_data(series_data)
        if self.mode == "test":
            return input
        else:
            target = self._get_target_data(series_data)
            return input, target


if __name__ == "__main__":
    series_df = pd.read_parquet()
