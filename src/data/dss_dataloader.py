import numpy as np
import pandas as pd  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DSSDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.data_length = 17280

        maxpool_kernel_size = 11  # 奇数じゃないと同じ長さで出力できない
        maxpool_stride = 1
        # 入力サイズと出力サイズが一致するようにpaddingを調整
        maxpool_padding = int((maxpool_kernel_size - maxpool_stride) / 2)
        self.max_pool = nn.MaxPool1d(
            kernel_size=maxpool_kernel_size,
            stride=maxpool_stride,
            padding=maxpool_padding,
        )
        ave_kernel_size = 11
        ave_stride = 1
        ave_padding = int((ave_kernel_size - ave_stride) / 2)
        self.average_pool = nn.AvgPool1d(
            kernel_size=ave_kernel_size, stride=ave_stride, padding=ave_padding
        )

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(self, series_: np.ndarray) -> np.ndarray:
        if len(series_) < self.data_length:
            padding_length = self.data_length - len(series_)
            # TODO:計測できていないときのデータが0の場合は変更する必要がある
            padding_data = np.zeros(padding_length)
            series_data = np.concatenate([series_, padding_data])
        elif len(series_) > self.data_length:
            # print(f"[warning] data length is over.")
            series_data = series_[: self.data_length]
        else:
            series_data = series_
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
        target = series_df_["event"].values
        target = self._padding_data_to_same_length(target)
        target = np.expand_dims(target, axis=0)  # [channel=1, data_length]
        target = torch.tensor(target, dtype=torch.long)
        return target

    def _soft_label(self, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        target = target.unsqueeze(0)  # [channel=1, data_length]
        target = self.max_pool(target)
        target = self.average_pool(target)
        return target

    def _get_event_target_data(self, series_df_: pd.DataFrame) -> torch.Tensor:
        event_onset = series_df_["event_onset"].values
        event_onset = self._padding_data_to_same_length(event_onset)
        event_onset = torch.tensor(event_onset, dtype=torch.float32)
        event_onset = self._soft_label(event_onset)

        event_wakeup = series_df_["event_wakeup"].values
        event_wakeup = self._padding_data_to_same_length(event_wakeup)
        event_wakeup = torch.tensor(event_wakeup, dtype=torch.float32)
        event_wakeup = self._soft_label(event_wakeup)
        event_target = torch.cat([event_onset, event_wakeup], dim=0)
        return event_target

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        # if len(series_data) > self.data_length:
        #     print(f"[warning] data length is over. series_date_key: {data_key}")
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        else:
            target = self._get_target_data(series_data)
            event_target = self._get_event_target_data(series_data)
            return input, target, event_target, input_info_dict


def get_loader(CFG, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"):
    dataset = DSSDataset(key_df, series_df)
    if mode == "train":
        shuffle = True
    else:
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    key_df = pd.read_csv("/kaggle/input/datakey_unique_non_null.csv")
    series_df = pd.read_parquet("/kaggle/input/processed_train_withkey_nonull.parquet")

    dataset = DSSDataset(key_df, series_df)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    for input, target, event_target, input_info in dataloader:
        print(input.shape)
        print(target.shape)
        print(event_target.shape)
        print(input_info)
        break
