import numpy as np
import pandas as pd  # type: ignore
import torch
import torch.nn as nn
from scipy import signal
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DSSDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
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

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        else:
            target = self._get_target_data(series_data)
            return input, target, input_info_dict


class DSSDenseDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.mean_std_rollnum_list = [36, 60]
        self.data_length = 17280

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(self, series_: np.ndarray) -> np.ndarray:
        data_length = self.data_length
        if series_.shape[-1] < data_length:  # [ch, data_len] or [data_len,]
            padding_length = data_length - series_.shape[-1]
            padding_data = np.zeros(padding_length)
            if series_.ndim != 1:
                padding_data = np.expand_dims(padding_data, axis=0)
                padding_data = np.tile(padding_data, (series_.shape[0], 1))
            series_data = np.concatenate([series_, padding_data], axis=-1)
        elif series_.shape[-1] > data_length:
            if series_.ndim == 1:
                series_data = series_[:data_length]
            else:
                series_data = series_[:, :data_length]
        else:
            series_data = series_
        return series_data

    def _get_input_data(self, series_df_: pd.DataFrame) -> torch.Tensor:
        input_data = series_df_[
            [
                "anglez",
                "enmo",
            ]
        ].values.T
        input_data[0] = input_data[0] / 90.0
        input_data[1] = input_data[1] / 5.0
        for roll_num in self.mean_std_rollnum_list:
            anglez_mean = series_df_[f"anglez_mean_{roll_num}"].values / 90.0
            anglez_std = series_df_[f"anglez_std_{roll_num}"].values
            input_data = np.concatenate(
                [
                    input_data,
                    np.expand_dims(anglez_mean, axis=0),
                    np.expand_dims(anglez_std, axis=0),
                ]
            )
        input_data = self._padding_data_to_same_length(input_data)
        input_data = torch.tensor(input_data, dtype=torch.float32)
        return input_data

    def _get_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_["event"].values
        target = self._padding_data_to_same_length(target)
        target = np.expand_dims(target, axis=0)  # [channel=1, data_length]
        target = torch.tensor(target, dtype=torch.long)
        return target

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        else:
            target = self._get_target_data(series_data)
            return input, target, input_info_dict


class DSSDense2chDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.mean_std_rollnum_list = [36, 60]
        self.data_length = 17280

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(self, series_: np.ndarray) -> np.ndarray:
        data_length = self.data_length
        if series_.shape[-1] < data_length:  # [ch, data_len] or [data_len,]
            padding_length = data_length - series_.shape[-1]
            padding_data = np.zeros(padding_length)
            if series_.ndim != 1:
                padding_data = np.expand_dims(padding_data, axis=0)
                padding_data = np.tile(padding_data, (series_.shape[0], 1))
            series_data = np.concatenate([series_, padding_data], axis=-1)
        elif series_.shape[-1] > data_length:
            if series_.ndim == 1:
                series_data = series_[:data_length]
            else:
                series_data = series_[:, :data_length]
        else:
            series_data = series_
        return series_data

    def _get_input_data(self, series_df_: pd.DataFrame) -> torch.Tensor:
        input_data = series_df_[
            [
                "anglez",
                "enmo",
            ]
        ].values.T
        input_data[0] = input_data[0] / 90.0
        input_data[1] = input_data[1] / 5.0
        for roll_num in self.mean_std_rollnum_list:
            anglez_mean = series_df_[f"anglez_mean_{roll_num}"].values / 90.0
            anglez_std = series_df_[f"anglez_std_{roll_num}"].values
            input_data = np.concatenate(
                [
                    input_data,
                    np.expand_dims(anglez_mean, axis=0),
                    np.expand_dims(anglez_std, axis=0),
                ]
            )
        input_data = self._padding_data_to_same_length(input_data)
        input_data = torch.tensor(input_data, dtype=torch.float32)
        return input_data

    def _get_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_["event"].values
        target_not_detect = 1.0 - (series_df_["event"] == -1).values
        target = np.concatenate(
            [
                np.expand_dims(target, axis=0),
                np.expand_dims(target_not_detect, axis=0),
            ],
            axis=0,
        )
        target = self._padding_data_to_same_length(target)
        target = torch.tensor(target, dtype=torch.long)
        return target

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        else:
            target = self._get_target_data(series_data)
            return input, target, input_info_dict


class DSSDownSampleDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.data_length = 1440  # 17280/12

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
        anglez = series_df_["anglez"].values / 90.0
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

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        else:
            target = self._get_target_data(series_data)
            return input, target, input_info_dict


class DSSTargetDownsampleDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.mean_std_rollnum_list = [36, 60]
        self.input_data_length = 17280
        self.target_data_length = 1440  # 17280/12

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(
        self, series_: np.ndarray, input_type="input"
    ) -> np.ndarray:
        if input_type == "input":
            data_length = self.input_data_length
        elif input_type == "target":
            data_length = self.target_data_length
        else:
            raise ValueError("input_type must be input or target.")
        if series_.shape[-1] < data_length:  # [ch, data_len] or [data_len,]
            padding_length = data_length - series_.shape[-1]
            padding_data = np.zeros(padding_length)
            if series_.ndim != 1:
                padding_data = np.expand_dims(padding_data, axis=0)
                padding_data = np.tile(padding_data, (series_.shape[0], 1))
            series_data = np.concatenate([series_, padding_data], axis=-1)
        elif len(series_) > data_length:
            # print(f"[warning] data length is over.")
            series_data = series_[:data_length]
        else:
            series_data = series_
        return series_data

    def _to_coord(self, x: np.ndarray, max_: int) -> np.ndarray:
        rad = 2 * np.pi * (x % max_) / max_
        x_sin = np.sin(rad)
        x_cos = np.cos(rad)
        x_sincos = np.concatenate(
            [np.expand_dims(x_sin, axis=0), np.expand_dims(x_cos, axis=0)], axis=0
        )
        return x_sincos

    def _get_input_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        input_data = series_df_[
            [
                "anglez",
                "enmo",
            ]
        ].values.T
        input_data[0] = input_data[0] / 90.0
        input_data[1] = input_data[1] / 5.0
        for roll_num in self.mean_std_rollnum_list:
            anglez_mean = series_df_[f"anglez_mean_{roll_num}"].values / 90.0
            # enmo_mean = series_df_[f"enmo_mean_{roll_num}"].values / 5.0
            anglez_std = series_df_[f"anglez_std_{roll_num}"].values
            # enmo_std = series_df_[f"enmo_std_{roll_num}"].values
            input_data = np.concatenate(
                [
                    input_data,
                    np.expand_dims(anglez_mean, axis=0),
                    np.expand_dims(anglez_std, axis=0),
                    # np.expand_dims(enmo_mean, axis=0),
                    # np.expand_dims(enmo_std, axis=0),
                ]
            )
        # hour = series_df_["hour"].values
        # hour_sincos = self._to_coord(hour, max_=24)
        # input_data = np.concatenate(
        #     [
        #         input_data,
        #         hour_sincos,
        #     ]
        # )
        input_data = self._padding_data_to_same_length(input_data, input_type="input")
        input_data = torch.tensor(input_data, dtype=torch.float32)
        return input_data

    def _get_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_[series_df_["second"] == 0]["event"].values
        target = self._padding_data_to_same_length(target, input_type="target")
        target = np.expand_dims(target, axis=0)  # [channel=1, data_length]
        target = torch.tensor(target, dtype=torch.long)
        return target

    def _get_flip_data(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        if np.random.rand() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[-1])
            target_tensor = torch.flip(target_tensor, dims=[-1])
        return input_tensor, target_tensor

    def _get_roll_data(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        if np.random.rand() > 0.5:
            shit_num = np.random.randint(10, 17280 // 3)
            input_tensor = torch.roll(input_tensor, shifts=shit_num, dims=-1)
            target_tensor = torch.roll(target_tensor, shifts=shit_num // 12, dims=-1)
        return input_tensor, target_tensor

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        elif self.mode == "train":
            target = self._get_target_data(series_data)
            input, target = self._get_flip_data(input, target)
            return input, target, input_info_dict
        else:
            target = self._get_target_data(series_data)
            return input, target, input_info_dict


# 微分を追加DTDT
class DSSTargetDownsampleDTDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.mean_std_rollnum_list = [36, 60]
        self.input_data_length = 17280
        self.target_data_length = 1440  # 17280/12

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(
        self, series_: np.ndarray, input_type="input"
    ) -> np.ndarray:
        if input_type == "input":
            data_length = self.input_data_length
        elif input_type == "target":
            data_length = self.target_data_length
        else:
            raise ValueError("input_type must be input or target.")
        if series_.shape[-1] < data_length:  # [ch, data_len] or [data_len,]
            padding_length = data_length - series_.shape[-1]
            padding_data = np.zeros(padding_length)
            if series_.ndim != 1:
                padding_data = np.expand_dims(padding_data, axis=0)
                padding_data = np.tile(padding_data, (series_.shape[0], 1))
            series_data = np.concatenate([series_, padding_data], axis=-1)
        elif len(series_) > data_length:
            # print(f"[warning] data length is over.")
            series_data = series_[:data_length]
        else:
            series_data = series_
        return series_data

    def _to_coord(self, x: np.ndarray, max_: int) -> np.ndarray:
        rad = 2 * np.pi * (x % max_) / max_
        x_sin = np.sin(rad)
        x_cos = np.cos(rad)
        x_sincos = np.concatenate(
            [np.expand_dims(x_sin, axis=0), np.expand_dims(x_cos, axis=0)], axis=0
        )
        return x_sincos

    def _get_input_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        input_data = series_df_[
            [
                "anglez",
                "enmo",
            ]
        ].values.T
        input_data[0] = input_data[0] / 90.0
        input_data[1] = input_data[1] / 5.0
        smooth_input = signal.savgol_filter(input_data[0], 60, 2, deriv=1, axis=0)
        # なめらかにした
        input_onederiv = (
            signal.savgol_filter(input_data[0], 12, 2, deriv=1, axis=0) * 5
        )  # 1次微分

        input_data = np.concatenate(
            [
                input_data,
                np.expand_dims(smooth_input, axis=0),
                np.expand_dims(input_onederiv, axis=0),
            ]
        )
        for roll_num in self.mean_std_rollnum_list:
            anglez_mean = series_df_[f"anglez_mean_{roll_num}"].values / 90.0
            anglez_std = series_df_[f"anglez_std_{roll_num}"].values
            input_data = np.concatenate(
                [
                    input_data,
                    np.expand_dims(anglez_mean, axis=0),
                    np.expand_dims(anglez_std, axis=0),
                ]
            )

        input_data = self._padding_data_to_same_length(input_data, input_type="input")
        input_data = torch.tensor(input_data, dtype=torch.float32)
        return input_data

    def _get_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_[series_df_["second"] == 0]["event"].values
        target = self._padding_data_to_same_length(target, input_type="target")
        target = np.expand_dims(target, axis=0)  # [channel=1, data_length]
        target = torch.tensor(target, dtype=torch.long)
        return target

    def _get_flip_data(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        if np.random.rand() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[-1])
            target_tensor = torch.flip(target_tensor, dims=[-1])
        return input_tensor, target_tensor

    def _get_roll_data(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        if np.random.rand() > 0.5:
            shit_num = np.random.randint(10, 17280 // 3)
            input_tensor = torch.roll(input_tensor, shifts=shit_num, dims=-1)
            target_tensor = torch.roll(target_tensor, shifts=shit_num // 12, dims=-1)
        return input_tensor, target_tensor

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        elif self.mode == "train":
            target = self._get_target_data(series_data)
            input, target = self._get_flip_data(input, target)
            return input, target, input_info_dict
        else:
            target = self._get_target_data(series_data)
            return input, target, input_info_dict


class DSSTargetDownsampleEventDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.mean_std_rollnum_list = [36, 60]
        self.input_data_length = 17280
        self.target_data_length = 1440  # 17280/12
        max_kernel_size = 31
        self.max_pool = nn.MaxPool1d(
            kernel_size=max_kernel_size,
            stride=1,
            padding=int((max_kernel_size - 1) / 2),
        )
        ave_kernel_size = 31
        self.average_pool = nn.AvgPool1d(
            kernel_size=ave_kernel_size,
            stride=1,
            padding=int((ave_kernel_size - 1) / 2),
        )

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(
        self, series_: np.ndarray, input_type="input"
    ) -> np.ndarray:
        if input_type == "input":
            data_length = self.input_data_length
        elif input_type == "target":
            data_length = self.target_data_length
        else:
            raise ValueError("input_type must be input or target.")
        if series_.shape[-1] < data_length:  # [ch, data_len] or [data_len,]
            padding_length = data_length - series_.shape[-1]
            padding_data = np.zeros(padding_length)
            if series_.ndim != 1:
                padding_data = np.expand_dims(padding_data, axis=0)
                padding_data = np.tile(padding_data, (series_.shape[0], 1))
            series_data = np.concatenate([series_, padding_data], axis=-1)
        elif len(series_) > data_length:
            # print(f"[warning] data length is over.")
            series_data = series_[:data_length]
        else:
            series_data = series_
        return series_data

    def _get_input_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        input_data = series_df_[
            [
                "anglez",
                "enmo",
            ]
        ].values.T
        input_data[0] = input_data[0] / 90.0
        input_data[1] = input_data[1] / 5.0
        for roll_num in self.mean_std_rollnum_list:
            anglez_mean = series_df_[f"anglez_mean_{roll_num}"].values / 90.0
            enmo_mean = series_df_[f"enmo_mean_{roll_num}"].values / 5.0
            anglez_std = series_df_[f"anglez_std_{roll_num}"].values
            enmo_std = series_df_[f"enmo_std_{roll_num}"].values
            input_data = np.concatenate(
                [
                    input_data,
                    np.expand_dims(anglez_mean, axis=0),
                    np.expand_dims(anglez_std, axis=0),
                    np.expand_dims(enmo_mean, axis=0),
                    np.expand_dims(enmo_std, axis=0),
                ]
            )
        input_data = self._padding_data_to_same_length(input_data, input_type="input")
        input_data = torch.tensor(input_data, dtype=torch.float32)
        return input_data

    def _soft_label(self, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        target = target.unsqueeze(0)  # [channel=1, data_length]
        positive_target = (target == 1).float()
        target = self.max_pool(target)
        target = self.average_pool(target)
        # target = self.average_pool(target)
        target = torch.clip(positive_target + target, 0, 1)
        return target

    def _get_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        event_onset = series_df_[series_df_["second"] == 0]["event_onset"].values
        event_onset = self._padding_data_to_same_length(
            event_onset, input_type="target"
        )
        event_onset = torch.tensor(event_onset, dtype=torch.float32)
        event_onset = event_onset.unsqueeze(0)  # softlabelingしないとき
        # event_onset = self._soft_label(event_onset)
        event_wakeup = series_df_[series_df_["second"] == 0]["event_wakeup"].values
        event_wakeup = self._padding_data_to_same_length(
            event_wakeup, input_type="target"
        )
        event_wakeup = torch.tensor(event_wakeup, dtype=torch.float32)
        event_wakeup = event_wakeup.unsqueeze(0)
        # event_wakeup = self._soft_label(event_wakeup)
        target = torch.cat([event_onset, event_wakeup], dim=0)
        return target

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        else:
            target = self._get_target_data(series_data)
            return input, target, input_info_dict


class DSSTargetDownsample3chDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.mean_std_rollnum_list = [36, 60]
        self.input_data_length = 17280
        self.target_data_length = 1440  # 17280/12

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(
        self, series_: np.ndarray, input_type="input"
    ) -> np.ndarray:
        if input_type == "input":
            data_length = self.input_data_length
        elif input_type == "target":
            data_length = self.target_data_length
        else:
            raise ValueError("input_type must be input or target.")
        if series_.shape[-1] < data_length:  # [ch, data_len] or [data_len,]
            padding_length = data_length - series_.shape[-1]
            padding_data = np.zeros(padding_length)
            if series_.ndim != 1:
                padding_data = np.expand_dims(padding_data, axis=0)
                padding_data = np.tile(padding_data, (series_.shape[0], 1))
            series_data = np.concatenate([series_, padding_data], axis=-1)
        elif len(series_) > data_length:
            # print(f"[warning] data length is over.")
            series_data = series_[:data_length]
        else:
            series_data = series_
        return series_data

    def _get_input_data(self, series_df_: pd.DataFrame) -> torch.Tensor:
        input_data = series_df_[
            [
                "anglez",
                "enmo",
            ]
        ].values.T
        input_data[0] = input_data[0] / 90.0
        input_data[1] = input_data[1] / 5.0
        for roll_num in self.mean_std_rollnum_list:
            anglez_mean = series_df_[f"anglez_mean_{roll_num}"].values / 90.0
            anglez_std = series_df_[f"anglez_std_{roll_num}"].values
            input_data = np.concatenate(
                [
                    input_data,
                    np.expand_dims(anglez_mean, axis=0),
                    np.expand_dims(anglez_std, axis=0),
                ]
            )
        input_data = self._padding_data_to_same_length(input_data, input_type="input")
        input_data = torch.tensor(input_data, dtype=torch.float32)
        return input_data

    def _get_target_data(self, series_df_: pd.DataFrame) -> torch.Tensor:
        event_onset = series_df_[series_df_["second"] == 0]["event_onset"].values
        event_wakeup = series_df_[series_df_["second"] == 0]["event_wakeup"].values
        event = series_df_[series_df_["second"] == 0]["event"].values
        target = np.concatenate(
            [
                event_onset.reshape(1, -1),
                event_wakeup.reshape(1, -1),
                event.reshape(1, -1),
            ],
            axis=0,
        )
        target = self._padding_data_to_same_length(target, input_type="target")
        target = torch.tensor(target, dtype=torch.long)
        return target

    def _get_flip_data(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        if np.random.rand() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[-1])
            target_tensor = torch.flip(target_tensor, dims=[-1])
        return input_tensor, target_tensor

    def _get_roll_data(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        if np.random.rand() > 0.5:
            shit_num = np.random.randint(1, 17280 // 3)
            input_tensor = torch.roll(input_tensor, shifts=shit_num, dims=-1)
            target_tensor = torch.roll(target_tensor, shifts=shit_num // 12, dims=-1)
        return input_tensor, target_tensor

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        elif self.mode == "train":
            target = self._get_target_data(series_data)
            input, target = self._get_flip_data(input, target)
            input, target = self._get_roll_data(input, target)
            return input, target, input_info_dict
        else:
            target = self._get_target_data(series_data)
            return input, target, input_info_dict


class DSSMeanStdsDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        if mode == "train" or mode == "valid":
            self.use_col = [
                "series_date_key",
                "step",
                "event",
                "anglez",
                "enmo",
            ]
        else:
            self.use_col = [
                "series_date_key",
                "step",
                "anglez",
                "enmo",
            ]
        self.mean_std_rollnum_list = [15, 30, 45]
        self.key_df = key_df["series_date_key"].values
        self.series_df = series_df[self.use_col]
        for col in self.series_df.columns:
            if self.series_df[col].dtype == np.float64:
                self.series_df[col] = self.series_df[col].astype(np.float32)
            elif self.series_df[col].dtype == np.int64:
                self.series_df[col] = self.series_df[col].astype(np.int32)
        self.mode = mode
        self.data_length = 17280

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(self, series_: np.ndarray) -> np.ndarray:
        # series_.shape = [channel, data_length]
        if series_.shape[-1] < self.data_length:
            padding_length = self.data_length - series_.shape[-1]
            # TODO:計測できていないときのデータが0の場合は変更する必要がある
            pad_shape = [(0, 0), (0, padding_length)]
            series_data = np.pad(series_, pad_shape, "edge")
        elif series_.shape[-1] > self.data_length:
            # print(f"[warning] data length is over.")
            series_data = series_[:, : self.data_length]
        else:
            series_data = series_
        return series_data

    def _get_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_["event"].values
        target = target.reshape(1, -1)  # [channel=1, data_length]
        target = self._padding_data_to_same_length(target)
        target = torch.tensor(target, dtype=torch.long)
        return target

    def _get_input_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        input_data = series_df_[
            [
                "anglez",
                "enmo",
            ]
        ].values.T
        input_data[0] = input_data[0] / 90.0
        input_data[1] = input_data[1] / 5.0
        for roll_num in self.mean_std_rollnum_list:
            anglez_mean = (
                series_df_["anglez"]
                .rolling(roll_num, center=True)
                .mean()
                .fillna(0)
                .values
            )
            anglez_std = (
                series_df_["anglez"]
                .rolling(roll_num, center=True)
                .std()
                .fillna(0)
                .values
            )
            enmo_mean = (
                series_df_["enmo"]
                .rolling(roll_num, center=True)
                .mean()
                .fillna(0)
                .values
            )
            enmo_std = (
                series_df_["enmo"].rolling(roll_num, center=True).std().fillna(0).values
            )
            input_data = np.concatenate(
                [
                    input_data,
                    np.expand_dims(anglez_mean, axis=0) / 90.0,
                    np.expand_dims(anglez_std, axis=0) / 5.0,
                    np.expand_dims(enmo_mean, axis=0) / 90.0,
                    np.expand_dims(enmo_std, axis=0) / 5.0,
                ]
            )
        input_data = self._padding_data_to_same_length(input_data)
        input_data = torch.tensor(input_data, dtype=torch.float16)
        return input_data

    # def _normalize_input_data(self, input_data: pd.DataFrame) -> np.ndarray:
    #     # 各ch方向で別々に正規化
    #     for ch in range(input_data.shape[0]):
    #         if input_data[ch].max() == input_data[ch].min():
    #             continue
    #         input_data[ch] = (input_data[ch] - input_data[ch].min()) / (
    #             input_data[ch].max() - input_data[ch].min()
    #         )
    #     input_data = torch.tensor(input_data, dtype=torch.float16)
    #     return input_data

    def __getitem__(self, idx):
        data_key = self.key_df[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        # if len(series_data) > self.data_length:
        #     print(f"[warning] data length is over. series_date_key: {data_key}")
        input = self._get_input_data(series_data)
        # input = self._normalize_input_data(input)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].values[0].astype(np.int32),
            "end_step": series_data["step"].values[-1].astype(np.int32),
        }
        if self.mode == "train" or self.mode == "valid":
            target = self._get_target_data(series_data)
            return input, target, input_info_dict
        else:
            return input, input_info_dict


class DSSAddRolldiffDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        if mode == "train" or mode == "valid":
            self.use_col = [
                "series_date_key",
                "step",
                "event",
                "anglez",
                "enmo",
                "anglez_absdiff_ave",
                "enmo_absdiff_ave",
                # "anglez_ave",
                # "enmo_ave",
                # "anglez_std",
                # "enmo_std",
            ]
        else:
            self.use_col = [
                "series_date_key",
                "step",
                "anglez",
                "enmo",
                "anglez_absdiff_ave",
                "enmo_absdiff_ave",
                # "anglez_ave",
                # "enmo_ave",
                # "anglez_std",
                # "enmo_std",
            ]

        self.key_df = key_df["series_date_key"].values
        self.series_df = series_df[self.use_col]
        self.mode = mode
        self.data_length = 17280

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(self, series_: np.ndarray) -> np.ndarray:
        # series_.shape = [channel, data_length]
        if series_.shape[-1] < self.data_length:
            padding_length = self.data_length - series_.shape[-1]
            # TODO:計測できていないときのデータが0の場合は変更する必要がある
            pad_shape = [(0, 0), (0, padding_length)]
            series_data = np.pad(series_, pad_shape, "edge")
        elif series_.shape[-1] > self.data_length:
            # print(f"[warning] data length is over.")
            series_data = series_[:, : self.data_length]
        else:
            series_data = series_
        return series_data

    def _get_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_["event"].values
        target = target.reshape(1, -1)  # [channel=1, data_length]
        target = self._padding_data_to_same_length(target)
        target = torch.tensor(target, dtype=torch.long)
        return target

    def _get_rolldiff_input_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        input_data = series_df_[
            [
                "anglez",
                "enmo",
                "anglez_absdiff_ave",
                "enmo_absdiff_ave",
                # "anglez_ave",
                # "enmo_ave",
                # "anglez_std",
                # "enmo_std",
            ]
        ].values.T
        input_data = self._padding_data_to_same_length(input_data)
        input_data = torch.tensor(input_data, dtype=torch.float16)
        return input_data

    def __getitem__(self, idx):
        data_key = self.key_df[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        # if len(series_data) > self.data_length:
        #     print(f"[warning] data length is over. series_date_key: {data_key}")
        input = self._get_rolldiff_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].values[0].astype(np.int32),
            "end_step": series_data["step"].values[-1].astype(np.int32),
        }
        if self.mode == "train" or self.mode == "valid":
            target = self._get_target_data(series_data)
            return input, target, input_info_dict
        else:
            return input, input_info_dict


class DSSPseudoDataset(Dataset):
    def __init__(
        self,
        key_df: pd.DataFrame,
        series_df: pd.DataFrame,
        mode: str = "train",
        pseudo_threshold: float = 0.3,
    ) -> None:
        self.use_col = [
            "series_date_key",
            "step",
            "event",
            "anglez",
            "enmo",
            # "anglez_absdiff_ave",
            # "enmo_absdiff_ave",
            "class_pseudo_pred",
        ]
        self.key_df = key_df["series_date_key"].values
        self.series_df = series_df[self.use_col]

        self.mean_std_rollnum_list = [15, 30, 45]
        for col in self.series_df.columns:
            if self.series_df[col].dtype == np.float64:
                self.series_df[col] = self.series_df[col].astype(np.float32)
            elif self.series_df[col].dtype == np.int64:
                self.series_df[col] = self.series_df[col].astype(np.int32)

        self.mode = mode
        self.data_length = 17280
        self.pseudo_threshold = pseudo_threshold

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(self, series_: np.ndarray) -> np.ndarray:
        # series_.shape = [channel, data_length]
        if series_.shape[-1] < self.data_length:
            padding_length = self.data_length - series_.shape[-1]
            # TODO:計測できていないときのデータが0の場合は変更する必要がある
            pad_shape = [(0, 0), (0, padding_length)]
            series_data = np.pad(series_, pad_shape, "edge")
        elif series_.shape[-1] > self.data_length:
            # print(f"[warning] data length is over.")
            series_data = series_[:, : self.data_length]
        else:
            series_data = series_
        return series_data

    def _get_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_["event"].values
        target = target.reshape(1, -1)  # [channel=1, data_length]
        target = self._padding_data_to_same_length(target)
        target = torch.tensor(target, dtype=torch.long)
        return target

    def _get_pseudo_target_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_["class_pseudo_pred"].values
        target = target.reshape(1, -1)  # [channel=1, data_length]
        # 0.5-threshold ~ 0.5+thresholdの値は-1にする
        condition_ignore = (0.5 - self.pseudo_threshold < target) & (
            target < 0.5 + self.pseudo_threshold
        )
        target = np.where(condition_ignore, -np.ones_like(target), target)
        # pseudo_threshold以下の値は0、0.5+pseudo_threshold以上の値は1にする
        condition_zero = (0.0 <= target) & (target <= 0.5 - self.pseudo_threshold)
        target = np.where(condition_zero, np.zeros_like(target), target)
        target = np.where(
            target >= 0.5 + self.pseudo_threshold, np.ones_like(target), target
        )
        # if target.sum() == 0:
        #     raise ValueError("pseudo target is all zero.")
        target = self._padding_data_to_same_length(target)
        target = torch.tensor(target, dtype=torch.long)
        return target

    def _get_pseuo_target_orig(self, series_df_: pd.DataFrame) -> np.ndarray:
        target = series_df_["class_pseudo_pred"].values
        target = target.reshape(1, -1)  # [channel=1, data_length]
        target = self._padding_data_to_same_length(target)
        target = torch.tensor(target, dtype=torch.float32)
        return target

    def _get_rolldiff_input_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        input_data = series_df_[
            ["anglez", "enmo", "anglez_absdiff_ave", "enmo_absdiff_ave"]
        ].values.T
        input_data = self._padding_data_to_same_length(input_data)
        input_data = torch.tensor(input_data, dtype=torch.float16)
        return input_data

    def _get_input_data(self, series_df_: pd.DataFrame) -> np.ndarray:
        input_data = series_df_[
            [
                "anglez",
                "enmo",
            ]
        ].values.T
        for roll_num in self.mean_std_rollnum_list:
            anglez_mean = (
                series_df_["anglez"]
                .rolling(roll_num, center=True)
                .mean()
                .fillna(0)
                .values
            )
            anglez_std = (
                series_df_["anglez"]
                .rolling(roll_num, center=True)
                .std()
                .fillna(0)
                .values
            )
            enmo_mean = (
                series_df_["enmo"]
                .rolling(roll_num, center=True)
                .mean()
                .fillna(0)
                .values
            )
            enmo_std = (
                series_df_["enmo"].rolling(roll_num, center=True).std().fillna(0).values
            )
            input_data = np.concatenate(
                [
                    input_data,
                    np.expand_dims(anglez_mean, axis=0),
                    np.expand_dims(anglez_std, axis=0),
                    np.expand_dims(enmo_mean, axis=0),
                    np.expand_dims(enmo_std, axis=0),
                ]
            )
        input_data = self._padding_data_to_same_length(input_data)
        input_data = torch.tensor(input_data, dtype=torch.float16)
        return input_data

    def __getitem__(self, idx):
        data_key = self.key_df[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        # input = self._get_rolldiff_input_data(series_data)
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].values[0].astype(np.int32),
            "end_step": series_data["step"].values[-1].astype(np.int32),
        }
        target = self._get_target_data(series_data)
        pseudo_target = self._get_pseudo_target_data(series_data)
        # pseudo_target_orig = self._get_pseuo_target_orig(series_data)
        # return input, target, pseudo_target, input_info_dict, pseudo_target_orig
        return input, target, pseudo_target, input_info_dict


class DSSEventDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.data_length = 17280

        maxpool_kernel_size = 361  # 奇数じゃないと同じ長さで出力できない
        maxpool_stride = 1
        # 入力サイズと出力サイズが一致するようにpaddingを調整
        maxpool_padding = int((maxpool_kernel_size - maxpool_stride) / 2)
        self.max_pool = nn.MaxPool1d(
            kernel_size=maxpool_kernel_size,
            stride=maxpool_stride,
            padding=maxpool_padding,
        )
        ave_kernel_size = 101
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
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        else:
            target = self._get_target_data(series_data)
            event_target = self._get_event_target_data(series_data)
            return input, target, event_target, input_info_dict


class DSSEventDetDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.mode = mode
        self.data_length = 17280

        maxpool_kernel_size = 61  # scoreが1.0の範囲
        maxpool_stride = 1
        # 入力サイズと出力サイズが一致するようにpaddingを調整
        maxpool_padding = int((maxpool_kernel_size - maxpool_stride) / 2)
        self.max_pool = nn.MaxPool1d(
            kernel_size=maxpool_kernel_size,
            stride=maxpool_stride,
            padding=maxpool_padding,
        )
        # negativeが多くなりすぎてlogitsが小さくなってしまうのでかなり広めにとる
        ave_kernel_size = 361  # scoreが入る範囲
        # ave_kernel_size = 3601  # scoreが入る範囲
        ave_stride = 1
        ave_padding = int((ave_kernel_size - ave_stride) / 2)
        self.average_pool = nn.AvgPool1d(
            kernel_size=ave_kernel_size, stride=ave_stride, padding=ave_padding
        )
        print("dateset_initialized")

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
        class_pred = series_df_["class_pred"].values
        anglez = self._padding_data_to_same_length(anglez)
        enmo = self._padding_data_to_same_length(enmo)
        class_pred = self._padding_data_to_same_length(class_pred)
        input_data = np.concatenate(
            [
                np.expand_dims(anglez, axis=0),
                np.expand_dims(enmo, axis=0),
                np.expand_dims(class_pred, axis=0),
            ]
        )  # [channel, data_length
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
        positive_target = (target == 1).float()
        for _ in range(100):
            target = self.average_pool(target)
        # target = self.average_pool(target)
        target = torch.clip(positive_target + target, 0, 1)
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
        input = self._get_input_data(series_data)
        # series_date_keyと開始時刻のstepをdictにしておく
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        else:
            event_target = self._get_event_target_data(series_data)
            return input, event_target, input_info_dict


def get_loader(CFG, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"):
    if mode == "pseudo":
        if hasattr(CFG, "pseudo_threshold"):
            pseudo_threshold = CFG.pseudo_threshold
        else:
            pseudo_threshold = 0.15
        dataset = DSSPseudoDataset(key_df, series_df, mode, pseudo_threshold)
    else:
        if CFG.model_type == "event_output":
            dataset = DSSEventDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "add_rolldiff":
            dataset = DSSAddRolldiffDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "mean_stds":
            dataset = DSSMeanStdsDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "event_detect":
            dataset = DSSEventDetDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "downsample":
            dataset = DSSDownSampleDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "target_downsample":
            dataset = DSSTargetDownsampleDataset(
                key_df, series_df, mode
            )  # type: ignore
        elif CFG.model_type == "input_target_downsample":
            dataset = DSSTargetDownsampleDataset(
                key_df, series_df, mode
            )  # type: ignore
        elif CFG.model_type == "input_target_downsample_dense":
            dataset = DSSTargetDownsampleDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "input_target_downsample_3ch":
            dataset = DSSTargetDownsample3chDataset(
                key_df, series_df, mode
            )  # type: ignore
        elif CFG.model_type == "target_downsample_event":
            dataset = DSSTargetDownsampleEventDataset(
                key_df, series_df, mode
            )  # type: ignore
        elif CFG.model_type == "input_target_downsample_dt":
            dataset = DSSTargetDownsampleDTDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "dense":
            dataset = DSSDenseDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "dense2ch":
            dataset = DSSDense2chDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "dense_lstm":
            dataset = DSSDenseDataset(key_df, series_df, mode)  # type: ignore
        else:
            dataset = DSSDataset(key_df, series_df, mode)  # type: ignore
    if mode == "train" or mode == "pseudo":
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
    num_workers = 0

    class CFG:
        pseudo_threshold = 0.3
        num_workers = num_workers
        batch_size = 2
        mode = "train"
        # mode = "test"
        model_type = "dense_lstm"

    # series_df = pd.read_parquet(
    #     "/kaggle/input/targetdownsample_train_series_hour_fold.parquet"
    # )
    series_df = pd.read_parquet("/kaggle/input/train_series_alldata_skffold.parquet")

    # print(series_df.head())
    key_df = series_df[["series_date_key", "series_date_key_str"]].drop_duplicates()
    key_df["series_id"], key_df["date"] = (
        key_df["series_date_key_str"].str.split("_", 1).str
    )
    key_df = key_df.drop(columns=["series_date_key_str"], axis=1)

    dataloader = get_loader(CFG, key_df, series_df, mode=CFG.mode)
    print("dataloader length:", len(dataloader))
    import time

    start_time = time.time()
    if CFG.mode != "pseudo":
        for idx, (input, target, input_info) in enumerate(dataloader):
            # print("\r idx", idx, end="")
            print(idx)
            load_time = time.time() - start_time
            print("load time:", load_time)
            start_time = time.time()

            print(input.shape)
            print(target.shape)
            print(input_info)
            break
    else:
        for idx, (input, target, pseudo_target, input_info) in enumerate(dataloader):
            print(idx)
            load_time = time.time() - start_time
            print("load time:", load_time)
            start_time = time.time()
            for batch_idx in range(pseudo_target.shape[0]):
                print("pseudo_target sum", (pseudo_target[batch_idx] != -1).sum())

            print(input.shape)
            print(target.shape)
            # print(pseudo_target.shape)
            break
