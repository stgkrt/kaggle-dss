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
            return input, target, input_info_dict


class DSSAddRolldiffDataset(Dataset):
    def __init__(
        self, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"
    ) -> None:
        print(mode)
        if mode == "train" or mode == "valid":
            self.use_col = [
                "series_date_key",
                "step",
                "event",
                "anglez",
                "enmo",
                "anglez_absdiff_ave",
                "enmo_absdiff_ave",
            ]
        else:
            self.use_col = [
                "series_date_key",
                "step",
                "anglez",
                "enmo",
                "anglez_absdiff_ave",
                "enmo_absdiff_ave",
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
            ["anglez", "enmo", "anglez_absdiff_ave", "enmo_absdiff_ave"]
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
            "anglez_absdiff_ave",
            "enmo_absdiff_ave",
            "class_pseudo_pred",
        ]
        self.key_df = key_df["series_date_key"].values
        self.series_df = series_df[self.use_col]
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
        condition_ignore = (0.5 - self.pseudo_threshold <= target) & (
            target <= 0.5 + self.pseudo_threshold
        )
        target = np.where(condition_ignore, -np.ones_like(target), target)
        # pseudo_threshold以下の値は0、0.5+pseudo_threshold以上の値は1にする
        condition_zero = (0.0 <= target) & (target <= 0.5 - self.pseudo_threshold)
        target = np.where(condition_zero, np.zeros_like(target), target)
        target = np.where(
            target > 0.5 + self.pseudo_threshold, np.ones_like(target), target
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

    def __getitem__(self, idx):
        data_key = self.key_df[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_rolldiff_input_data(series_data)
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


def get_loader(CFG, key_df: pd.DataFrame, series_df: pd.DataFrame, mode: str = "train"):
    if mode == "pseudo":
        if hasattr(CFG, "pseudo_threshold"):
            pseudo_threshold = CFG.pseudo_threshold
        else:
            pseudo_threshold = 0.3
        dataset = DSSPseudoDataset(key_df, series_df, mode, pseudo_threshold)
    else:
        if CFG.model_type == "event_output":
            dataset = DSSEventDataset(key_df, series_df, mode)  # type: ignore
        elif CFG.model_type == "add_rolldiff":
            dataset = DSSAddRolldiffDataset(key_df, series_df, mode)  # type: ignore
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
    import os

    num_workers = os.cpu_count()
    num_workers = 0
    is_pseudo = True

    class CFG:
        pseudo_threshold = 0.3
        num_workers = num_workers
        batch_size = 2

    # series_df = pd.read_parquet("/kaggle/input/preprocessed_train_series_le.parquet")
    series_df = pd.read_parquet("/kaggle/input/pseudo_train_series_fold0_le20.parquet")
    # series_date_keyとseries_data_key_strのuniqueだけでdataframeを作る
    key_df = series_df[["series_date_key", "series_date_key_str"]].drop_duplicates()
    key_df["series_id"], key_df["date"] = (
        key_df["series_date_key_str"].str.split("_", 1).str
    )
    key_df = key_df.drop(columns=["series_date_key_str"], axis=1)
    print(key_df.head())

    print(series_df.head())

    print("data loaded")

    dataloader = get_loader(CFG, key_df, series_df, mode="pseudo")
    print("dataloader length:", len(dataloader))
    import time

    start_time = time.time()
    if not is_pseudo:
        for idx, (input, target, input_info) in enumerate(dataloader):
            print(idx)
            load_time = time.time() - start_time
            print("load time:", load_time)
            start_time = time.time()

            print(input.shape)
            print(target.shape)
            # print(input_info)
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
            print(pseudo_target.shape)
            # print(input_info)
            break
