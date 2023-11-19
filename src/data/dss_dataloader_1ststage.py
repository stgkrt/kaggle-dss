import numpy as np
import pandas as pd  # type: ignore
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DSS1stDataset(Dataset):
    def __init__(
        self,
        key_df: pd.DataFrame,
        series_df: pd.DataFrame,
        event_df: pd.DataFrame | None = None,
        mode: str = "train",
    ) -> None:
        self.key_df = key_df
        self.series_df = series_df
        self.event_df = event_df
        self.mode = mode
        if self.mode == "train" and self.event_df is None:
            raise ValueError("event_df is None.")
        self.data_length = 17280

    def __len__(self) -> int:
        return len(self.key_df)

    def _padding_data_to_same_length(self, series_: np.ndarray) -> np.ndarray:
        if len(series_) < self.data_length:
            padding_length = self.data_length - len(series_)
            padding_data = -np.ones(padding_length)
            series_data = np.concatenate([series_, padding_data])
        elif len(series_) > self.data_length:
            # print(f"[warning] data length is over.")
            series_data = series_[: self.data_length]
        else:
            series_data = series_
        return series_data

    def _get_input_data(self, series_df_: pd.DataFrame) -> torch.Tensor:
        anglez = series_df_["anglez"].values
        enmo = series_df_["enmo"].values
        anglez = self._padding_data_to_same_length(anglez)
        enmo = self._padding_data_to_same_length(enmo)
        input_data = np.concatenate(
            [np.expand_dims(anglez, axis=0), np.expand_dims(enmo, axis=0)]
        )  # [channel, data_length]
        input_data = torch.tensor(input_data, dtype=torch.float32)
        return input_data

    def _get_target_data(
        self, series_df_: pd.DataFrame, event_df: pd.DataFrame
    ) -> torch.Tensor:
        min_step = series_df_["step"].min()
        max_step = series_df_["step"].max()
        series_id = series_df_["series_id"].iloc[0]
        target_event = event_df[event_df["series_id"] == series_id]
        target_event = target_event[
            (target_event["step"] >= min_step) & (target_event["step"] <= max_step)
        ]
        target = np.array(len(target_event) > 0).astype(np.int32)
        target = torch.tensor(target, dtype=torch.long)
        return target.unsqueeze(0)

    def __getitem__(self, idx):
        data_key = self.key_df["series_date_key"].iloc[idx]
        series_data = self.series_df[self.series_df["series_date_key"] == data_key]
        input = self._get_input_data(series_data)
        input_info_dict = {
            "series_date_key": data_key,
            "start_step": series_data["step"].iloc[0].astype(np.int32),
            "end_step": series_data["step"].iloc[-1].astype(np.int32),
        }
        if self.mode == "test":
            return input, input_info_dict
        else:
            target = self._get_target_data(series_data, self.event_df)  # type: ignore
            return input, target, input_info_dict


def get_loader(
    CFG,
    key_df: pd.DataFrame,
    series_df: pd.DataFrame,
    event_df: pd.DataFrame | None = None,
    mode: str = "train",
):
    dataset = DSS1stDataset(key_df, series_df, event_df, mode=mode)
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
        num_workers = num_workers
        batch_size = 16
        mode = "train"
        # mode = "test"
        model_type = "dense_lstm"

    # series_df = pd.read_parquet(
    #     "/kaggle/input/targetdownsample_train_series_hour_fold.parquet"
    # )
    series_df = pd.read_parquet("/kaggle/input/train_series_alldata_skffold.parquet")
    event_df = pd.read_csv(
        "/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv"
    )
    # print(series_df.head())
    key_df = series_df[["series_date_key", "series_date_key_str"]].drop_duplicates()
    key_df["series_id"], key_df["date"] = (
        key_df["series_date_key_str"].str.split("_", 1).str
    )
    key_df = key_df.drop(columns=["series_date_key_str"], axis=1)

    dataloader = get_loader(CFG, key_df, series_df, event_df, mode=CFG.mode)
    print("dataloader length:", len(dataloader))
    import time

    start_time = time.time()
    for idx, (input, target, input_info) in enumerate(dataloader):
        # print("\r idx", idx, end="")
        print(idx)
        load_time = time.time() - start_time
        print("load time:", load_time)
        start_time = time.time()

        print(input.shape)
        print(target.shape)
        print(target)
        print(input_info)
        break
