import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel

warnings.filterwarnings("ignore")


# Local Time converter
def to_date_time(x):
    return pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")  # utc=True


def to_localize(t):
    return t.tz_localize(None)


def preprocess_input(train_series_: pd.DataFrame) -> pd.DataFrame:
    # series_idでgroup_byして一つずらしたanglezとの差分を取る
    print("get anglez diff")
    train_series_["anglez_absdiff"] = np.abs(
        train_series_.groupby("series_id")["anglez"].diff()
    )
    train_series_["enmo_absdiff"] = np.abs(
        train_series_.groupby("series_id")["enmo"].diff()
    )
    train_series_["anglez_absdiff"] = train_series_["anglez_absdiff"].fillna(0)
    train_series_["enmo_absdiff"] = train_series_["enmo_absdiff"].fillna(0)
    # angle_absdiffとenmo_absdiffのaverage poolを取る
    print("get anglez_absdiff and enmo_absdiff rolling mean")
    train_series_["anglez_absdiff_ave"] = (
        train_series_.groupby("series_id")["anglez_absdiff"]
        .rolling(101, center=True)
        .mean()
        .reset_index(0, drop=True)
    )
    train_series_["enmo_absdiff_ave"] = (
        train_series_.groupby("series_id")["enmo_absdiff"]
        .rolling(101, center=True)
        .mean()
        .reset_index(0, drop=True)
    )
    train_series_["anglez_absdiff_ave"] = train_series_["anglez_absdiff_ave"].fillna(0)
    train_series_["enmo_absdiff_ave"] = train_series_["enmo_absdiff_ave"].fillna(0)
    return train_series_


def set_train_groupby_label(
    train_series_: pd.DataFrame, train_event_: pd.DataFrame
) -> pd.DataFrame:
    # abs diffにつかうaverage pool
    print("set unknown_onset and unknown_wakeup for null step")
    train_event_.loc[
        (train_event_["step"].isnull()) & (train_event_["event"] == "onset"), "event"
    ] = "unknown_onset"
    train_event_.loc[
        (train_event_["step"].isnull()) & (train_event_["event"] == "wakeup"), "event"
    ] = "unknown_wakeup"

    # series_idでgroup_byしてunknown_onsetとunknown_wakeupのstepを補完する
    print("fill unknown_onset and unknown_wakeup step")
    train_event_["step"] = train_event_.groupby("series_id")["step"].apply(
        lambda x: x.bfill(axis="rows")
    )
    train_event_["step"] = train_event_.groupby("series_id")["step"].apply(
        lambda x: x.ffill(axis="rows")
    )
    train_series_ = preprocess_input(train_series_)
    # eventのonsetを0, wakeupを1, unknown_onsetとunknown_wakeupを2とする
    print("set event label")
    train_event_["event"] = train_event_["event"].map(
        {"onset": 0, "wakeup": 1, "unknown_onset": -1, "unknown_wakeup": -1}
    )

    # series_idとstepでmergeする
    print("merge series_id and step")
    df = pd.merge(
        train_series_,
        train_event_[["series_id", "step", "event"]],
        on=["series_id", "step"],
        how="left",
    )
    print("fill event")
    df["event_onset"] = df["event"].apply(lambda x: 1 if x == 0 else 0)
    df["event_wakeup"] = df["event"].apply(lambda x: 1 if x == 1 else 0)
    df = df.bfill(axis="rows")
    return df


def get_date_from_timestamp(timestamp):
    return timestamp.split("T")[0]


def set_seriesdatekey(train_series_: pd.DataFrame) -> pd.DataFrame:
    pandarallel.initialize(progress_bar=True)
    train_series_["date"] = train_series_["timestamp"].parallel_apply(
        get_date_from_timestamp
    )
    train_series_["series_date_key"] = (
        train_series_["series_id"].astype(str) + "_" + train_series_["date"].astype(str)
    )
    return train_series_


def label_encode_series_date_key(train_series_: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    train_series_["series_date_key_str"] = train_series_["series_date_key"].astype(str)
    train_series_["series_date_key"] = le.fit_transform(
        train_series_["series_date_key_str"]
    )
    train_series_["series_date_key"] = train_series_["series_date_key"].astype("int16")
    return train_series_


def preprocess_train_series(
    train_series_: pd.DataFrame, train_event_: pd.DataFrame
) -> pd.DataFrame:
    print("set unknown_onset and unknown_wakeup for null step")
    train_series_ = set_train_groupby_label(train_series_, train_event_)
    print("set series date key")
    train_series_ = set_seriesdatekey(train_series_)
    print("label encode series date key")
    train_series_ = label_encode_series_date_key(train_series_)
    return train_series_


if __name__ == "__main__":
    print("load data")
    train_series_df = pd.read_parquet(
        "/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet"
    )
    train_event_df = pd.read_csv(
        "input/child-mind-institute-detect-sleep-states/train_events.csv"
    )
    print("preprocessing data...")
    preprocessed_df = preprocess_train_series(train_series_df, train_event_df)
    print(preprocessed_df.head())

    # preprocessed_df.to_parquet("/kaggle/input/preprocessed_train_series_le.parquet")
