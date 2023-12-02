import datetime
import warnings

import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")


def pl_datetime_preprocess(train_series_):
    # train_series_ = train_series_.with_columns(
    #     pl.col("timestamp").str.to_datetime().dt.replace_time_zone(None)
    # )
    train_series_ = train_series_.with_columns(
        pl.col("timestamp").dt.second().cast(pl.Int32).alias("second")
    )
    train_series_ = train_series_.with_columns(
        pl.col("timestamp").dt.date().cast(str).alias("date")
    )
    return train_series_


def add_elapsed_date(df: pl.DataFrame):
    # series_idごとにmin, maxを取得
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime().dt.replace_time_zone(None)
    )
    min_time_df = (
        df.select(["series_id", "timestamp"])
        .groupby("series_id", maintain_order=True)
        .min()
    )
    unique_series_id = df["series_id"].unique()
    add_elapsed_date_list = []
    for series_id in unique_series_id:
        first_date = min_time_df.filter(pl.col("series_id") == series_id).get_column(
            "timestamp"
        )[0]
        elaped_date = (
            df.filter(pl.col("series_id") == series_id).get_column("timestamp")
            - first_date
        )
        elaped_date = elaped_date.dt.days()
        # elpaed dateを最大値で割る
        normalized_elapsed_date = elaped_date / elaped_date.max()
        add_elapsed_date_list.extend(normalized_elapsed_date)
    df = df.with_columns(pl.Series(add_elapsed_date_list).alias("elapsed_date"))
    return df


def pl_datetime_preprocess_hourshift(train_series_, shift_hour=12):
    train_series_ = train_series_.with_columns(
        pl.col("timestamp").str.to_datetime().dt.replace_time_zone(None)
    )
    if shift_hour != 0:
        train_series_ = train_series_.with_columns(
            pl.col("timestamp") + datetime.timedelta(hours=12)
        )
    train_series_ = train_series_.with_columns(
        pl.col("timestamp").dt.second().cast(pl.Int32).alias("second")
    )
    train_series_ = train_series_.with_columns(
        pl.col("timestamp").dt.date().cast(str).alias("date")
    )
    return train_series_


def add_filter_feature(series_df: pl.DataFrame):
    series_df = series_df.with_columns(
        pl.col("anglez").diff(1).abs().alias("anglez_abs_diff")
    )
    # anglez_abs_diffのnanを0で埋める
    series_df = series_df.with_columns(
        pl.col("anglez_abs_diff").fill_nan(0).alias("anglez_abs_diff")
    )
    # series_df["filtered_anglez"] = savgol_filter(series_df["anglez_abs_diff"].copy())
    series_df = series_df.with_columns(
        pl.Series("anglez_abs_diff")
        .apply(savgol_filter, return_dtype=float)
        .alias("filtered_anglez")
    )
    return series_df


def add_duplicate(df: pl.DataFrame):
    # NumPy配列への変換
    array = df[["enmo", "anglez"]].to_numpy()

    # 180行ごとに分割
    subsets = [array[i : i + 180] for i in range(0, len(array), 180)]

    subsets_dict = {}
    for i, subset in enumerate(subsets):
        # サブセットをタプルのリストに変換
        subset_key = tuple(map(tuple, subset))
        if subset_key in subsets_dict:
            subsets_dict[subset_key].append(i)
        else:
            subsets_dict[subset_key] = [i]

    # 完全に一致するサブセットのインデックスペアを探す
    matching_subsets = []
    for indices in subsets_dict.values():
        if len(indices) > 1:
            matching_subsets.extend(indices)

    duplicate_array = np.zeros(len(df), dtype=int)

    # matching_subsetsに含まれるサブセットに対応する配列の範囲を1に更新
    for index in matching_subsets:
        if np.mean(subsets[index][:, 0]) == 0:
            continue
        start_row = index * 180
        end_row = start_row + 180
        duplicate_array[start_row:end_row] = 1

    df = df.with_columns(pl.Series(duplicate_array).alias("duplicate"))
    return df


def preprocess_input(train_series_: pd.DataFrame) -> pd.DataFrame:
    train_series_ = train_series_.drop(columns=["timestamp"], axis=1)
    print("get anglez and enmo rolling mean and std")
    for roll_num in [36, 60]:  # 雰囲気で選んだ
        train_series_[f"anglez_mean_{roll_num}"] = (
            train_series_.groupby("series_id")["anglez"]
            .rolling(roll_num, center=True)
            .mean()
            .reset_index(0, drop=True)
        )
        train_series_[f"anglez_std_{roll_num}"] = (
            train_series_.groupby("series_id")["anglez"]
            .rolling(roll_num, center=True)
            .std()
            .reset_index(0, drop=True)
        )
        train_series_[f"anglez_mean_{roll_num}"] = train_series_[
            f"anglez_mean_{roll_num}"
        ].fillna(0)
        train_series_[f"anglez_std_{roll_num}"] = train_series_[
            f"anglez_std_{roll_num}"
        ].fillna(0)

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
    train_event_["event"] = train_event_["event"].fillna(-1)
    train_series_["step"] = train_series_["step"].astype("float64")
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
    df["event"] = df["event"].fillna(-1)

    return df


def set_seriesdatekey(train_series_: pd.DataFrame) -> pd.DataFrame:
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
    print("set groupby label")
    train_series_ = set_train_groupby_label(train_series_, train_event_)
    print("set series date key")
    train_series_ = set_seriesdatekey(train_series_)
    print("label encode series date key")
    train_series_ = label_encode_series_date_key(train_series_)
    return train_series_


if __name__ == "__main__":
    print("load data")

    train_series_df = pl.read_parquet(
        "/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet"
    )
    train_event_df = pd.read_csv(
        "/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv"
    )
    print("add elapsed date")
    train_series_df = add_elapsed_date(train_series_df)
    print("preprocess polars train series")
    train_series_df = pl_datetime_preprocess(train_series_df)
    # print("add duplicate")
    # train_series_df = add_duplicate(train_series_df)
    # print("add filter feature")
    # train_series_df = add_filter_feature(train_series_df)
    # train_series_df = pl_datetime_preprocess_12hourshift(train_series_df)
    train_series_df = train_series_df.to_pandas()
    print(train_series_df.info())
    train_series_df = preprocess_train_series(train_series_df, train_event_df)
    print(len(train_series_df))
    print(train_series_df.info())

    print(len(train_series_df))
    train_series_df.to_parquet("/kaggle/input/train_series_alldata_elapseddate.parquet")
