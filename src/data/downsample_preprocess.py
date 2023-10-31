import warnings

import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")


def pl_datetime_preprocess(train_series_):
    train_series_ = train_series_.with_columns(
        pl.col("timestamp").str.to_datetime().dt.replace_time_zone(None)
    )
    train_series_ = train_series_.with_columns(
        pl.col("timestamp").dt.second().cast(pl.Int32).alias("second")
    )
    train_series_ = train_series_.with_columns(
        pl.col("timestamp").dt.date().cast(str).alias("date")
    )
    return train_series_


def preprocess_input(train_series_: pd.DataFrame) -> pd.DataFrame:
    print("get anglez and enmo rolling mean")
    roll_num = 12
    train_series_["anglez_mean"] = (
        train_series_.groupby("series_id")["anglez"]
        .rolling(roll_num, center=True)
        .mean()
        .reset_index(0, drop=True)
    )
    train_series_["enmo_mean"] = (
        train_series_.groupby("series_id")["enmo"]
        .rolling(roll_num, center=True)
        .mean()
        .reset_index(0, drop=True)
    )
    train_series_["anglez"] = train_series_["anglez_mean"].fillna(0)
    train_series_["enmo"] = train_series_["enmo_mean"].fillna(0)
    train_series_ = train_series_.drop(columns=["anglez_mean", "enmo_mean"])

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
    # df["event"] = df["event"].fillna(-1)
    df["event_onset"] = df["event"].apply(lambda x: 1 if x == 0 else 0)
    df["event_wakeup"] = df["event"].apply(lambda x: 1 if x == 1 else 0)
    df = df.bfill(axis="rows")
    df["event"] = df["event"].fillna(-1)

    # train_series_ = cast_64to16(train_series_)
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
    print("set unknown_onset and unknown_wakeup for null step")
    train_series_ = set_train_groupby_label(train_series_, train_event_)
    print("set series date key")
    train_series_ = set_seriesdatekey(train_series_)
    print("label encode series date key")
    train_series_ = label_encode_series_date_key(train_series_)
    return train_series_


if __name__ == "__main__":
    print("load data")
    # train_series_df = pl.read_parquet(
    #     "/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet"
    # )
    # print("preprocess polars train series")

    # print("preprocess train series")
    # train_series_df = pl_datetime_preprocess(train_series_df)
    # train_series_df = train_series_df.to_pandas()
    # print("preprocess train series")
    # train_series_df = preprocess_train_series(train_series_df, train_event_df)

    # train_series_df = pl.read_parquet("/kaggle/input/downsample_train_series.parquet")
    # train_series_df = train_series_df.to_pandas()
    # print(len(train_series_df))
    # train_event_df = pl.read_csv(
    #     "input/child-mind-institute-detect-sleep-states/train_events.csv"
    # )

    # # train_event_df = pl_datetime_preprocess(train_event_df)
    # # train_event_df = train_event_df.to_pandas()
    # train_event_df["series_date_key"] = (
    #     train_event_df["series_id"].astype(str)
    #     + "_"
    #     + train_event_df["date"].astype(str)
    # )
    # train_event_df = train_event_df.dropna(subset=["step"])
    # event_keys = train_event_df["series_date_key"].unique()
    # train_series_df = train_series_df[
    #     train_series_df["series_date_key_str"].isin(event_keys)
    # ]
    # train_series_df = train_series_df[train_series_df["second"] == 0]
    # train_series_df = train_series_df.reset_index(drop=True)
    # print(len(train_series_df))
    # train_series_df.to_parquet("/kaggle/input/downsample_train_series_event.parquet")

    train_series_df = pd.read_parquet(
        "/kaggle/input/downsample_train_series_fold.parquet"
    )
    train_series_df = train_series_df.drop(columns=["__index_level_0__"])
    train_series_df = train_series_df[train_series_df["second"] == 0].reset_index(
        drop=True
    )
    print(train_series_df.head())

    train_series_df.to_parquet(
        "/kaggle/input/downsample_train_series_fold_zerosec.parquet"
    )
