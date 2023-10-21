from sklearn.model_selection import GroupKFold  # type: ignore


def get_group_groupkfold_split(df, CFG):
    gkf = GroupKFold(n_splits=CFG.n_folds)
    df["fold"] = -1
    for fold, (_, valid_idx) in enumerate(
        gkf.split(df, groups=df[CFG.group_key].values)
    ):
        df.loc[valid_idx, "fold"] = fold
    return df


def get_train_valid_key_df(df, fold, CFG):
    df = get_group_groupkfold_split(df, CFG)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    return train_df, valid_df


def get_train_valid_series_df(series_df, key_df, fold, mode="train"):
    if mode == "train":
        series_ids = key_df[key_df["fold"] != fold]["series_id"].unique()
    elif mode == "valid":
        series_ids = key_df[key_df["fold"] == fold]["series_id"].unique()
    series_df = series_df[series_df["series_id"].isin(series_ids)]
    return series_df


if __name__ == "__main__":

    class CFG:
        n_folds = 5
        group_key = "series_id"

    import pandas as pd

    # series_df = pd.read_parquet(
    #     "/kaggle/input/preprocessed_train_series_le_200.parquet"
    # )
    series_df = pd.read_parquet("/kaggle/input/preprocessed_train_series_le.parquet")
    print("series_df loaded.")
    train_seires_unique = series_df["series_id"].unique()
    train_seires_unique = train_seires_unique[:50]
    series_df = series_df[series_df["series_id"].isin(train_seires_unique)]
    print("series_df", series_df.head())
    key_df = series_df[["series_date_key", "series_date_key_str"]].drop_duplicates()
    print("key_df from series_df", key_df.head())
    key_df = key_df.reset_index(drop=True)
    key_df["series_id"], key_df["date"] = (
        key_df["series_date_key_str"].str.split("_", 1).str
    )
    key_df = key_df.drop(columns=["series_date_key_str"], axis=1)
    print("get key_df", key_df.head())
    key_df = get_group_groupkfold_split(key_df, CFG)
    print("get key_df", key_df.head())
    series_df["fold"] = -1
    for fold in key_df["fold"].unique():
        series_ids = key_df[key_df["fold"] == fold]["series_id"].unique()
        series_df.loc[series_df["series_id"].isin(series_ids), "fold"] = fold

    series_df.to_parquet("/kaggle/input/preprocessed_train_series_le_50_fold.parquet")
