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
