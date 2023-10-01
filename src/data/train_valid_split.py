from sklearn.model_selection import GroupKFold  # type: ignore


def get_group_groupkfold_split(df, CFG):
    gkf = GroupKFold(n_splits=CFG.n_folds)
    df["fold"] = -1
    for fold, (_, valid_idx) in enumerate(
        gkf.split(df, groups=df[CFG.group_key].values)
    ):
        df.loc[valid_idx, "fold"] = fold
    return df


def get_train_valid_df(df, fold, CFG):
    df = get_group_groupkfold_split(df, n_splits=CFG.n_fold, seed=CFG.seed)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    return train_df, valid_df


def get_train_valid_series_df(series_df, key_df, fold):
    train_series_ids = key_df[key_df["fold"] != fold]["series_id"].unique()
    valid_series_ids = key_df[key_df["fold"] == fold]["series_id"].unique()
    train_series_df = series_df[series_df["series_id"].isin(train_series_ids)]
    valid_series_df = series_df[series_df["series_id"].isin(valid_series_ids)]
    return train_series_df, valid_series_df
