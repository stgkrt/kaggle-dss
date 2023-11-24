import pandas as pd
import yaml


def set_fold_from_yaml(CFG, series_df, yaml_path_list):
    series_df["fold"] = -1
    for fold, fold_yaml_path in enumerate(CFG.fold_yamls):
        with open(fold_yaml_path, "r") as f:
            fold_series_dict = yaml.load(f, Loader=yaml.FullLoader)
        print(
            len(fold_series_dict["train_series_ids"]),
            len(fold_series_dict["valid_series_ids"]),
        )

        series_df.loc[
            series_df["series_id"].isin(fold_series_dict["valid_series_ids"]), "fold"
        ] = fold
        print(len(series_df[series_df["fold"] == fold]["series_id"].unique()))
    return series_df


if __name__ == "__main__":

    class CFG:
        n_folds = 5
        group_key = "series_id"

        # filename = "/kaggle/input/train_series_alldata.parquet"
        filename = "/kaggle/input/train_series_alldata_halflabel.parquet"
        # output_filename = "/kaggle/input/train_series_alldata_fold.parquet"
        output_filename = "/kaggle/input/train_series_alldata_halflabel_fold.parquet"
        fold_yamls = [
            f"/kaggle/input/fold_split_yaml/stfk_event_count_bins105_fold{i}.yaml"
            for i in range(n_folds)
        ]

    series_df = pd.read_parquet(CFG.filename)
    series_df = set_fold_from_yaml(CFG, series_df, CFG.fold_yamls)
    series_df.to_parquet(CFG.output_filename)
