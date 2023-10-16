import argparse
import gc
import os
import sys
import time

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(SRC_DIR, "exp"))
sys.path.append(os.path.join(SRC_DIR, "data"))
sys.path.append(os.path.join(SRC_DIR, "model"))
sys.path.append(os.path.join(SRC_DIR, "dss_utils"))


import numpy as np
import pandas as pd
import torch

from dss_dataloader import get_loader  # isort:skip
from dss_model import get_model  # isort:skip
from postprocess import detect_event_from_classpred, make_submission_df  # isort:skip
from preprocess import label_encode_series_date_key  # isort:skip
from preprocess import preprocess_input  # isort:skip
from preprocess import set_seriesdatekey  # isort:skip
from training_loop import concat_valid_input_info, get_valid_values_dict  # isort:skip


def predict(CFG, model, infer_loader):
    model.eval()

    infer_predictions = {"class_preds": np.empty(0)}
    infer_input_info = {"series_date_key": [], "start_step": [], "end_step": []}

    for _, (inputs, input_info_dict) in enumerate(infer_loader):
        inputs = inputs.to(CFG.device, non_blocking=True).float()
        with torch.no_grad():
            preds = model(inputs)

        infer_predictions = get_valid_values_dict(
            preds, infer_predictions, mode="preds"
        )
        infer_input_info = concat_valid_input_info(infer_input_info, input_info_dict)

    del inputs, preds
    gc.collect()
    torch.cuda.empty_cache()
    return infer_predictions, infer_input_info


def split_data(series_df, tmp_file_path, split_num=3):
    series_id_unique = series_df["series_id"].unique()
    for idx in range(split_num):
        series_id_split = series_id_unique[
            idx
            * (len(series_id_unique) // split_num) : (idx + 1)
            * (len(series_id_unique) // split_num)
        ]
        series_df_split = series_df[series_df["series_id"].isin(series_id_split)]
        print(f"series_df_split_{idx} data num : {len(series_df_split)}")
        split_df_path = os.path.join(tmp_file_path, f"series_df_split_{idx}.parquet")
        series_df_split.to_parquet(split_df_path, index=False)
        print(f"split_df is saved as {split_df_path}")


def get_pred_df(
    input_info_dict: dict,
    preds_dict: dict,
    pred_df: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    start_time = time.time()
    print("creating oof_df", end=" ... ")
    for key in input_info_dict.keys():
        if not isinstance(input_info_dict[key], np.ndarray):
            input_info_dict[key] = input_info_dict[key].numpy()
    for idx, (series_date_key, start_step, end_step) in enumerate(
        zip(
            input_info_dict["series_date_key"],
            input_info_dict["start_step"],
            input_info_dict["end_step"],
        )
    ):
        # preds targets shape: [batch, ch, data_length]
        class_pred = preds_dict["class_preds"][idx]
        data_condition = (
            (pred_df["series_date_key"] == series_date_key)
            & (start_step <= pred_df["step"])
            & (pred_df["step"] <= end_step + 1)
        )
        series_date_data_num = len((pred_df[data_condition]))
        # steps = range(start_step, end_step + 1, 1)
        steps = range(start_step, start_step + series_date_data_num, 1)
        if series_date_data_num < len(class_pred[0]):
            class_pred = class_pred[0, :series_date_data_num]
        elif series_date_data_num > len(class_pred[0]):
            padding_num = series_date_data_num - len(class_pred[0])
            class_pred = np.concatenate(
                [class_pred[0], -1 * np.ones(padding_num)], axis=0
            )
        else:
            class_pred = class_pred[0]
        if not len(class_pred) == len(steps):
            print("len(class_pred)", len(class_pred))
            print("len(steps)", len(steps))
            raise ValueError("preds and targets length is not same")
        pred_col_name = f"class_pred_fold{fold}"
        pred_df.loc[data_condition, pred_col_name] = class_pred
    elapsed = int(time.time() - start_time) / 60
    print(f" >> oof_df created. elapsed time: {elapsed:.2f} min")
    return pred_df


def inference(
    CFG, exp_dir, exp_name, series_df_path, tmp_file_path, sub_path, split_num=3
):
    infer_start_time = time.time()
    series_df = pd.read_parquet(series_df_path)

    os.makedirs(tmp_file_path, exist_ok=True)
    split_data(series_df, tmp_file_path, split_num)
    del series_df
    gc.collect()
    for idx in range(split_num):
        series_df = pd.read_parquet(
            os.path.join(tmp_file_path, f"series_df_split_{idx}.parquet")
        )
        series_df = preprocess_input(series_df)
        print("preprocess series_df")
        series_df = set_seriesdatekey(series_df)
        print("label encode series date key")
        series_df = label_encode_series_date_key(series_df)
        key_df = series_df[["series_date_key", "series_date_key_str"]].drop_duplicates()
        key_df = key_df.reset_index(drop=True)
        key_df["series_id"], key_df["date"] = (
            key_df["series_date_key_str"].str.split("_", 1).str
        )
        key_df = key_df.drop(columns=["series_date_key_str"], axis=1)

        print("series_df data num : {}".format(len(series_df)))
        print("key_df data num : {}".format(len(key_df)))
        pred_df = series_df[["series_id", "series_date_key", "step"]].copy()
        for fold in CFG.folds:
            pred_df[f"class_pred_fold{fold}"] = -1
            print(f"-- fold{fold} inference start --")
            # set model & learning fn
            model = get_model(CFG)
            model_path = os.path.join(exp_dir, exp_name, f"fold{fold}_model.pth")
            model.load_state_dict(torch.load(model_path))
            model = model.to(CFG.device)
            # training
            start_time = time.time()
            # separate train/valid data
            print(f"fold[{fold}] get_loader finished")
            infer_loader = get_loader(CFG, key_df, series_df, mode="infer")
            infer_preds, infer_input_dict = predict(CFG, model, infer_loader)
            elapsed = int(time.time() - start_time) / 60
            print(f"split[{idx}] fold[{fold}] prediction finished.")
            print("elapsed time: {:.2f} min".format(elapsed))
            pred_df = get_pred_df(
                infer_input_dict,
                infer_preds,
                pred_df,
                fold,
            )
        pred_df["class_pred"] = pred_df[
            [f"class_pred_fold{fold}" for fold in CFG.folds]
        ].mean(axis=1)
        pred_df = pred_df.drop(columns=[f"class_pred_fold{fold}" for fold in CFG.folds])
        pred_df = detect_event_from_classpred(pred_df)
        pred_df.to_csv(os.path.join(tmp_file_path, f"pred_df_split_{idx}.csv"))

        sub_df_split = make_submission_df(pred_df)
        # sub_df_split = sub_df_split.drop(columns=["event_pred"])
        sub_df_split = sub_df_split.reset_index(drop=True)
        print(f"sub_df_split_{idx} data num : {len(sub_df_split)}")
        sub_df_split_path = os.path.join(tmp_file_path, f"sub_df_split_{idx}.csv")
        sub_df_split.to_csv(sub_df_split_path, index=False)
        print(f"sub_df_split is saved as {sub_df_split_path}")

    sub_df = pd.concat(
        [
            pd.read_csv(os.path.join(tmp_file_path, f"sub_df_split_{idx}.csv"))
            for idx in range(split_num)
        ],
        axis=0,
    )
    sub_df = sub_df.drop("event_pred", axis=1)
    sub_df = sub_df.reset_index(drop=True)
    print(f"sub_df data num : {len(sub_df)}")
    sub_df_path = os.path.join(sub_path, "submission.csv")
    sub_df.to_csv(sub_df_path, index=False)
    print(f"sub_df is saved as {sub_df_path}")
    elapsed = int(time.time() - infer_start_time) / 60
    print(f"inference finished. elapsed time: {elapsed:.2f} min")
    return sub_df


if __name__ == "__main__":
    exp_dir = "/kaggle/working/"
    exp_name = "baseline_alldata_allfold_000"
    # series_df_path = (
    #     "/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet"
    # )
    series_df_path = (
        "/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet"
    )
    tmp_file_path = "/kaggle/working/submission"
    sub_path = "/kaggle/working/submission"

    # test_df = pd.read_parquet(series_df_path)
    # print(test_df.head())
    # print(test_df["series_id"].unique())
    import yaml  # type: ignore

    config_path = os.path.join(exp_dir, exp_name, "config.yaml")

    config = yaml.load(open(config_path, "r"), Loader=yaml.SafeLoader)
    print(config)
    config = argparse.Namespace(**config)
    print(config)
    inference(
        config,
        exp_dir,
        exp_name,
        series_df_path,
        tmp_file_path,
        sub_path,
    )
