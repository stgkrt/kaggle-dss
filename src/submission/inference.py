# import argparse
# import gc
# import os
# import sys
# import time

# SRC_DIR = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), os.path.pardir))
# sys.path.append(os.path.join(SRC_DIR, "exp"))
# sys.path.append(os.path.join(SRC_DIR, "data"))
# sys.path.append(os.path.join(SRC_DIR, "model"))
# sys.path.append(os.path.join(SRC_DIR, "dss_utils"))

# import numpy as np
# import pandas as pd
# import torch
# from sklearn.preprocessing import LabelEncoder

# from dss_dataloader import get_loader  # isort:skip
# from dss_model import get_model  # isort:skip
# from postprocess import detect_event_from_classpred, make_submission_df  # isort:skip
# from preprocess import label_encode_series_date_key  # isort:skip
# from preprocess import preprocess_input  # isort:skip
# from preprocess import set_seriesdatekey  # isort:skip
# from training_loop import concat_valid_input_info, get_valid_values_dict  # isort:skip

# def preprocess_input(train_series_: pd.DataFrame) -> pd.DataFrame:
#     # series_idでgroup_byして一つずらしたanglezとの差分を取る
#     #     print("get anglez diff")
#     train_series_["anglez_absdiff"] = np.abs(
#         train_series_.groupby("series_id")["anglez"].diff())
#     train_series_["enmo_absdiff"] = np.abs(
#         train_series_.groupby("series_id")["enmo"].diff())
#     train_series_["anglez_absdiff"] = train_series_["anglez_absdiff"].fillna(0)
#     train_series_["enmo_absdiff"] = train_series_["enmo_absdiff"].fillna(0)
#     # angle_absdiffとenmo_absdiffのaverage poolを取る
#     #     print("get anglez_absdiff and enmo_absdiff rolling mean")
#     train_series_["anglez_absdiff_ave"] = (
#         train_series_.groupby("series_id")["anglez_absdiff"].rolling(
#             101, center=True).mean().reset_index(0, drop=True))
#     train_series_["enmo_absdiff_ave"] = (
#         train_series_.groupby("series_id")["enmo_absdiff"].rolling(
#             101, center=True).mean().reset_index(0, drop=True))
#     train_series_["anglez_absdiff_ave"] = train_series_[
#         "anglez_absdiff_ave"].fillna(0)
#     train_series_["enmo_absdiff_ave"] = train_series_[
#         "enmo_absdiff_ave"].fillna(0)
#     return train_series_

# def get_date_from_timestamp(timestamp):
#     return timestamp.split("T")[0]

# def set_seriesdatekey(train_series_: pd.DataFrame) -> pd.DataFrame:
#     train_series_["date"] = train_series_["timestamp"].apply(
#         get_date_from_timestamp)
#     train_series_["series_date_key"] = (
#         train_series_["series_id"].astype(str) + "_" +
#         train_series_["date"].astype(str))
#     return train_series_

# def label_encode_series_date_key(train_series_: pd.DataFrame) -> pd.DataFrame:
#     le = LabelEncoder()
#     train_series_["series_date_key_str"] = train_series_[
#         "series_date_key"].astype(str)
#     train_series_["series_date_key"] = le.fit_transform(
#         train_series_["series_date_key_str"])
#     train_series_["series_date_key"] = train_series_["series_date_key"].astype(
#         "int16")
#     return train_series_

# # 1step 0.5secで30minなら60*30=1800step?
# # metric的にいっぱい検出してもいい？とりあえず小さめ
# def detect_event_from_classpred(df,
#                                 N=301,
#                                 maxpool_kernel_size=41,
#                                 maxpool_stride=1) -> pd.DataFrame:
#     # series_idでgroupbyして、class_predに対して対象の列のデータから前のN個の列までの
# データの平均をとる
#     df["class_pred"] = df["class_pred"].astype(np.float16)
#     df["class_pred_beforemean"] = (df.groupby("series_id")["class_pred"].apply(
#         lambda x: x.rolling(N, min_periods=1).mean()).reset_index(drop=True))
#     df["class_pred_aftermean"] = (df.groupby("series_id")["class_pred"].apply(
#         lambda x: x[::-1].rolling(N, min_periods=1).mean()[::-1]).reset_index(
#             drop=True))
#     df["event_pred"] = df["class_pred_beforemean"] - df["class_pred_aftermean"]
#     df = df.drop(["class_pred_beforemean", "class_pred_aftermean"], axis=1)

#     # 入力サイズと出力サイズが一致するようにpaddingを調整
#     maxpool_padding = int((maxpool_kernel_size - maxpool_stride) / 2)
#     # maxpoolしてピーク検出
#     max_pooling = nn.MaxPool1d(maxpool_kernel_size,
#                                stride=maxpool_stride,
#                                padding=maxpool_padding)
#     event_pred = df["event_pred"].values
#     event_pred = torch.tensor(event_pred).unsqueeze(0)
#     pooled_event_pred = max_pooling(np.abs(event_pred)).squeeze(0).numpy()
#     event_pred = event_pred.squeeze(0).numpy()
#     # peakのところだけ残すmaskを作成
#     peak_event_pred_mask = np.where(pooled_event_pred == np.abs(event_pred), 1,
#                                     0)
#     peak_event_pred = event_pred * peak_event_pred_mask
#     df["event_pred"] = peak_event_pred

#     return df

# def make_submission_df(df, threshold: float = 0.01) -> pd.DataFrame:
#     df = df[["series_id", "step", "event_pred"]]
#     # thresholdより大きいときは1,-thresholdより小さいときは-1,それ以外は0
#     df["event"] = df["event_pred"].apply(lambda x: 1 if x > threshold else -1
#                                          if x < -threshold else 0)
#     df = df[df["event"] != 0]
#     df["event"] = df["event"].replace({1: "wakeup", -1: "onset"})
#     df["score"] = df["event_pred"].apply(
#         lambda x: np.clip(np.abs(x), 0.0, 1.0))
#     return df

# def predict(CFG, model, infer_loader):
#     model.eval()

#     infer_predictions = {"class_preds": np.empty(0)}
#     infer_input_info = {
#         "series_date_key": [],
#         "start_step": [],
#         "end_step": []
#     }

#     for _, (inputs, input_info_dict) in enumerate(infer_loader):
#         inputs = inputs.to(CFG.device, non_blocking=True).float()
#         with torch.no_grad():
#             preds = model(inputs)

#         infer_predictions = get_valid_values_dict(preds,
#                                                   infer_predictions,
#                                                   mode="preds")
#         infer_input_info = concat_valid_input_info(infer_input_info,
#                                                    input_info_dict)

#     del inputs, preds
#     gc.collect()
#     torch.cuda.empty_cache()
#     return infer_predictions, infer_input_info

# def split_data(series_df, tmp_file_path, split_num=3):
#     series_id_unique = series_df["series_id"].unique()
#     for idx in range(split_num):
#         series_id_split = series_id_unique[idx * (len(series_id_unique) //
#                                                   split_num):(idx + 1) *
#                                            (len(series_id_unique) //
#                                             split_num)]
#         series_df_split = series_df[series_df["series_id"].isin(
#             series_id_split)]
#         print(f"series_df_split_{idx} data num : {len(series_df_split)}")
#         split_df_path = os.path.join(tmp_file_path,
#                                      f"series_df_split_{idx}.parquet")
#         series_df_split.to_parquet(split_df_path, index=False)
#         print(f"split_df is saved as {split_df_path}")

# def get_pred_df(
#     input_info_dict: dict,
#     preds_dict: dict,
#     pred_df: pd.DataFrame,
#     fold: int,
# ) -> pd.DataFrame:
#     start_time = time.time()
#     print("creating oof_df", end=" ... ")
#     for key in input_info_dict.keys():
#         if not isinstance(input_info_dict[key], np.ndarray):
#             input_info_dict[key] = input_info_dict[key].numpy()
#     for idx, (series_date_key, start_step, end_step) in enumerate(
#             zip(
#                 input_info_dict["series_date_key"],
#                 input_info_dict["start_step"],
#                 input_info_dict["end_step"],
#             )):
#         # preds targets shape: [batch, ch, data_length]
#         class_pred = preds_dict["class_preds"][idx]
#         data_condition = ((pred_df["series_date_key"] == series_date_key)
#                           & (start_step <= pred_df["step"])
#                           & (pred_df["step"] <= end_step + 1))
#         series_date_data_num = len((pred_df[data_condition]))
#         # steps = range(start_step, end_step + 1, 1)
#         steps = range(start_step, start_step + series_date_data_num, 1)
#         if series_date_data_num < len(class_pred[0]):
#             class_pred = class_pred[0, :series_date_data_num]
#         elif series_date_data_num > len(class_pred[0]):
#             padding_num = series_date_data_num - len(class_pred[0])
#             class_pred = np.concatenate(
#                 [class_pred[0], -1 * np.ones(padding_num)], axis=0)
#         else:
#             class_pred = class_pred[0]
#         if not len(class_pred) == len(steps):
#             print("len(class_pred)", len(class_pred))
#             print("len(steps)", len(steps))
#             raise ValueError("preds and targets length is not same")
#         pred_col_name = f"class_pred_fold{fold}"
#         pred_df.loc[data_condition, pred_col_name] = class_pred
#     elapsed = int(time.time() - start_time) / 60
#     print(f" >> oof_df created. elapsed time: {elapsed:.2f} min")
#     return pred_df

# def inference(CFG,
#               exp_dir,
#               exp_name,
#               series_df_path,
#               tmp_file_path,
#               split_num=3):
#     infer_start_time = time.time()
#     series_df = pd.read_parquet(series_df_path)

#     os.makedirs(tmp_file_path, exist_ok=True)
#     split_data(series_df, tmp_file_path, split_num)
#     del series_df
#     gc.collect()
#     for idx in range(split_num):
#         print("split idx:", idx)
#         series_df = pd.read_parquet(
#             os.path.join(tmp_file_path, f"series_df_split_{idx}.parquet"))
#         series_df = preprocess_input(series_df)
#         series_df = set_seriesdatekey(series_df)
#         series_df = label_encode_series_date_key(series_df)
#         key_df = series_df[["series_date_key",
#                             "series_date_key_str"]].drop_duplicates()
#         key_df = key_df.reset_index(drop=True)
#         if len(key_df) == 0:
#             # subのときは通らないはず
#             print("[Warning] key_df is empty")
#             sub_df_split = pd.DataFrame({
#                 "series_id": [],
#                 "step": [],
#                 "event": [],
#                 "score": []
#             })
#             sub_df_split_path = os.path.join(tmp_file_path,
#                                              f"sub_df_split_{idx}.csv")
#             sub_df_split.to_csv(sub_df_split_path, index=False)
#             continue

#         key_df["series_id"], key_df["date"] = key_df[
#             "series_date_key_str"].str.split("_", expand=True)
#         key_df = key_df.drop(columns=["series_date_key_str"], axis=1)

#         pred_df = series_df[["series_id", "series_date_key", "step"]].copy()
#         for fold in CFG.folds:
#             pred_df[f"class_pred_fold{fold}"] = -1
#             print(f"-- fold{fold} inference start --")
#             # set model & learning fn
#             model = get_model(CFG)
#             model_path = os.path.join(exp_dir, exp_name,
#                                       f"fold{fold}_model.pth")
#             print("model loading", model_path)
#             model.load_state_dict(torch.load(model_path))
#             model = model.to(CFG.device)
#             # separate train/valid data
#             infer_loader = get_loader(CFG, key_df, series_df, mode="infer")
#             infer_preds, infer_input_dict = predict(CFG, model, infer_loader)
#             print(f"split[{idx}] fold[{fold}] prediction finished.")
#             pred_df = get_pred_df(
#                 infer_input_dict,
#                 infer_preds,
#                 pred_df,
#                 fold,
#             )
#             del infer_preds, infer_input_dict, infer_loader, model
#             gc.collect()
#             torch.cuda.empty_cache()

#         pred_df["class_pred"] = pred_df[[
#             f"class_pred_fold{fold}" for fold in CFG.folds
#         ]].mean(axis=1)
#         pred_df = pred_df.drop(
#             columns=[f"class_pred_fold{fold}" for fold in CFG.folds])
#         pred_df = detect_event_from_classpred(pred_df)
#         pred_df.to_csv(os.path.join(tmp_file_path, f"pred_df_split_{idx}.csv"))

#         sub_df_split = make_submission_df(pred_df)
#         sub_df_split = sub_df_split.drop("event_pred", axis=1)
#         sub_df_split = sub_df_split.reset_index(drop=True)
#         print(f"sub_df_split_{idx} data num : {len(sub_df_split)}")
#         sub_df_split_path = os.path.join(tmp_file_path,
#                                          f"sub_df_split_{idx}.csv")
#         sub_df_split.to_csv(sub_df_split_path, index=False)
#         print(f"sub_df_split is saved as {sub_df_split_path}")
#         del sub_df_split, pred_df, series_df, key_df
#         gc.collect()
#         torch.cuda.empty_cache()

#     sub_df = pd.DataFrame()
#     for idx in range(split_num):
#         if idx == 0:
#             sub_df = pd.read_csv(
#                 os.path.join(tmp_file_path, f"sub_df_split_{idx}.csv"))
#         else:
#             sub_df_tmp = pd.read_csv(
#                 os.path.join(tmp_file_path, f"sub_df_split_{idx}.csv"))
#             if len(sub_df_tmp) > 0:
#                 sub_df = pd.concat([sub_df, sub_df_tmp], axis=0)

#     del sub_df_tmp
#     gc.collect()
#     torch.cuda.empty_cache()
#     sub_df = sub_df.reset_index(drop=True)
#     print(f"sub_df data num : {len(sub_df)}")
#     return sub_df

# if __name__ == "__main__":
#     exp_dir = "/kaggle/working/"
#     exp_name = "baseline_alldata_allfold_000"
#     series_df_path = (
#         "/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet"
#     )

#     tmp_file_path = "/kaggle/working/submission"
#     sub_path = "/kaggle/working/submission"

#     # test_df = pd.read_parquet(series_df_path)
#     # print(test_df.head())
#     # print(test_df["series_id"].unique())
#     import yaml  # type: ignore

#     config_path = os.path.join(exp_dir, exp_name, "config.yaml")

#     config = yaml.load(open(config_path, "r"), Loader=yaml.SafeLoader)
#     print(config)
#     config = argparse.Namespace(**config)
#     print(config)
#     inference(
#         config,
#         exp_dir,
#         exp_name,
#         series_df_path,
#         tmp_file_path,
#         sub_path,
#     )
