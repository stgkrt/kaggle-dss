import numpy as np
import torch
import torch.nn as nn


# 1step 0.5secで30minなら60*30=1800step
# metric的にいっぱい検出してもいい？とりあえず小さめで
def detect_event_from_classpred(df, N=300, maxpool_kernel_size=41, maxpool_stride=1):
    df = df.copy()

    # series_idでgroupbyして、class_predに対して対象の列のデータから前のN個の列までのデータの平均をとる
    df["class_pred_beforemean"] = df.groupby("series_id")["class_pred"].apply(
        lambda x: x.rolling(N, min_periods=1).mean()
    )
    df["class_pred_aftermean"] = df.groupby("series_id")["class_pred"].apply(
        lambda x: x[::-1].rolling(N, min_periods=1).mean()[::-1]
    )
    df["event_pred"] = df["class_pred_beforemean"] - df["class_pred_aftermean"]

    # 入力サイズと出力サイズが一致するようにpaddingを調整
    maxpool_padding = int((maxpool_kernel_size - maxpool_stride) / 2)
    # maxpoolしてピーク検出
    max_pooling = nn.MaxPool1d(
        maxpool_kernel_size, stride=maxpool_stride, padding=maxpool_padding
    )
    event_pred = df["event_pred"].values
    event_pred = torch.tensor(event_pred).unsqueeze(0)
    pooled_event_pred = max_pooling(np.abs(event_pred)).squeeze(0).numpy()
    event_pred = event_pred.squeeze(0).numpy()
    # peakのところだけ残すmaskを作成
    peak_event_pred_mask = np.where(pooled_event_pred == np.abs(event_pred), 1, 0)
    peak_event_pred = event_pred * peak_event_pred_mask
    df["event_pred"] = peak_event_pred
    df = df.drop(["class_pred_beforemean", "class_pred_aftermean"], axis=1)

    return df


def make_submission_df(df, threshold=0.01):
    df = df[["series_id", "step", "event_pred"]].copy()
    # thresholdより大きいときは1,-thresholdより小さいときは-1,それ以外は0
    df["event"] = df["event_pred"].apply(
        lambda x: 1 if x > threshold else -1 if x < -threshold else 0
    )
    df = df[df["event"] != 0].copy()
    df["event"] = df["event"].replace({1: "wakeup", -1: "onset"})
    df["score"] = df["event_pred"].apply(lambda x: np.clip(np.abs(x), 0.0, 1.0))
    return df
