import gc
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd  # type: ignore
import polars as pl
import scipy.signal as signal
import torch
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(SRC_DIR, "dss_utils"))
sys.path.append(os.path.join(SRC_DIR, "data"))
sys.path.append(os.path.join(SRC_DIR, "model"))

from dss_dataloader import get_loader
from dss_metrics import score
from dss_model import get_model
from logger import AverageMeter
from logger import ProgressLogger
from logger import WandbLogger
from logger import init_logger
from losses import get_class_criterion
from scheduler import get_optimizer
from scheduler import get_scheduler


def seed_everything(seed=42):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def train_fn(CFG, epoch, model, train_loader, class_criterion, optimizer, LOGGER):
    model.train()
    prog_loagger = ProgressLogger(
        data_num=len(train_loader),
        print_freq=CFG.print_freq,
        logger=LOGGER,
    )

    losses = AverageMeter()
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        inputs = inputs.to(CFG.device, non_blocking=True).float()
        targets = targets.to(CFG.device, non_blocking=True).float()

        preds = model(inputs)
        loss = class_criterion(preds, targets)
        # preds = torch.sigmoid(preds)  # sigmoidいらない場合はこれを消す
        loss.backward()
        losses.update(loss.item(), CFG.batch_size)
        optimizer.step()  # モデル更新
        optimizer.zero_grad()  # 勾配の初期化
        prog_loagger.log_progress(epoch, batch_idx, losses)

    del inputs, preds, targets
    gc.collect()
    torch.cuda.empty_cache()
    return losses.avg


def get_valid_values_dict(
    event_values: torch.Tensor,
    validation_dict: dict,
    mode: str = "preds",
) -> dict:
    event_values = event_values.detach().cpu().numpy()
    # event_values = event_values.astype(np.float16)  # type: ignore
    if len(validation_dict[f"onset_{mode}"]) == 0:
        validation_dict[f"sleep_{mode}"] = event_values[:, 0, :]
        validation_dict[f"onset_{mode}"] = event_values[:, 1, :]
        validation_dict[f"wakeup_{mode}"] = event_values[:, 2, :]
    else:
        validation_dict[f"sleep_{mode}"] = np.concatenate(
            [validation_dict[f"sleep_{mode}"], event_values[:, 0, :]], axis=0
        )
        validation_dict[f"onset_{mode}"] = np.concatenate(
            [validation_dict[f"onset_{mode}"], event_values[:, 1, :]], axis=0
        )
        validation_dict[f"wakeup_{mode}"] = np.concatenate(
            [validation_dict[f"wakeup_{mode}"], event_values[:, 2, :]], axis=0
        )
    return validation_dict


def concat_valid_input_info(valid_input_info: dict, input_info: dict) -> dict:
    if len(valid_input_info["series_date_key"]) == 0:
        valid_input_info["series_date_key"] = input_info["series_date_key"]
        valid_input_info["start_step"] = input_info["start_step"]
        valid_input_info["end_step"] = input_info["end_step"]
    else:
        valid_input_info["series_date_key"] = np.concatenate(
            [valid_input_info["series_date_key"], input_info["series_date_key"]], axis=0
        )
        valid_input_info["start_step"] = np.concatenate(
            [valid_input_info["start_step"], input_info["start_step"]], axis=0
        )
        valid_input_info["end_step"] = np.concatenate(
            [valid_input_info["end_step"], input_info["end_step"]], axis=0
        )
    return valid_input_info


def valid_fn(CFG, epoch, model, valid_loader, criterion, LOGGER):
    model.eval()

    losses = AverageMeter()
    losses = AverageMeter()
    prog_loagger = ProgressLogger(
        data_num=len(valid_loader),
        print_freq=CFG.print_freq,
        logger=LOGGER,
        mode="valid",
    )
    valid_predictions = {
        "sleep_preds": np.empty(0),
        "onset_preds": np.empty(0),
        "wakeup_preds": np.empty(0),
    }
    valid_targets = {
        "sleep_targets": np.empty(0),
        "onset_targets": np.empty(0),
        "wakeup_targets": np.empty(0),
    }
    valid_input_info = {"series_date_key": [], "start_step": [], "end_step": []}

    for batch_idx, (inputs, targets, input_info_dict) in enumerate(valid_loader):
        inputs = inputs.to(CFG.device, non_blocking=True).float()
        targets = targets.to(CFG.device, non_blocking=True).float()
        with torch.no_grad():
            preds = model(inputs)
            loss = criterion(preds, targets)
            preds = torch.sigmoid(preds)
        losses.update(loss.item(), CFG.batch_size)
        prog_loagger.log_progress(epoch, batch_idx, losses)

        valid_predictions = get_valid_values_dict(
            preds, valid_predictions, mode="preds"
        )
        valid_targets = get_valid_values_dict(targets, valid_targets, mode="targets")
        valid_input_info = concat_valid_input_info(valid_input_info, input_info_dict)

    del inputs, preds, targets
    gc.collect()
    torch.cuda.empty_cache()
    return (
        valid_predictions,
        valid_targets,
        valid_input_info,
        losses.avg,
    )


def get_event_oof_df(
    valid_input_info_dict: dict,
    valid_preds_dict: dict,
    valid_targets_dict: dict,
    oof_df_fold: pd.DataFrame,
) -> pd.DataFrame:
    pred_target_col_list = [
        "sleep_pred",
        "onset_pred",
        "wakeup_pred",
        "sleep_target",
        "onset_target",
        "wakeup_target",
    ]
    drop_col_list = [col for col in pred_target_col_list if col in oof_df_fold.columns]
    oof_df_fold = oof_df_fold.drop(drop_col_list, axis=1)
    start_time = time.time()
    sleep_pred_list, onset_pred_list, wakeup_pred_list = [], [], []
    sleep_target_list, onset_target_list, wakeup_target_list = [], [], []
    steps_list = []
    series_date_key_list = []
    print("creating oof_df", end=" ... ")
    for idx, (series_date_key, start_step, end_step) in enumerate(
        zip(
            valid_input_info_dict["series_date_key"],
            valid_input_info_dict["start_step"],
            valid_input_info_dict["end_step"],
        )
    ):
        # preds targets shape: [batch, ch, data_length]
        sleep_pred = valid_preds_dict["sleep_preds"][idx]
        onset_pred = valid_preds_dict["onset_preds"][idx]
        wakeup_pred = valid_preds_dict["wakeup_preds"][idx]
        sleep_target = valid_targets_dict["sleep_targets"][idx]
        onset_target = valid_targets_dict["onset_targets"][idx]
        wakeup_target = valid_targets_dict["wakeup_targets"][idx]

        steps = range(start_step, end_step + 1, 1)
        series_date_data_num = len(steps)
        if series_date_data_num < onset_pred.shape[0]:
            sleep_pred = sleep_pred[:series_date_data_num]
            onset_pred = onset_pred[:series_date_data_num]
            wakeup_pred = wakeup_pred[:series_date_data_num]
            sleep_target = sleep_target[:series_date_data_num]
            onset_target = onset_target[:series_date_data_num]
            wakeup_target = wakeup_target[:series_date_data_num]
        elif series_date_data_num > onset_pred.shape[0]:
            padding_num = series_date_data_num - onset_pred.shape[0]
            sleep_pred = np.concatenate([sleep_pred, -1 * np.ones(padding_num)], axis=0)
            onset_pred = np.concatenate([onset_pred, -1 * np.ones(padding_num)], axis=0)
            wakeup_pred = np.concatenate(
                [wakeup_pred, -1 * np.ones(padding_num)], axis=0
            )
            sleep_target = np.concatenate(
                [sleep_target, -1 * np.ones(padding_num)], axis=0
            )
            onset_target = np.concatenate(
                [onset_target, -1 * np.ones(padding_num)], axis=0
            )
            wakeup_target = np.concatenate(
                [wakeup_target, -1 * np.ones(padding_num)], axis=0
            )
        else:
            pass
        if not (onset_pred.shape[0] == onset_target.shape[0]) or not (
            onset_pred.shape[0] == len(steps)
        ):
            print("len(event_pred)", onset_pred.shape[0])
            print("len(event_target)", onset_target.shape[0])
            print("len(steps)", len(steps))
            raise ValueError("preds and targets length is not same")
        sleep_pred_list.extend(sleep_pred)
        onset_pred_list.extend(onset_pred)
        wakeup_pred_list.extend(wakeup_pred)
        sleep_target_list.extend(sleep_target)
        onset_target_list.extend(onset_target)
        wakeup_target_list.extend(wakeup_target)
        steps_list.extend(steps)
        series_date_key_list.extend([series_date_key] * len(steps))
    oof_pred_target_df = pd.DataFrame(
        {
            "series_date_key": series_date_key_list,
            "step": steps_list,
            "sleep_pred": sleep_pred_list,
            "onset_pred": onset_pred_list,
            "wakeup_pred": wakeup_pred_list,
            "sleep_target": sleep_target_list,
            "onset_target": onset_target_list,
            "wakeup_target": wakeup_target_list,
        }
    )
    oof_df_fold = pd.merge(
        oof_df_fold, oof_pred_target_df, on=["series_date_key", "step"], how="left"
    )
    # sleep_pred, onset_pred, wakeup_predのnanを線形補間
    oof_df_fold["sleep_pred"] = oof_df_fold["sleep_pred"].interpolate()
    oof_df_fold["onset_pred"] = oof_df_fold["onset_pred"].interpolate()
    oof_df_fold["wakeup_pred"] = oof_df_fold["wakeup_pred"].interpolate()

    elapsed = int(time.time() - start_time) / 60
    print(f" >> oof_df created. elapsed time: {elapsed:.2f} min")
    return oof_df_fold


def get_event_downsample_oof_df(
    valid_input_info_dict: dict,
    valid_preds_dict: dict,
    valid_targets_dict: dict,
    oof_df_fold: pd.DataFrame,
    config,
) -> pd.DataFrame:
    start_time = time.time()
    pred_target_col_list = [
        "sleep_pred",
        "onset_pred",
        "wakeup_pred",
        "sleep_target",
        "onset_target",
        "wakeup_target",
    ]
    drop_col_list = [col for col in pred_target_col_list if col in oof_df_fold.columns]
    oof_df_fold = oof_df_fold.drop(drop_col_list, axis=1)
    print("creating oof_df", end=" ... ")
    sleep_pred_list, onset_pred_list, wakeup_pred_list = [], [], []
    sleep_target_list, onset_target_list, wakeup_target_list = [], [], []
    steps_list = []
    series_date_key_list = []
    for idx, (series_date_key, start_step, end_step) in enumerate(
        zip(
            valid_input_info_dict["series_date_key"],
            valid_input_info_dict["start_step"],
            valid_input_info_dict["end_step"],
        )
    ):
        # preds targets shape: [batch, ch, data_length]
        sleep_pred = valid_preds_dict["sleep_preds"][idx]
        onset_pred = valid_preds_dict["onset_preds"][idx]
        wakeup_pred = valid_preds_dict["wakeup_preds"][idx]
        sleep_target = valid_targets_dict["sleep_targets"][idx]
        onset_target = valid_targets_dict["onset_targets"][idx]
        wakeup_target = valid_targets_dict["wakeup_targets"][idx]
        steps = range(start_step, end_step + 1, config.downsample_rate)
        series_date_data_num = len(steps)

        if series_date_data_num < onset_pred.shape[0]:
            sleep_pred = sleep_pred[:series_date_data_num]
            onset_pred = onset_pred[:series_date_data_num]
            wakeup_pred = wakeup_pred[:series_date_data_num]
            sleep_target = sleep_target[:series_date_data_num]
            onset_target = onset_target[:series_date_data_num]
            wakeup_target = wakeup_target[:series_date_data_num]
        elif series_date_data_num > onset_pred.shape[0]:
            padding_num = series_date_data_num - onset_pred.shape[0]
            sleep_pred = np.concatenate([sleep_pred, -1 * np.ones(padding_num)], axis=0)
            onset_pred = np.concatenate([onset_pred, -1 * np.ones(padding_num)], axis=0)
            wakeup_pred = np.concatenate(
                [wakeup_pred, -1 * np.ones(padding_num)], axis=0
            )
            sleep_target = np.concatenate(
                [sleep_target, -1 * np.ones(padding_num)], axis=0
            )
            onset_target = np.concatenate(
                [onset_target, -1 * np.ones(padding_num)], axis=0
            )
            wakeup_target = np.concatenate(
                [wakeup_target, -1 * np.ones(padding_num)], axis=0
            )
        else:
            pass
        if not (onset_pred.shape[0] == onset_target.shape[0]) or not (
            onset_pred.shape[0] == len(steps)
        ):
            print("len(event_pred)", onset_pred.shape[0])
            print("len(event_target)", onset_target.shape[0])
            print("len(steps)", len(steps))
            raise ValueError("preds and targets length is not same")
        sleep_pred_list.extend(sleep_pred)
        onset_pred_list.extend(onset_pred)
        wakeup_pred_list.extend(wakeup_pred)
        sleep_target_list.extend(sleep_target)
        onset_target_list.extend(onset_target)
        wakeup_target_list.extend(wakeup_target)
        steps_list.extend(steps)
        series_date_key_list.extend([series_date_key] * len(steps))
    oof_pred_target_df = pd.DataFrame(
        {
            "series_date_key": series_date_key_list,
            "step": steps_list,
            "sleep_pred": sleep_pred_list,
            "onset_pred": onset_pred_list,
            "wakeup_pred": wakeup_pred_list,
            "sleep_target": sleep_target_list,
            "onset_target": onset_target_list,
            "wakeup_target": wakeup_target_list,
        }
    )
    print("merging oof_df")
    oof_df_fold = pd.merge(
        oof_df_fold, oof_pred_target_df, on=["series_date_key", "step"], how="left"
    )
    oof_df_fold["sleep_pred"] = oof_df_fold["sleep_pred"].interpolate()
    oof_df_fold["onset_pred"] = oof_df_fold["onset_pred"].interpolate()
    oof_df_fold["wakeup_pred"] = oof_df_fold["wakeup_pred"].interpolate()
    elapsed = int(time.time() - start_time) / 60
    print(f" >> oof_df created. elapsed time: {elapsed:.2f} min")
    return oof_df_fold


def low_path_filter(wave: np.ndarray, hour: int, fe: int = 60, n: int = 3):
    fs = 12 * 60 * hour
    nyq = fs / 2.0
    b, a = signal.butter(1, fe / nyq, btype="low")
    for i in range(0, n):
        wave = signal.filtfilt(b, a, wave)
    return wave


def make_submission_from_eventdf(
    df, threshold=0.001, distance=70, low_pass_filter_hour=5
):
    df = df[["series_id", "step", "onset_pred", "wakeup_pred"]].copy()
    df = df.rename(
        columns={"sleep_pred": "sleep", "onset_pred": "onset", "wakeup_pred": "wakeup"}
    )
    df["step"] = df["step"].astype(np.float64)
    unique_series_ids = df["series_id"].unique()
    records = []
    for series_id in unique_series_ids:
        this_series_preds = df[df["series_id"] == series_id][["onset", "wakeup"]].values
        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            # ローパスフィルタ
            # this_event_preds = low_path_filter(
            #     this_event_preds, hour=low_pass_filter_hour
            # )  # (seq,)
            # ガウシアンフィルタ(LPFとあんまり変わらない。両方かけるとcv下がった)
            # this_event_preds = gaussian_filter1d(this_event_preds, sigma=10)

            steps = find_peaks(this_event_preds, height=threshold, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score_ in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score_,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    sub_df = sub_df.to_pandas()
    print("detected event num: ", len(sub_df))
    return sub_df


def get_oof_df_and_fold_score(
    CFG,
    LOGGER,
    input_info_dict_list,
    valid_predictions,
    valid_targets,
    valid_key_df,
    oof_df_fold,
    event_df,
    fold,
    epoch,
):
    if "downsample" in CFG.model_type:
        oof_df_fold = get_event_downsample_oof_df(
            input_info_dict_list, valid_predictions, valid_targets, oof_df_fold, CFG
        )
    else:
        oof_df_fold = get_event_oof_df(
            input_info_dict_list, valid_predictions, valid_targets, oof_df_fold
        )
    oof_dir = os.path.join(CFG.output_dir, "_oof", CFG.exp_name)
    oof_df_fold_path = os.path.join(oof_dir, f"oof_df_fold{fold}.parquet")
    print("save oof_df to ", oof_df_fold_path)
    oof_df_fold.to_parquet(oof_df_fold_path)
    LOGGER.info(f"fold{fold} event detected.")
    oof_scoring_df = make_submission_from_eventdf(oof_df_fold)
    LOGGER.info(f"fold{fold} submission df created.")
    event_df_fold = event_df[event_df["series_id"].isin(valid_key_df["series_id"])]
    event_df_fold = event_df_fold[event_df_fold["step"].notnull()]
    oof_score = score(event_df_fold, oof_scoring_df)
    LOGGER.info(f"fold{fold} epoch {epoch} oof score: {oof_score:.4f}")
    return oof_df_fold, oof_score


def get_key_df(series_df: pd.DataFrame) -> pd.DataFrame:
    key_df = series_df[["series_date_key", "series_date_key_str"]].drop_duplicates()
    key_df = key_df.reset_index(drop=True)
    key_df["series_id"], key_df["date"] = (
        key_df["series_date_key_str"].str.split("_", 1).str
    )
    key_df = key_df.drop(columns=["series_date_key_str"], axis=1)
    return key_df


def eventdet_training_loop(CFG, LOGGER):
    # key_df = pd.read_csv(CFG.key_df)
    LOGGER.info("loading series_df")
    series_df = pd.read_parquet(CFG.series_df)
    key_df = get_key_df(series_df)
    train_event_df = pl.read_csv(CFG.event_df)
    train_event_df = (
        train_event_df.pivot(
            index=["series_id", "night"], columns="event", values="step"
        )
        .drop_nulls()
        .to_pandas()
    )
    LOGGER.info("series_df data num : {}".format(len(series_df)))
    LOGGER.info("key_df data num : {}".format(len(key_df)))
    nan_event_keys = np.load("/kaggle/working/nan_event_keys.npy")
    LOGGER.info("nan event keys")
    LOGGER.info(nan_event_keys)
    event_df = pd.read_csv(os.path.join(CFG.event_df))
    event_df = event_df.dropna(subset=["step"])
    # oof_df = pd.DataFrame()
    oof_dir = os.path.join(CFG.output_dir, "_oof", CFG.exp_name)
    os.makedirs(oof_dir, exist_ok=True)
    oof_score_list = []
    for fold in CFG.folds:
        LOGGER.info(f"-- fold{fold} event detection training start --")
        wandb_logger = WandbLogger(CFG)
        wandb_log_dict = {}
        # set model & learning fn
        model = get_model(CFG)
        model = model.to(CFG.device)
        class_criterion = get_class_criterion(CFG)
        optimizer = get_optimizer(model, CFG)
        scheduler = get_scheduler(optimizer, CFG)

        # training
        start_time = time.time()
        LOGGER.info(f"fold[{fold}] loading train/valid data")
        train_series_df = series_df[series_df["fold"] != fold]
        # nan event keyの除外
        train_series_df = train_series_df[
            ~train_series_df["series_date_key"].isin(nan_event_keys)
        ].reset_index(drop=True)
        train_key_df = get_key_df(train_series_df)
        valid_series_df = series_df[series_df["fold"] == fold]
        valid_key_df = get_key_df(valid_series_df)
        train_key_num = len(train_key_df)
        valid_key_num = len(valid_key_df)
        LOGGER.info(f"fold[{fold}] train data key num: {train_key_num}")
        LOGGER.info(f"fold[{fold}] valid data key num: {valid_key_num}")
        train_loader = get_loader(
            CFG, train_key_df, train_series_df, event_df=train_event_df, mode="train"
        )
        valid_loader = get_loader(
            CFG, valid_key_df, valid_series_df, event_df=train_event_df, mode="valid"
        )
        LOGGER.info(f"fold[{fold}] get_loader finished")

        oof_df_fold = valid_series_df.copy()
        fold_best_score = 0.0
        for epoch in range(0, CFG.n_epoch):
            LOGGER.info(f"- epoch:{epoch} -")
            train_loss_avg = train_fn(
                CFG,
                epoch,
                model,
                train_loader,
                class_criterion,
                optimizer,
                LOGGER,
            )
            (
                valid_predictions,
                valid_targets,
                input_info_dict_list,
                valid_loss_avg,
            ) = valid_fn(
                CFG,
                epoch,
                model,
                valid_loader,
                class_criterion,
                LOGGER,
            )
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(epoch + 1)
            oof_df_fold, oof_score = get_oof_df_and_fold_score(
                CFG,
                LOGGER,
                input_info_dict_list,
                valid_predictions,
                valid_targets,
                valid_key_df,
                oof_df_fold,
                event_df,
                fold,
                epoch,
            )
            if oof_score > fold_best_score:
                fold_best_score = oof_score
                model_path = os.path.join(CFG.exp_dir, f"fold{fold}_best_model.pth")
                torch.save(model.state_dict(), model_path)
                oof_df_fold_path = os.path.join(
                    oof_dir, f"fold{fold}_best_oof_df.parquet"
                )
                oof_df_fold.to_parquet(oof_df_fold_path)
            elapsed = int(time.time() - start_time) / 60
            log_str = f"FOLD:{fold}, Epoch:{epoch}"
            log_str += f", train:{train_loss_avg:.4f}, valid:{valid_loss_avg:.4f}"
            log_str += f", lr:{lr:.6f}, elapsed time:{elapsed:.2f} min"
            LOGGER.info(log_str)
            # competition scoreを計算するように変更する

            # wandb log
            wandb_log_dict[f"train_loss/fold{fold}"] = train_loss_avg
            wandb_log_dict[f"valid_loss/fold{fold}"] = valid_loss_avg
            wandb_log_dict[f"lr/fold{fold}"] = lr
            wandb_logger.log_progress(epoch, wandb_log_dict)
        oof_score_list.append(fold_best_score)
        # model save
        model_path = os.path.join(CFG.exp_dir, f"fold{fold}_model.pth")
        torch.save(model.state_dict(), model_path)
        if len(oof_df_fold["series_date_key"].unique()) != len(
            valid_series_df["series_date_key"].unique()
        ):
            raise ValueError("oof data key num is not same")
        del (
            model,
            train_loader,
            valid_loader,
            train_series_df,
            valid_series_df,
            train_key_df,
        )
        gc.collect()
        torch.cuda.empty_cache()
    over_all_score_mean = np.mean(oof_score_list)
    LOGGER.info(f"overall oof score mean: {over_all_score_mean:.4f}")
    for fold, oof_score in enumerate(oof_score_list):
        LOGGER.info(f"fold{fold} best oof score: {oof_score:.4f}")
        wandb_logger.log_best_score(fold, oof_score)
    wandb_logger.log_overall_oofscore(over_all_score_mean)


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))

    class CFG:
        # exp
        exp_name = "event_det_debug"

        # directory
        input_dir = os.path.abspath(
            os.path.join(
                ROOT_DIR,
                "input",
            )
        )
        competition_dir = os.path.join(
            input_dir,
            "child-mind-institute-detect-sleep-states",
        )
        output_dir = os.path.abspath(os.path.join(ROOT_DIR, "working"))
        exp_dir = os.path.join(output_dir, exp_name)
        series_df = os.path.join("/kaggle/working/_oof/exp006_addlayer/oof_df.parquet")
        event_df = (
            "/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv"
        )
        # data
        # folds = [0, 1, 2, 3, 4]
        folds = [0]
        n_folds = 5
        num_workers = os.cpu_count()
        seed = 42
        group_key = "series_id"

        # model
        model_type = "event_detect"
        input_channels = 3
        output_channels = 2
        embedding_base_channels = 16
        ave_kernel_size = 301
        maxpool_kernel_size = 11

        class_loss_weight = 1.0
        event_loss_weight = 100.0

        # training
        # n_epoch = 5
        n_epoch = 1
        # batch_size = 32
        batch_size = 128
        # optimizer
        lr = 1e-3
        weight_decay = 1e-6
        # scheduler
        # T_0 = 10
        T_0 = n_epoch
        T_mult = 1
        eta_min = 1e-9

        # log setting
        print_freq = 100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wandb_available = False

    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.exp_dir, exist_ok=True)

    LOGGER = init_logger(log_file=os.path.join(CFG.output_dir, "check.log"))
    LOGGER.info(f"using device: {CFG.device}")
    eventdet_training_loop(CFG, LOGGER)
