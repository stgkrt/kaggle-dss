import gc
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd  # type: ignore
import torch
import torch.nn as nn

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

        preds = torch.sigmoid(preds)  # sigmoidいらない場合はこれを消す
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
        validation_dict[f"onset_{mode}"] = event_values[:, 0, :]
        validation_dict[f"wakeup_{mode}"] = event_values[:, 1, :]
        validation_dict[f"class_{mode}"] = event_values[:, 2, :]
    else:
        validation_dict[f"onset_{mode}"] = np.concatenate(
            [validation_dict[f"onset_{mode}"], event_values[:, 0, :]], axis=0
        )
        validation_dict[f"wakeup_{mode}"] = np.concatenate(
            [validation_dict[f"wakeup_{mode}"], event_values[:, 1, :]], axis=0
        )
        validation_dict[f"class_{mode}"] = np.concatenate(
            [validation_dict[f"class_{mode}"], event_values[:, 2, :]], axis=0
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
        "onset_preds": np.empty(0),
        "wakeup_preds": np.empty(0),
        "class_preds": np.empty(0),
    }
    valid_targets = {
        "onset_targets": np.empty(0),
        "wakeup_targets": np.empty(0),
        "class_targets": np.empty(0),
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
    start_time = time.time()
    print("creating oof_df", end=" ... ")
    for idx, (series_date_key, start_step, end_step) in enumerate(
        zip(
            valid_input_info_dict["series_date_key"],
            valid_input_info_dict["start_step"],
            valid_input_info_dict["end_step"],
        )
    ):
        # preds targets shape: [batch, ch, data_length]
        onset_pred = valid_preds_dict["onset_preds"][idx]
        wakeup_pred = valid_preds_dict["wakeup_preds"][idx]
        class_pred = valid_preds_dict["class_preds"][idx]
        onset_target = valid_targets_dict["onset_targets"][idx]
        wakeup_target = valid_targets_dict["wakeup_targets"][idx]
        class_target = valid_targets_dict["class_targets"][idx]
        data_condition = (
            (oof_df_fold["series_date_key"] == series_date_key)
            & (start_step <= oof_df_fold["step"])
            & (oof_df_fold["step"] <= end_step + 1)
        )
        series_date_data_num = len((oof_df_fold[data_condition]))
        steps = range(start_step, start_step + series_date_data_num, 1)
        if series_date_data_num < onset_pred.shape[0]:
            onset_pred = onset_pred[:series_date_data_num]
            wakeup_pred = wakeup_pred[:series_date_data_num]
            class_pred = class_pred[:series_date_data_num]
            onset_target = onset_target[:series_date_data_num]
            wakeup_target = wakeup_target[:series_date_data_num]
            class_target = class_target[:series_date_data_num]
        elif series_date_data_num > onset_pred.shape[0]:
            padding_num = series_date_data_num - onset_pred.shape[0]
            onset_pred = np.concatenate([onset_pred, -1 * np.ones(padding_num)], axis=0)
            wakeup_pred = np.concatenate(
                [wakeup_pred, -1 * np.ones(padding_num)], axis=0
            )
            class_pred = np.concatenate([class_pred, -1 * np.ones(padding_num)], axis=0)
            onset_target = np.concatenate(
                [onset_target, -1 * np.ones(padding_num)], axis=0
            )
            wakeup_target = np.concatenate(
                [wakeup_target, -1 * np.ones(padding_num)], axis=0
            )
            class_target = np.concatenate(
                [class_target, -1 * np.ones(padding_num)], axis=0
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
        oof_df_fold.loc[data_condition, "onset_pred"] = onset_pred
        oof_df_fold.loc[data_condition, "wakeup_pred"] = wakeup_pred
        oof_df_fold.loc[data_condition, "class_pred"] = class_pred
        oof_df_fold.loc[data_condition, "onset_target"] = onset_target
        oof_df_fold.loc[data_condition, "wakeup_target"] = wakeup_target
        oof_df_fold.loc[data_condition, "class_target"] = class_target
    elapsed = int(time.time() - start_time) / 60
    print(f" >> oof_df created. elapsed time: {elapsed:.2f} min")
    return oof_df_fold


def get_event_downsample_oof_df(
    valid_input_info_dict: dict,
    valid_preds_dict: dict,
    valid_targets_dict: dict,
    oof_df_fold: pd.DataFrame,
) -> pd.DataFrame:
    start_time = time.time()
    if "onset_pred" in oof_df_fold.columns:
        oof_df_fold = oof_df_fold.drop(["onset_pred"], axis=1)
    if "wakeup_pred" in oof_df_fold.columns:
        oof_df_fold = oof_df_fold.drop(["wakeup_pred"], axis=1)
    if "class_pred" in oof_df_fold.columns:
        oof_df_fold = oof_df_fold.drop(["class_pred"], axis=1)
    print("creating oof_df", end=" ... ")
    onset_pred_list, wakeup_pred_list, class_pred_list = [], [], []
    onset_target_list, wakeup_target_list, class_target_list = [], [], []

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
        onset_pred = valid_preds_dict["onset_preds"][idx]
        wakeup_pred = valid_preds_dict["wakeup_preds"][idx]
        class_pred = valid_preds_dict["class_preds"][idx]
        onset_target = valid_targets_dict["onset_targets"][idx]
        wakeup_target = valid_targets_dict["wakeup_targets"][idx]
        class_target = valid_targets_dict["class_targets"][idx]
        steps = range(start_step, end_step + 1, 12)
        series_date_data_num = len(steps)
        if series_date_data_num < onset_pred.shape[0]:
            onset_pred = onset_pred[:series_date_data_num]
            wakeup_pred = wakeup_pred[:series_date_data_num]
            class_pred = class_pred[:series_date_data_num]
            onset_target = onset_target[:series_date_data_num]
            wakeup_target = wakeup_target[:series_date_data_num]
            class_target = class_target[:series_date_data_num]
        elif series_date_data_num > onset_pred.shape[0]:
            padding_num = series_date_data_num - onset_pred.shape[0]
            onset_pred = np.concatenate([onset_pred, -1 * np.ones(padding_num)], axis=0)
            class_pred = np.concatenate([class_pred, -1 * np.ones(padding_num)], axis=0)
            wakeup_pred = np.concatenate(
                [wakeup_pred, -1 * np.ones(padding_num)], axis=0
            )
            onset_target = np.concatenate(
                [onset_target, -1 * np.ones(padding_num)], axis=0
            )
            wakeup_target = np.concatenate(
                [wakeup_target, -1 * np.ones(padding_num)], axis=0
            )
            class_target = np.concatenate(
                [class_target, -1 * np.ones(padding_num)], axis=0
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
        onset_pred_list.extend(onset_pred)
        wakeup_pred_list.extend(wakeup_pred)
        class_pred_list.extend(class_pred)
        onset_target_list.extend(onset_target)
        wakeup_target_list.extend(wakeup_target)
        class_target_list.extend(class_target)
        steps_list.extend(steps)
        series_date_key_list.extend([series_date_key] * len(steps))
    oof_pred_target_df = pd.DataFrame(
        {
            "series_date_key": series_date_key_list,
            "step": steps_list,
            "onset_pred": onset_pred_list,
            "wakeup_pred": wakeup_pred_list,
            "class_pred": class_pred_list,
            "onset_target": onset_target_list,
            "wakeup_target": wakeup_target_list,
            "class_target": class_target_list,
        }
    )
    print("merging oof_df")
    oof_df_fold = pd.merge(
        oof_df_fold, oof_pred_target_df, on=["series_date_key", "step"], how="left"
    )
    oof_df_fold["onset_pred"] = oof_df_fold["onset_pred"].fillna(0)
    oof_df_fold["wakeup_pred"] = oof_df_fold["wakeup_pred"].fillna(0)
    oof_df_fold["class_pred"] = oof_df_fold["class_pred"].fillna(-1)
    elapsed = int(time.time() - start_time) / 60
    print(f" >> oof_df created. elapsed time: {elapsed:.2f} min")
    return oof_df_fold


def make_submission_from_eventdf(df, threshold=0.1, max_pool_size=3):
    df = df[["series_id", "step", "onset_pred", "wakeup_pred"]].copy()
    df["step"] = df["step"].astype(np.float64)
    max_pool = nn.MaxPool1d(
        max_pool_size, stride=1, padding=int((max_pool_size - 1) / 2)
    )
    onset_pred = (
        max_pool(torch.tensor(df["onset_pred"].values).unsqueeze(0)).squeeze(0).numpy()
    )
    wakeup_pred = (
        max_pool(torch.tensor(df["wakeup_pred"].values).unsqueeze(0)).squeeze(0).numpy()
    )
    peak_mask = onset_pred == df["onset_pred"].values
    df["onset_pred"] = peak_mask * df["onset_pred"].values
    peak_mask = wakeup_pred == df["wakeup_pred"].values
    df["wakeup_pred"] = peak_mask * df["wakeup_pred"].values
    # onset_predが大きい場合-onset_predの値を入力し、
    # wakeup_predが大きい場合wakeup_predの値を入力する
    df["event_pred"] = np.where(
        df["onset_pred"].values > df["wakeup_pred"].values,
        -df["onset_pred"].values,
        df["wakeup_pred"].values,
    )
    # event_predがthreshold以上の場合、wakeup_predが大きい場合はwakeup、
    # onset_predが大きい場合はonsetとする
    df["event_score"] = df["event_pred"].apply(
        lambda x: 1 if x > threshold else -1 if x < -threshold else 0
    )
    df = df[df["event_score"] != 0].copy()
    df["event"] = df["event_score"].replace({1: "wakeup", -1: "onset"})
    df["score"] = df["event_pred"].apply(lambda x: np.clip(np.abs(x), 0.0, 1.0))
    df = df.drop(["event_pred", "onset_pred", "wakeup_pred", "event_score"], axis=1)
    return df


def get_key_df(series_df: pd.DataFrame) -> pd.DataFrame:
    key_df = series_df[["series_date_key", "series_date_key_str"]].drop_duplicates()
    key_df = key_df.reset_index(drop=True)
    key_df["series_id"], key_df["date"] = (
        key_df["series_date_key_str"].str.split("_", 1).str
    )
    key_df = key_df.drop(columns=["series_date_key_str"], axis=1)
    return key_df


def eventclass_training_loop(CFG, LOGGER):
    # key_df = pd.read_csv(CFG.key_df)
    LOGGER.info("loading series_df")
    series_df = pd.read_parquet(CFG.series_df)
    key_df = get_key_df(series_df)

    LOGGER.info("series_df data num : {}".format(len(series_df)))
    LOGGER.info("key_df data num : {}".format(len(key_df)))
    event_df = pd.read_csv(os.path.join(CFG.event_df))
    event_df = event_df[event_df["series_id"].isin(series_df["series_id"])]
    # oof_df = pd.DataFrame()
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
        train_key_df = get_key_df(train_series_df)
        if len(train_series_df["series_date_key"].unique()) != len(train_key_df):
            raise ValueError("train data key num is not same")
        valid_series_df = series_df[series_df["fold"] == fold]
        valid_key_df = get_key_df(valid_series_df)
        if len(valid_series_df["series_date_key"].unique()) != len(valid_key_df):
            raise ValueError("valid data key num is not same")
        train_key_num = len(train_key_df)
        valid_key_num = len(valid_key_df)
        LOGGER.info(f"fold[{fold}] train data key num: {train_key_num}")
        LOGGER.info(f"fold[{fold}] valid data key num: {valid_key_num}")
        if train_key_num + valid_key_num != len(key_df):
            raise ValueError("train/valid data key num is not same")
        train_loader = get_loader(CFG, train_key_df, train_series_df, mode="train")
        valid_loader = get_loader(CFG, valid_key_df, valid_series_df, mode="valid")
        LOGGER.info(f"fold[{fold}] get_loader finished")

        oof_df_fold = valid_series_df.copy()
        init_cols = [
            "event_pred",
            "event_target",
        ]
        oof_df_fold = oof_df_fold.assign(
            **{col: -1 * np.ones(len(oof_df_fold)) for col in init_cols}
        )

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

            lr = scheduler.get_last_lr()[0]
            scheduler.step()

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
        LOGGER.info(f"fold{fold} model saved.")
        if "downsample" in CFG.model_type:
            oof_df_fold = get_event_downsample_oof_df(
                input_info_dict_list,
                valid_predictions,
                valid_targets,
                oof_df_fold,
            )
        else:
            oof_df_fold = get_event_oof_df(
                input_info_dict_list,
                valid_predictions,
                valid_targets,
                oof_df_fold,
            )
        LOGGER.info(f"fold{fold} oof_df created.")

        oof_dir = os.path.join(CFG.output_dir, "_oof", CFG.exp_name)
        os.makedirs(oof_dir, exist_ok=True)
        oof_df_fold_path = os.path.join(oof_dir, f"oof_df_fold{fold}.parquet")
        print("save oof_df to ", oof_df_fold_path)
        oof_df_fold.to_parquet(oof_df_fold_path)
        LOGGER.info(f"fold{fold} event detected.")
        oof_fold_sub_df = make_submission_from_eventdf(oof_df_fold, threshold=0.1)
        LOGGER.info(f"fold{fold} submission df created.")
        event_df_fold = event_df[event_df["series_id"].isin(valid_key_df["series_id"])]
        event_df_fold = event_df_fold[event_df_fold["step"].notnull()]
        detected_event_rate = len(oof_fold_sub_df) / len(oof_df_fold) * 100
        LOGGER.info(f"detect event={len(oof_fold_sub_df)}({detected_event_rate:.4f}%)")
        LOGGER.info("scoring ...")
        scoring_time = time.time()
        oof_score = score(event_df_fold, oof_fold_sub_df)
        elapsed = int(time.time() - scoring_time) / 60
        LOGGER.info(f"scoring finished. elapsed time: {elapsed:.2f} min")
        oof_score_list.append(oof_score)
        LOGGER.info(f"fold{fold} oof score: {oof_score:.4f}")
        wandb_logger.log_oofscore(fold, oof_score)
        del oof_df_fold, oof_fold_sub_df, event_df_fold
        gc.collect()
    over_all_score_mean = np.mean(oof_score_list)
    LOGGER.info(f"overall oof score mean: {over_all_score_mean:.4f}")
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
    eventclass_training_loop(CFG, LOGGER)
