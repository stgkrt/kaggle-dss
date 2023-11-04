import gc
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd  # type: ignore
import torch

warnings.filterwarnings("ignore")

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir))
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
from postprocess import detect_event_from_classpred
from postprocess import make_submission_df
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


def train_fn(CFG, epoch, model, train_loader, class_criterion, optimizer,
             LOGGER):
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
    class_values: torch.Tensor,
    validation_dict: dict,
    mode: str = "preds",
) -> dict:
    class_values = class_values.detach().cpu().numpy()
    class_values = class_values.astype(np.float16)  # type: ignore
    if len(validation_dict[f"class_{mode}"]) == 0:
        validation_dict[f"class_{mode}"] = class_values
    else:
        validation_dict[f"class_{mode}"] = np.concatenate(
            [validation_dict[f"class_{mode}"], class_values], axis=0)
    return validation_dict


def concat_valid_input_info(valid_input_info: dict, input_info: dict) -> dict:
    if len(valid_input_info["series_date_key"]) == 0:
        valid_input_info["series_date_key"] = input_info["series_date_key"]
        valid_input_info["start_step"] = input_info["start_step"]
        valid_input_info["end_step"] = input_info["end_step"]
    else:
        valid_input_info["series_date_key"] = np.concatenate([
            valid_input_info["series_date_key"], input_info["series_date_key"]
        ],
                                                             axis=0)
        valid_input_info["start_step"] = np.concatenate(
            [valid_input_info["start_step"], input_info["start_step"]], axis=0)
        valid_input_info["end_step"] = np.concatenate(
            [valid_input_info["end_step"], input_info["end_step"]], axis=0)
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
    valid_predictions = {"class_preds": np.empty(0)}
    valid_targets = {"class_targets": np.empty(0)}
    valid_input_info = {
        "series_date_key": [],
        "start_step": [],
        "end_step": []
    }

    for batch_idx, (inputs, targets,
                    input_info_dict) in enumerate(valid_loader):
        inputs = inputs.to(CFG.device, non_blocking=True).float()
        targets = targets.to(CFG.device, non_blocking=True).float()
        with torch.no_grad():
            preds = model(inputs)
            # preds = torch.sigmoid(preds)
            loss = criterion(preds, targets)
        losses.update(loss.item(), CFG.batch_size)
        prog_loagger.log_progress(epoch, batch_idx, losses)

        valid_predictions = get_valid_values_dict(preds,
                                                  valid_predictions,
                                                  mode="preds")
        valid_targets = get_valid_values_dict(targets,
                                              valid_targets,
                                              mode="targets")
        valid_input_info = concat_valid_input_info(valid_input_info,
                                                   input_info_dict)

    del inputs, preds, targets
    gc.collect()
    torch.cuda.empty_cache()
    return (
        valid_predictions,
        valid_targets,
        valid_input_info,
        losses.avg,
    )


def get_oof_df(
    valid_input_info_dict: dict,
    valid_preds_dict: dict,
    valid_targets_dict: dict,
    oof_df_fold: pd.DataFrame,
    config,
) -> pd.DataFrame:
    start_time = time.time()
    print("creating oof_df", end=" ... ")
    print("inf df", oof_df_fold[oof_df_fold["step"] == np.inf])
    for idx, (series_date_key, start_step, end_step) in enumerate(
            zip(
                valid_input_info_dict["series_date_key"],
                valid_input_info_dict["start_step"],
                valid_input_info_dict["end_step"],
            )):
        # preds targets shape: [batch, ch, data_length]
        class_pred = valid_preds_dict["class_preds"][idx]
        class_target = valid_targets_dict["class_targets"][idx]
        data_condition = ((oof_df_fold["series_date_key"] == series_date_key)
                          & (start_step <= oof_df_fold["step"])
                          & (oof_df_fold["step"] <= end_step + 1))
        series_date_data_num = len((oof_df_fold[data_condition]))
        # steps = range(start_step, end_step + 1, 1)
        if config.model_type == "downsample":
            steps = range(start_step, end_step + 1, 12)
        else:
            steps = range(start_step, start_step + series_date_data_num, 1)
        if series_date_data_num < len(class_pred[0]):
            class_pred = class_pred[0, :series_date_data_num]
            class_target = class_target[0, :series_date_data_num]
        elif series_date_data_num > len(class_pred[0]):
            padding_num = series_date_data_num - len(class_pred[0])
            class_pred = np.concatenate(
                [class_pred[0], -1 * np.ones(padding_num)], axis=0)
            class_target = np.concatenate(
                [class_target[0], -1 * np.ones(padding_num)], axis=0)
        else:
            class_pred = class_pred[0]
            class_target = class_target[0]
        if not (len(class_pred) == len(class_target)) or not (len(class_pred)
                                                              == len(steps)):
            print("len(class_pred)", len(class_pred))
            print("len(class_target)", len(class_target))
            print("len(steps)", len(steps))
            raise ValueError("preds and targets length is not same")
        oof_df_fold.loc[data_condition, "class_pred"] = class_pred
        oof_df_fold.loc[data_condition, "class_target"] = class_target

    elapsed = int(time.time() - start_time) / 60
    print(f" >> oof_df created. elapsed time: {elapsed:.2f} min")
    return oof_df_fold


def get_key_df(series_df: pd.DataFrame) -> pd.DataFrame:
    key_df = series_df[["series_date_key",
                        "series_date_key_str"]].drop_duplicates()
    key_df = key_df.reset_index(drop=True)
    key_df["series_id"], key_df["date"] = (
        key_df["series_date_key_str"].str.split("_", 1).str)
    key_df = key_df.drop(columns=["series_date_key_str"], axis=1)
    return key_df


def training_loop(CFG, LOGGER):
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
        LOGGER.info(f"-- fold{fold} training start --")
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
        # separate train/valid data
        # train_key_df, valid_key_df = get_train_valid_key_df(key_df, fold, CFG)
        # train_series_df = get_train_valid_series_df(
        #     series_df, key_df, fold, mode="train"
        # )
        # valid_series_df = get_train_valid_series_df(
        #     series_df, key_df, fold, mode="valid"
        # )
        train_series_df = series_df[series_df["fold"] != fold]
        train_key_df = get_key_df(train_series_df)
        valid_series_df = series_df[series_df["fold"] == fold]
        valid_key_df = get_key_df(valid_series_df)
        train_key_num = len(train_key_df)
        valid_key_num = len(valid_key_df)
        LOGGER.info(f"fold[{fold}] train data key num: {train_key_num}")
        LOGGER.info(f"fold[{fold}] valid data key num: {valid_key_num}")
        if train_key_num + valid_key_num != len(key_df):
            raise ValueError("train/valid data key num is not same")
        train_loader = get_loader(CFG,
                                  train_key_df,
                                  train_series_df,
                                  mode="train")
        valid_loader = get_loader(CFG,
                                  valid_key_df,
                                  valid_series_df,
                                  mode="valid")
        LOGGER.info(f"fold[{fold}] get_loader finished")

        oof_df_fold = valid_series_df.copy()
        init_cols = [
            "class_pred",
            "class_target",
        ]
        oof_df_fold = oof_df_fold.assign(
            **{col: -1 * np.ones(len(oof_df_fold))
               for col in init_cols})

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
            # wandb log
            wandb_log_dict[f"train_loss/fold{fold}"] = train_loss_avg
            wandb_log_dict[f"valid_loss/fold{fold}"] = valid_loss_avg
            wandb_log_dict[f"lr/fold{fold}"] = lr
            wandb_logger.log_progress(epoch, wandb_log_dict)

        # model save
        model_path = os.path.join(CFG.exp_dir, f"fold{fold}_model.pth")
        torch.save(model.state_dict(), model_path)
        if len(oof_df_fold["series_date_key"].unique()) != len(
                valid_series_df["series_date_key"].unique()):
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
        oof_df_fold = get_oof_df(
            input_info_dict_list,
            valid_predictions,
            valid_targets,
            oof_df_fold,
            CFG,
        )
        LOGGER.info(f"fold{fold} oof_df created.")
        if "downsample" in CFG.model_type:
            oof_df_fold = detect_event_from_classpred(oof_df_fold,
                                                      N=21,
                                                      maxpool_kernel_size=3)
        else:
            oof_df_fold = detect_event_from_classpred(oof_df_fold)
        oof_dir = os.path.join(CFG.output_dir, "_oof", CFG.exp_name)
        os.makedirs(oof_dir, exist_ok=True)
        oof_df_fold_path = os.path.join(oof_dir, f"oof_df_fold{fold}.parquet")
        print("save oof_df to ", oof_df_fold_path)
        oof_df_fold.to_parquet(oof_df_fold_path)
        LOGGER.info(f"fold{fold} event detected.")
        oof_scoring_df = make_submission_df(oof_df_fold)
        LOGGER.info(f"fold{fold} submission df created.")
        del oof_df_fold
        gc.collect()

        event_df_fold = event_df[event_df["series_id"].isin(
            valid_key_df["series_id"])]
        event_df_fold = event_df_fold[event_df_fold["step"].notnull()]
        oof_score = score(event_df_fold, oof_scoring_df)
        oof_score_list.append(oof_score)
        LOGGER.info(f"fold{fold} oof score: {oof_score:.4f}")
        # oof_df_fold_path = os.path.join(CFG.exp_dir, f"oof_df_fold{fold}.parquet")

        wandb_logger.log_oofscore(fold, oof_score)
    # oof_df = detect_event_from_classpred(oof_df)
    # oof_scoring_df = make_submission_df(oof_df)
    # event_df = event_df[event_df["series_id"].isin(oof_scoring_df["series_id"])]
    # event_df = event_df[event_df["step"].notnull()]
    # oof_score = score(event_df, oof_scoring_df)
    # LOGGER.info(f"overall oof score: {oof_score:.4f}")
    # wandb_logger.log_overall_oofscore(oof_score)
    over_all_score_mean = np.mean(oof_score_list)
    LOGGER.info(f"overall oof score mean: {over_all_score_mean:.4f}")
    wandb_logger.log_overall_oofscore(over_all_score_mean)
    # oof_df_path = os.path.join(CFG.exp_dir, "oof_df.parquet")
    # print("save oof_df to ", oof_df_path)
    # oof_df.to_parquet(oof_df_path)
    # print("oof_df saved. finish exp.")
    # return oof_df


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))

    class CFG:
        # exp
        EXP_NAME = "no_scoring_check"

        # directory
        INPUT_DIR = os.path.abspath(os.path.join(
            ROOT_DIR,
            "input",
        ))
        COMPETITION_DIR = os.path.join(
            INPUT_DIR,
            "child-mind-institute-detect-sleep-states",
        )
        OUTPUT_DIR = os.path.abspath(os.path.join(ROOT_DIR, "working"))
        # TRAIN_DIR = os.path.join(COMPETITION_DIR, "train")
        # TEST_DIR = os.path.join(COMPETITION_DIR, "test")
        key_df = os.path.join(INPUT_DIR, "datakey_unique_non_null.csv")
        series_df = os.path.join(INPUT_DIR,
                                 "processed_train_withkey_nonull.parquet")
        # event_df = os.path.join(INPUT_DIR, "train_events.csv")
        event_df = os.path.join(
            "/kaggle/input/preprocessed_train_event_notnull.parquet")
        # data
        # folds = [0, 1, 2, 3, 4]
        folds = [0]
        n_folds = 5
        num_workers = os.cpu_count()
        seed = 42
        group_key = "series_id"

        # model
        model_type = "single_output"
        input_channels = 2
        class_output_channels = 1
        event_output_channels = 2
        embedding_base_channels = 16

        class_loss_weight = 1.0
        event_loss_weight = 100.0

        # training
        n_epoch = 5
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

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    LOGGER = init_logger(log_file=os.path.join(CFG.OUTPUT_DIR, "check.log"))
    LOGGER.info(f"using device: {CFG.device}")
    training_loop(CFG, LOGGER)
