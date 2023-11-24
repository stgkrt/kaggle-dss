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

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(SRC_DIR, "dss_utils"))
sys.path.append(os.path.join(SRC_DIR, "data"))
sys.path.append(os.path.join(SRC_DIR, "model"))

from dss_dataloader_1ststage import get_loader
from dss_model_1ststage import get_model
from logger import AverageMeter
from logger import ProgressLogger
from logger import WandbLogger
from logger import init_logger
from scheduler import get_optimizer
from scheduler import get_scheduler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


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
    class_values: torch.Tensor,
    validation_dict: dict,
    mode: str = "preds",
) -> dict:
    class_values = class_values.detach().cpu().numpy()
    # class_values = class_values.astype(np.float16)  # type: ignore
    if len(validation_dict[f"class_{mode}"]) == 0:
        validation_dict[f"class_{mode}"] = class_values
    else:
        validation_dict[f"class_{mode}"] = np.concatenate(
            [validation_dict[f"class_{mode}"], class_values], axis=0
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
    valid_predictions = {"class_preds": np.empty(0)}
    valid_targets = {"class_targets": np.empty(0)}
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


def get_key_df(series_df: pd.DataFrame) -> pd.DataFrame:
    key_df = series_df[["series_date_key", "series_date_key_str"]].drop_duplicates()
    key_df = key_df.reset_index(drop=True)
    key_df["series_id"], key_df["date"] = (
        key_df["series_date_key_str"].str.split("_", 1).str
    )
    key_df = key_df.drop(columns=["series_date_key_str"], axis=1)
    return key_df


def stage1st_training_loop_earlysave(CFG, LOGGER):
    # key_df = pd.read_csv(CFG.key_df)
    LOGGER.info("loading series_df")
    series_df = pd.read_parquet(CFG.series_df)
    key_df = get_key_df(series_df)
    event_df = pd.read_csv(CFG.event_df)

    LOGGER.info("series_df data num : {}".format(len(series_df)))
    LOGGER.info("key_df data num : {}".format(len(key_df)))
    event_df = pd.read_csv(os.path.join(CFG.event_df))
    event_df = event_df[event_df["series_id"].isin(series_df["series_id"])]
    # oof_df = pd.DataFrame()
    best_score_list = []
    oof_dir = os.path.join(CFG.output_dir, "_oof", CFG.exp_name)
    os.makedirs(oof_dir, exist_ok=True)
    for fold in CFG.folds:
        LOGGER.info(f"-- fold{fold} training start --")
        wandb_logger = WandbLogger(CFG)
        wandb_log_dict = {}
        # set model & learning fn
        model = get_model(CFG)
        model = model.to(CFG.device)
        class_criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = get_optimizer(model, CFG)
        scheduler = get_scheduler(optimizer, CFG)

        # training
        start_time = time.time()

        LOGGER.info(f"fold[{fold}] loading train/valid data")
        train_series_df = series_df[series_df["fold"] != fold]
        train_key_df = get_key_df(train_series_df)
        valid_series_df = series_df[series_df["fold"] == fold]
        valid_key_df = get_key_df(valid_series_df)
        train_key_num = len(train_key_df)
        valid_key_num = len(valid_key_df)
        LOGGER.info(f"fold[{fold}] train data key num: {train_key_num}")
        LOGGER.info(f"fold[{fold}] valid data key num: {valid_key_num}")
        # if train_key_num + valid_key_num != len(key_df):
        #     raise ValueError("train/valid data key num is not same")
        train_loader = get_loader(
            CFG, train_key_df, train_series_df, event_df, mode="train"
        )
        valid_loader = get_loader(
            CFG, valid_key_df, valid_series_df, event_df, mode="valid"
        )
        LOGGER.info(f"fold[{fold}] get_loader finished")

        oof_df_fold = valid_series_df.copy()
        fold_best_score = 0.0
        best_valid_predictions = {}
        best_valid_targets = {}
        best_input_info_dict_list = {}
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
            oof_score = roc_auc_score(
                valid_targets["class_targets"].reshape(-1),
                valid_predictions["class_preds"].reshape(-1),
            )
            LOGGER.info(f"fold{fold} oof score: {oof_score:.4f}")
            for thr in [0.05, 0.1, 0.2]:
                preds = (valid_predictions["class_preds"] > thr).astype(np.int32)
                conf_mat = confusion_matrix(
                    valid_targets["class_targets"].reshape(-1), preds.reshape(-1)
                )
                tn, fp, fn, tp = conf_mat.ravel()
                LOGGER.info(
                    f"fold{fold} thr:{thr} tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}"
                )

            if len(oof_df_fold["series_date_key"].unique()) != len(
                valid_series_df["series_date_key"].unique()
            ):
                raise ValueError("oof data key num is not same")
            if oof_score > fold_best_score:
                best_valid_predictions = valid_predictions
                best_valid_targets = valid_targets
                best_input_info_dict_list = input_info_dict_list
                fold_best_score = oof_score
                model_path = os.path.join(CFG.exp_dir, f"fold{fold}_best_model.pth")
                torch.save(model.state_dict(), model_path)
                oof_df_fold_path = os.path.join(
                    oof_dir, f"fold{fold}_best_oof_df.parquet"
                )
                oof_df_fold.to_parquet(oof_df_fold_path)
            LOGGER.info(f"fold{fold} best score: {fold_best_score:.4f}")
            elapsed = int(time.time() - start_time) / 60
            log_str = f"FOLD:{fold}, Epoch:{epoch}"
            log_str += f", train:{train_loss_avg:.4f}, valid:{valid_loss_avg:.4f}"
            log_str += f", lr:{lr:.6f}, elapsed time:{elapsed:.2f} min"
            LOGGER.info(log_str)
            # wandb log
            wandb_log_dict[f"train_loss/fold{fold}"] = train_loss_avg
            wandb_log_dict[f"valid_loss/fold{fold}"] = valid_loss_avg
            wandb_log_dict[f"lr/fold{fold}"] = lr
            wandb_log_dict[f"oof_score/fold{fold}"] = oof_score
            wandb_logger.log_progress(epoch, wandb_log_dict)

        # oof save
        best_pred_df = pd.DataFrame(
            {
                "series_date_key": best_input_info_dict_list["series_date_key"],
                "start_step": best_input_info_dict_list["start_step"],
                "end_step": best_input_info_dict_list["end_step"],
                "class_preds": best_valid_predictions["class_preds"].reshape(-1),
                "class_targets": best_valid_targets["class_targets"].reshape(-1),
            }
        )
        best_pred_df.to_parquet(
            os.path.join(oof_dir, f"fold{fold}_best_pred_df.parquet")
        )
        # model save
        model_path = os.path.join(CFG.exp_dir, f"fold{fold}_model.pth")
        torch.save(model.state_dict(), model_path)
        best_score_list.append(fold_best_score)
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
    over_all_score_mean = np.mean(best_score_list)
    LOGGER.info(f"overall oof score mean: {over_all_score_mean:.4f}")
    wandb_logger.log_overall_oofscore(over_all_score_mean)


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))

    class CFG:
        # exp
        exp_name = "debug"

        # directory
        output_dir = os.path.abspath(os.path.join(ROOT_DIR, "working"))
        exp_dir = os.path.join(output_dir, exp_name)
        series_df = "/kaggle/input/train_series_alldata_skffold.parquet"
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
        input_channels = 2
        class_output_channels = 1
        event_output_channels = 2
        embedding_base_channels = 16
        enc_kernelsize_list = [12, 24, 48, 96, 192, 384]
        lstm_num_layers = 2
        lstm_hidden_size = 64

        # training
        n_epoch = 1
        batch_size = 32
        # optimizer
        lr = 1e-3
        weight_decay = 1e-6
        T_0 = n_epoch
        T_mult = 1
        eta_min = 1e-9

        # log setting
        wandb_available = False
        print_freq = 100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CFG.output_dir, exist_ok=True)

    seed_everything(CFG.seed)
    LOGGER = init_logger(log_file=os.path.join(CFG.output_dir, "check.log"))
    LOGGER.info(f"using device: {CFG.device}")
    stage1st_training_loop_earlysave(CFG, LOGGER)
