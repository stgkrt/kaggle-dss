import gc
import os
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

from dss_dataloader import get_loader
from dss_model import get_model
from log_utils import AverageMeter, ProgressLogger, init_logger
from losses import get_criterion
from scheduler import get_optimizer, get_scheduler
from train_valid_split import get_train_valid_key_df, get_train_valid_series_df


def train_fn(CFG, epoch, model, train_loader, criterion, optimizer, LOGGER):
    model.train()
    prog_loagger = ProgressLogger(
        data_num=len(train_loader),
        print_freq=CFG.print_freq,
        logger=LOGGER,
    )
    losses = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(CFG.device, non_blocking=True).float()
        targets = targets.to(CFG.device, non_blocking=True).float()
        preds = model(inputs)
        loss = criterion(preds, targets)
        preds = torch.sigmoid(preds)  # sigmoidいらない場合はこれを消す
        losses.update(loss.item(), CFG.batch_size)
        loss.backward()  # パラメータの勾配を計算
        optimizer.step()  # モデル更新
        optimizer.zero_grad()  # 勾配の初期化
        prog_loagger.log_progress(epoch, batch_idx, losses)

        del preds, inputs, targets
    gc.collect()
    torch.cuda.empty_cache()
    return losses.avg


def get_validation_pred_target(
    targets: torch.Tensor,
    preds: torch.Tensor,
    test_targets: np.ndarray,
    test_preds: np.ndarray,
):
    targets = targets.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    if len(test_targets) == 0:
        test_targets = targets.reshape(-1)  # type: ignore
        test_preds = preds.reshape(-1)  # type: ignore
    else:
        test_targets = np.concatenate([test_targets, targets.reshape(-1)])
        test_preds = np.concatenate([test_preds, preds.reshape(-1)])
    return test_targets, test_preds


def valid_fn(
    CFG,
    epoch,
    model,
    valid_loader,
    criterion,
    LOGGER,
):
    model.eval()
    test_targets = np.empty(0)
    test_preds = np.empty(0)
    losses = AverageMeter()
    prog_loagger = ProgressLogger(
        data_num=len(valid_loader),
        print_freq=CFG.print_freq,
        logger=LOGGER,
        mode="valid",
    )

    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        inputs = inputs.to(CFG.device, non_blocking=True).float()
        targets = targets.to(CFG.device, non_blocking=True).float()
        with torch.no_grad():
            preds = model(inputs)
            preds = torch.sigmoid(preds)
            loss = criterion(preds, targets)
        losses.update(loss.item(), CFG.batch_size)
        prog_loagger.log_progress(epoch, batch_idx, losses)
        test_targets, test_preds = get_validation_pred_target(
            targets, preds, test_targets, test_preds
        )

        del preds, inputs, targets
        gc.collect()
        torch.cuda.empty_cache()
    return test_targets, test_preds, losses.avg


def training_loop(CFG, LOGGER):
    key_df = pd.read_csv(CFG.key_df)
    series_df = pd.read_parquet(CFG.series_df)
    for fold in CFG.folds:
        LOGGER.info(f"-- fold{fold} training start --")
        # set model & learning fn
        model = get_model(CFG)
        model = model.to(CFG.device)
        criterion = get_criterion(CFG)
        optimizer = get_optimizer(model, CFG)
        scheduler = get_scheduler(optimizer, CFG)

        # training

        start_time = time.time()

        # separate train/valid data
        train_key_df, valid_key_df = get_train_valid_key_df(key_df, fold, CFG)
        train_series_df = get_train_valid_series_df(
            series_df, key_df, fold, mode="train"
        )
        valid_series_df = get_train_valid_series_df(
            series_df, key_df, fold, mode="valid"
        )

        train_loader = get_loader(CFG, train_key_df, train_series_df, mode="train")
        valid_loader = get_loader(CFG, valid_key_df, valid_series_df, mode="valid")
        for epoch in range(0, CFG.n_epoch):
            LOGGER.info(f"- epoch:{epoch} -")
            train_loss_avg = train_fn(
                CFG, epoch, model, train_loader, criterion, optimizer, LOGGER
            )
            valid_targets, valid_preds, valid_loss_avg = valid_fn(
                CFG,
                epoch,
                model,
                valid_loader,
                criterion,
                LOGGER,
            )
            # auc = calc_auc(valid_targets, valid_preds)
            # score, threshold, _ = calc_cv(valid_targets, valid_preds)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            elapsed = int(time.time() - start_time) / 60
            log_str = f"FOLD:{fold}, Epoch:{epoch}"
            log_str += f", train:{train_loss_avg:.4f}, valid:{valid_loss_avg:.4f}"
            log_str += f", lr:{lr:.6f}, elapsed time:{elapsed:.2f} min"
            LOGGER.info(log_str)
            # competition scoreを計算するように変更する

        del model, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))

    class CFG:
        # exp
        EXP_NAME = "no_scoring_check"

        # directory
        INPUT_DIR = os.path.abspath(
            os.path.join(
                ROOT_DIR,
                "input",
            )
        )
        COMPETITION_DIR = os.path.join(
            INPUT_DIR,
            "child-mind-institute-detect-sleep-states",
        )
        OUTPUT_DIR = os.path.abspath(os.path.join(ROOT_DIR, "working", "debug"))
        # TRAIN_DIR = os.path.join(COMPETITION_DIR, "train")
        # TEST_DIR = os.path.join(COMPETITION_DIR, "test")
        key_df = os.path.join(INPUT_DIR, "datakey_unique_non_null.csv")
        series_df = os.path.join(INPUT_DIR, "train_series_withkey_non_null.parquet")
        # data
        folds = [0, 1, 2, 3, 4]
        n_folds = 5
        num_workers = os.cpu_count()
        seed = 42
        group_key = "series_id"

        # model
        input_channels = 2
        output_channels = 1
        embedding_base_channels = 16

        # training
        n_epoch = 10
        batch_size = 32
        # optimizer
        lr = 1e-4
        weight_decay = 1e-6
        # scheduler
        T_0 = 10
        T_mult = 1
        eta_min = 1e-9

        # log setting
        print_freq = 100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    LOGGER = init_logger(log_file=os.path.join(CFG.OUTPUT_DIR, "check.log"))
    LOGGER.info(f"using device: {CFG.device}")
    training_loop(CFG, LOGGER)
