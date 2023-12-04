import logging
import math
import os
import time

import yaml  # type: ignore

import wandb


def init_wandb(configs):
    if configs.wandb_available:
        WANDB_CONFIG = {
            "competition": configs.competition_name,
            "_wandb_kernel": configs.user_name,
        }
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            project=WANDB_CONFIG["competition"],
            config=configs,
            group=configs.exp_category,
            name=configs.exp_name,
            reinit=True,
            save_code=True,
        )


def save_config(configs):
    config_dict = vars(configs)
    with open(os.path.join(configs.exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f)


def set_wandb_make_dir(configs):
    configs.exp_dir = os.path.join(configs.output_dir, configs.exp_name)
    if "debug" in configs.exp_name:  # folderはdebug用で上書きOK。wandbも記録しない。
        # configs.folds = [0]
        os.makedirs(configs.exp_dir, exist_ok=True)
        configs.wandb_available = False
        # configs.wandb_available = True
    elif "check" in configs.exp_name:  # folderはdebug用で上書きOK。wandbは記録する。
        # configs.folds = [0]
        os.makedirs(configs.exp_dir, exist_ok=True)
        configs.wandb_available = True
    else:
        if os.path.exists(configs.exp_dir):
            configs.exp_name += "_" + str(time.time())
            configs.exp_dir = os.path.join(configs.output_dir, configs.exp_name)
        os.makedirs(configs.exp_dir)
        configs.wandb_available = True
    configs.logger_path = os.path.join(configs.exp_dir, "train.log")
    if configs.wandb_available:
        init_wandb(configs)
    return configs


def init_logger(log_file="train.log"):
    """Output Log."""
    from logging import INFO
    from logging import FileHandler
    from logging import Formatter
    from logging import StreamHandler
    from logging import getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


# defatult_logger = init_logger()


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressLogger:
    def __init__(
        self,
        data_num: int,
        logger: logging.Logger,
        print_freq: int = 10,
        mode: str = "train",
    ) -> None:
        self.data_num = data_num
        self.print_freq = print_freq
        self.logger = logger
        self.mode = mode
        self.start_time = time.time()
        self.end_time = time.time()
        self.batch_time = AverageMeter()

    def _asMinutes(self, s):
        """Convert Seconds to Minutes."""
        m = math.floor(s / 60)
        s -= m * 60
        return "%dm %ds" % (m, s)

    def _timeSince(self, since, percent):
        """Accessing and Converting Time Data."""
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return "%s (remain %s)" % (self._asMinutes(s), self._asMinutes(rs))

    def log_progress(
        self,
        epoch: int,
        batch_idx: int,
        class_losses: AverageMeter,
        event_losses: AverageMeter | None = None,
    ) -> None:
        self.batch_time.update(time.time() - self.end_time)
        if (batch_idx % self.print_freq == 0) or (batch_idx == (self.data_num) - 1):
            remain_time = self._timeSince(
                self.start_time,
                float(batch_idx + 1) / self.data_num,
            )
            log_str = f"[{self.mode}] Epoch: [{epoch}][{batch_idx}/{self.data_num}]"
            log_str += ", " + f"Elapsed {self.batch_time.val:.4f} s"
            log_str += ", " + f"remain {remain_time}"
            # log_str += ", " + f"class loss {class_losses.val:.4f}"
            log_str += ", " + f"avg class loss {class_losses.avg:.4f}"

            if event_losses is not None:
                # log_str += ", " + f"event loss {event_losses.val:.4f}"
                log_str += ", " + f"avg event loss {event_losses.avg:.4f}"
            self.logger.info(log_str)


class WandbLogger:
    def __init__(self, config) -> None:
        self.wadnb_available = config.wandb_available

    def log_progress(
        self,
        epoch: int,
        log_dict: dict,
    ) -> None:
        if self.wadnb_available:
            wandb.log(log_dict)  # type: ignore

    def log_oofscore(self, fold, score):
        if self.wadnb_available:
            wandb.log({"oof_score": score, "fold": fold})

    def log_overall_oofscore(self, score):
        if self.wadnb_available:
            wandb.log({"overall_oof_score": score})

    def log_best_score(self, fold, score):
        if self.wadnb_available:
            wandb.log({"best_score": score, "fold": fold})
