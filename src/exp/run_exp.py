import argparse
import os
import sys

import torch
from pseudo_training_loop import pseudo_training_loop
from training_loop import seed_everything, training_loop

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from logger import init_logger, save_config, set_wandb_make_dir

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "offline"


def parse_args():
    parser = argparse.ArgumentParser(description="run experiment.")
    parser.add_argument(
        "--train-mode", type=str, default="train", help="train or pseudo"
    )
    parser.add_argument(
        "--competition_name",
        type=str,
        default="dss",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="taro",
    )
    parser.add_argument(
        "--exp_category",
        type=str,
        default="debug",
        help="experiment category.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="no_scoring_check",
        help="experiment name.",
    )
    root_dir = os.path.abspath("/kaggle")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(root_dir, "input"),
        help="input directory.",
    )
    parser.add_argument(
        "--competition_dir",
        type=str,
        default=os.path.join(
            root_dir, "input", "child-mind-institute-detect-sleep-states"
        ),
        help="competition directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(root_dir, "working"),
        help="output directory.",
    )
    parser.add_argument(
        "--key_df",
        type=str,
        default=os.path.join(root_dir, "input", "datakey_unique_non_null.csv"),
    )
    parser.add_argument(
        "--series_df",
        type=str,
        default=os.path.join(
            root_dir,
            "input",
            "preprocessed_train_series_le_fold.parquet",
            # "preprocessed_train_series_6ch_lepseudo_fold.parquet",
            # "preprocessed_train_series_le_50_fold.parquet",
        ),
    )
    parser.add_argument(
        "--event_df",
        type=str,
        default=os.path.join(
            root_dir,
            "input",
            "child-mind-institute-detect-sleep-states",
            "train_events.csv",
        ),
    )
    parser.add_argument("--group_key", type=str, default="series_id")

    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="*", default=[0])
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--num_workers", type=int, default=2)

    # model
    parser.add_argument("--model_type", type=str, default="add_rolldiff")
    parser.add_argument("--input_channels", type=int, default=4)
    parser.add_argument("--class_output_channels", type=int, default=1)
    parser.add_argument("--event_output_channels", type=int, default=2)
    parser.add_argument("--embedding_base_channels", type=int, default=16)
    parser.add_argument("--ave-kernel-size", type=int, default=11)
    parser.add_argument("--pseudo_weight_exp", type=str, default="exp003")

    # training setting
    parser.add_argument("--n_epoch", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--T_0", type=int, default=10)
    parser.add_argument("--T_mult", type=int, default=1)
    parser.add_argument("--eta_min", type=float, default=1e-9)
    parser.add_argument("--class_loss_weight", type=float, default=1.0)
    parser.add_argument("--event_loss_weight", type=float, default=1.0)

    # log setting
    parser.add_argument("--print_freq", type=int, default=50)

    if torch.cuda.is_available():
        parser.add_argument("--device", type=str, default="cuda")
    else:
        raise Exception("please use gpu")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    if config.device == "cpu":
        raise Exception("please use gpu")

    print(config)
    config = set_wandb_make_dir(config)
    print("set wandb make dir done.")
    save_config(config)
    print("save config done.")
    logger = init_logger(config.logger_path)

    seed_everything(config.seed)
    if config.train_mode == "train":
        training_loop(config, logger)
    elif config.train_mode == "pseudo":
        pseudo_training_loop(config, logger)
