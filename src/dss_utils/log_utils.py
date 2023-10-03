import logging
import math
import time


def init_logger(log_file="train.log"):
    """Output Log."""
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

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
        event_losses: AverageMeter,
    ) -> None:
        self.batch_time.update(time.time() - self.end_time)
        if (batch_idx - 1 % self.print_freq == 0) or (batch_idx == (self.data_num) - 1):
            remain_time = self._timeSince(
                self.start_time,
                float(batch_idx + 1) / self.data_num,
            )
            log_str = f"[{self.mode}] Epoch: [{epoch}][{batch_idx}/{self.data_num}]"
            log_str += ", " + f"Elapsed {self.batch_time.val:.4f} s"
            log_str += ", " + f"remain {remain_time}"
            log_str += ", " + f"class loss {class_losses.val:.4f}"
            log_str += ", " + f"avg class loss {class_losses.avg:.4f}"
            log_str += ", " + f"event loss {event_losses.val:.4f}"
            log_str += ", " + f"avg event loss {event_losses.avg:.4f}"
            self.logger.info(log_str)
