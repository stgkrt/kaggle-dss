# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from timm.scheduler import CosineLRScheduler
from torch.optim import AdamW


def get_optimizer(model, CFG):
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
        amsgrad=False,
    )
    return optimizer


def get_scheduler(optimizer, CFG):
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=CFG.T_0,
    #     T_mult=CFG.T_mult,
    #     eta_min=CFG.eta_min,
    #     last_epoch=-1,
    # )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=CFG.T_0,
        lr_min=CFG.eta_min,
        warmup_lr_init=CFG.eta_min,
        warmup_t=CFG.warmup_t,
    )
    return scheduler
