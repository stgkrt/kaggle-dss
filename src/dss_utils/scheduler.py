from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def get_optimizer(model, CFG):
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
        amsgrad=False,
    )
    return optimizer


def get_scheduler(optimizer, CFG):
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CFG.T_0,
        T_mult=CFG.T_mult,
        eta_min=CFG.eta_min,
        last_epoch=-1,
    )
    return scheduler
