# bce loss

import torch
import torch.nn as nn


def get_criterion(CFG):
    if hasattr(CFG, "positive_weight"):
        positive_weight = torch.tensor([CFG.positive_weight])
    else:
        positive_weight = torch.tensor([0.5])
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    return criterion
