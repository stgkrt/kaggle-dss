# bce loss

import torch
import torch.nn as nn


def get_criterion(CFG):
    if hasattr(CFG, "positive_weight"):
        positive_weight = torch.tensor([CFG.positive_weight])
    else:
        positive_weight = torch.tensor([0.5])
    positive_weight = positive_weight.to(CFG.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    # criterion = nn.BCELoss(weight=positive_weight)
    # criterion = nn.CrossEntropyLoss(weight=positive_weight)
    # criterion = nn.CrossEntropyLoss()
    return criterion


if __name__ == "__main__":

    class CFG:
        device = "cpu"
        positive_weight = 0.5

    # softmax = nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()

    preds = torch.rand(1, 1, 5)
    targets = torch.rand(1, 1, 5)
    targets = (targets > 0.5).float()
    print(preds)
    preds = sigmoid(preds)
    print(preds)
    print(targets)

    criterion = get_criterion(CFG)
    loss = criterion(preds, targets)
    print(loss)
