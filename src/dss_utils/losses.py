# bce loss

import torch
import torch.nn as nn


def get_criterion(CFG):
    if hasattr(CFG, "positive_weight"):
        positive_weight = torch.tensor([CFG.positive_weight])
    else:
        positive_weight = torch.tensor([0.5])

    positive_weight = positive_weight.to(CFG.device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    criterion = nn.BCELoss(weight=positive_weight)
    # criterion = nn.CrossEntropyLoss(weight=positive_weight)
    # criterion = nn.CrossEntropyLoss()
    return criterion


# def get_event_criterion(CFG):
#     criterion = nn.CrossEntropyLoss()
#     return criterion


if __name__ == "__main__":

    class CFG:
        device = "cpu"
        positive_weight = 0.5

    softmax = nn.Softmax(dim=1)
    # sigmoid = nn.Sigmoid()

    # preds = torch.rand(1, 1, 3)
    # targets = torch.rand(1, 1, 3)
    # targets = (targets > 0.5).float()
    # print(preds)
    # preds = sigmoid(preds)

    # preds = torch.rand(1, 2, 3)
    # preds = torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]])
    preds = torch.tensor([[[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]])
    print("before preds", preds)
    # targets = torch.rand(1, 2, 3)
    # print("before targets", targets)
    # # targetsの一番大きい値だけ1にする
    # targets = (targets == targets.max(dim=2, keepdim=True)[0]).float()
    targets = torch.tensor([[[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]])

    print("targets", targets)

    # preds = softmax(preds)
    # print("after softmax", preds)

    criterion = get_criterion(CFG)
    loss = criterion(preds, targets)
    # loss = criterion(targets, targets)
    print(loss)
