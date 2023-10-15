# bce loss

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.bceloss = nn.BCELoss(reduction="none")

    def forward(self, outputs, targets):
        bce = self.bceloss(outputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = (1 - bce_exp) ** self.gamma * bce
        return focal_loss.mean()


class PositiveOnlyLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(PositiveOnlyLoss, self).__init__()
        self.eps = eps

    def forward(self, outputs, targets):
        positive_mask = (targets > 0.5).float()
        masked_outputs = outputs * positive_mask
        masked_targets = targets * positive_mask
        # bceっぽい感じのpositivie 部分だけ計算するlossにする
        loss = -masked_targets * torch.log(masked_outputs + self.eps)
        return loss.mean()


class PositiveAroundNegativeLoss(nn.Module):
    def __init__(self, pos_weight=10.0, neg_weight=1.0, eps=1e-7):
        # weightはもう一方のlossを見て雰囲気で決めた
        super(PositiveAroundNegativeLoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.eps = eps

    def forward(self, outputs, targets):
        # dataloaderのpoolのサイズでこの閾値は変化するので注意
        positive_mask = (targets > 0.5).float()
        # average poolで作った1の周りのposとnegの閾値の間のところはlossを計算しない
        negative_mask = (targets < self.eps).float()

        pos_masked_outputs = outputs * positive_mask
        neg_masked_outputs = outputs * negative_mask

        pos_loss = -positive_mask * torch.log(pos_masked_outputs + self.eps)
        neg_loss = -negative_mask * torch.log(1 - neg_masked_outputs + self.eps)

        loss = pos_loss * self.pos_weight + neg_loss * self.neg_weight
        return loss.mean()


class NegativeIgnoreBCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(NegativeIgnoreBCELoss, self).__init__()
        self.eps = eps

    def forward(self, outputs, targets):
        positive_mask = (targets > 0.0).float()
        neg_mask = (targets == 0.0).float()
        pos_masked_outputs = outputs * positive_mask
        pos_masked_targets = targets * positive_mask
        neg_masked_outputs = outputs * neg_mask
        neg_masked_targets = (1 - targets) * neg_mask
        # bceっぽい感じのpositivie 部分だけ計算するlossにする
        pos_loss = -pos_masked_targets * torch.log(pos_masked_outputs + self.eps)
        neg_loss = -neg_masked_targets * torch.log(1 - neg_masked_outputs + self.eps)
        loss = pos_loss + neg_loss
        return loss.mean()


def get_class_criterion(CFG):
    # if hasattr(CFG, "positive_weight"):
    #     positive_weight = torch.tensor([CFG.class_positive_weight])
    # else:
    #     positive_weight = torch.tensor([0.5])

    # positive_weight = positive_weight.to(CFG.device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    # criterion = nn.BCELoss(weight=positive_weight)
    # criterion = nn.CrossEntropyLoss(weight=positive_weight)
    # criterion = nn.CrossEntropyLoss()
    criterion = NegativeIgnoreBCELoss()
    return criterion


def get_event_criterion(CFG):
    # if hasattr(CFG, "positive_weight"):
    #     positive_weight = torch.tensor([CFG.event_positive_weight])
    # else:
    #     positive_weight = torch.tensor([0.5])

    # positive_weight = positive_weight.to(CFG.device)
    # criterion = nn.BCELoss(weight=positive_weight)

    # criterion = FocalLoss(gamma=2.0)
    # criterion = PositiveOnlyLoss()
    criterion = PositiveAroundNegativeLoss()
    return criterion


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

    criterion = get_class_criterion(CFG)
    loss = criterion(preds, targets)
    # loss = criterion(targets, targets)
    print(loss)
