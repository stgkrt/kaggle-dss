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

    def _get_masked_value(self, value, mask):
        return torch.clip(value * mask, min=self.eps, max=1.0 - self.eps)

    def forward(self, outputs, targets):
        # targetsの-1のところは無視する
        positive_mask = (targets > 0.0).float()
        neg_mask = (targets == 0.0).float()
        pos_masked_outputs = self._get_masked_value(positive_mask, outputs)
        pos_masked_targets = self._get_masked_value(positive_mask, targets)
        neg_masked_outputs = self._get_masked_value(neg_mask, outputs)
        neg_masked_targets = self._get_masked_value(neg_mask, 1 - targets)
        # bceっぽい感じのpositivie 部分だけ計算するlossにする
        pos_loss = -pos_masked_targets * torch.log(pos_masked_outputs + self.eps)
        neg_loss = -neg_masked_targets * torch.log(1 - neg_masked_outputs + self.eps)
        loss = pos_loss + neg_loss
        return loss.mean()


class PseudoNegativeIgnoreBCELoss(nn.Module):
    def __init__(self, eps=1e-6, orig_loss_weight=0.2):
        super(PseudoNegativeIgnoreBCELoss, self).__init__()
        self.eps = eps
        self.orig_loss_weight = orig_loss_weight

    def _get_masked_value(self, value, mask):
        return torch.clip(value * mask, min=self.eps, max=1.0 - self.eps)

    def forward(self, outputs, targets, pseudo_targets):
        # 普通のnegative ignore loss(上のと同じ)
        positive_mask = (targets > 0.0).float()
        neg_mask = (targets == 0.0).float()
        pos_masked_outputs = self._get_masked_value(positive_mask, outputs)
        pos_masked_targets = self._get_masked_value(positive_mask, targets)
        neg_masked_outputs = self._get_masked_value(neg_mask, outputs)
        neg_masked_targets = self._get_masked_value(neg_mask, 1 - targets)
        # bceっぽい感じのpositivie 部分だけ計算するlossにする
        pos_loss = -pos_masked_targets * torch.log(pos_masked_outputs + self.eps)
        neg_loss = -neg_masked_targets * torch.log(1 - neg_masked_outputs + self.eps)
        orig_loss = pos_loss + neg_loss
        # pseudo_loss original が-1のところだけ計算する。
        # pseudo labelも-1のところは無視する
        ps_positive_mask = (pseudo_targets == 1.0).float() * (targets == -1).float()
        ps_neg_mask = (pseudo_targets == 0.0).float() * (targets == -1).float()
        ps_pos_masked_outputs = self._get_masked_value(ps_positive_mask, outputs)
        ps_pos_masked_targets = self._get_masked_value(ps_positive_mask, pseudo_targets)
        ps_neg_masked_outputs = self._get_masked_value(ps_neg_mask, outputs)
        ps_neg_masked_targets = self._get_masked_value(ps_neg_mask, 1 - pseudo_targets)
        ps_pos_loss = -ps_pos_masked_targets * torch.log(
            ps_pos_masked_outputs + self.eps
        )
        ps_neg_loss = -ps_neg_masked_targets * torch.log(
            1 - ps_neg_masked_outputs + self.eps
        )

        pseudo_loss = ps_pos_loss + ps_neg_loss

        # 過学習防止にoriginalにはweightをかけて小さめにしてみる
        loss = self.orig_loss_weight * orig_loss.mean() + pseudo_loss.mean()
        return loss


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


def get_pseudo_criterion(CFG):
    if hasattr(CFG, "orig_loss_weight"):
        orig_loss_weight = CFG.orig_loss_weight
    else:
        orig_loss_weight = 0.2
    criterion = PseudoNegativeIgnoreBCELoss(orig_loss_weight)
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
    targets = torch.tensor([[[-1.0, -1.0, 0.1], [0.1, 0.1, 0.8]]])
    pseudo_targets = torch.tensor([[[1.0, 0.0, 0.1], [0.1, 0.1, 0.8]]]).long()
    print(pseudo_targets == 1.0)
    print("targets", targets)

    # preds = softmax(preds)
    # print("after softmax", preds)

    # criterion = get_class_criterion(CFG)
    criterion = get_pseudo_criterion(CFG)
    loss = criterion(preds, targets, pseudo_targets)
    # loss = criterion(targets, targets)
    print(loss)
