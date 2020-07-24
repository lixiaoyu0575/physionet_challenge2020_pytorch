import torch.nn.functional as F
import torch
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_with_logits_loss(output, target):
    loss = torch.nn.BCEWithLogitsLoss()
    # print(output.size(),target.size())
    return loss(output, target)

def bce_loss(output, target):
    loss = torch.nn.BCELoss()
    return loss(output, target)

def focal_loss(outputs, targets, alpha=1, gamma=2, logits=False, reduce=True):
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduce=False)
    else:
        BCE_loss = F.binary_cross_entropy(outputs, targets, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss

def dice_loss(output, target):
    N = target.size(0)
    smooth = 1

    input_flat = output.view(N, -1)
    target_flat = target.view(N, -1)

    intersection = input_flat * target_flat

    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    loss = 1 - loss.sum() / N

    return loss

def multiclass_dice_loss(output, target, weights=None):
    C = target.shape[1]

    # if weights is None:
    # 	weights = torch.ones(C) #uniform weights for all classes

    dice = DiceLoss()
    totalLoss = 0

    for i in range(C):
        diceLoss = dice(output[:, i], target[:, i])
        if weights is not None:
            diceLoss *= weights[i]
        totalLoss += diceLoss

    return totalLoss

## nn.Moudle
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, outputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(outputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss