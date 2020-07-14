import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss(output, target):
    loss = torch.nn.BCEWithLogitsLoss()
    # print(output.size(),target.size())
    return loss(output, target)