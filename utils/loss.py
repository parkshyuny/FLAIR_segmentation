from torch import nn
import torch

def dice_score(y_pred, y_true):
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    overlap = (y_pred * y_true).sum()
    dsc = (2 * overlap + 1.0) / (
          (y_pred.sum() + y_true.sum() + 1.0))

    return 1. - dsc

def bce_dice_score(y_pred, y_true):
    dice_loss = dice_score(y_pred, y_true)
    bce = nn.BCELoss()
    bce_loss = bce(y_pred, y_true)

    return bce_loss + dice_loss

# Loss check
# loss = bce_dice_score(torch.tensor([0.7, 1., 1.]), 
#                       torch.tensor([1.,1.,1.]))
# print(loss)