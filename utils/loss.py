from torch import nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)

        overlap = (y_pred * y_true).sum()
        dsc = (2 * overlap + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )

        return 1. - dsc
