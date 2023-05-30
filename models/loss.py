import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss. Adapted from https://bit.ly/2T6kfz7. If 0 < smoothing < 1,
    this smoothes the standard cross-entropy loss.
    """

    def __init__(self, smoothing, n_classes):
        super().__init__()
        _n_classes = n_classes
        self.confidence = 1. - smoothing
        smoothing_value = smoothing / (_n_classes - 1)
        one_hot = torch.full((_n_classes,), smoothing_value)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, output, target):
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        # return nn.CrossEntropyLoss()(output, model_prob)
        return F.kl_div(F.log_softmax(output, 1), model_prob, reduction='sum')
