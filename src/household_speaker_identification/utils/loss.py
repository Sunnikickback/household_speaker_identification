import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def __call__(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor):
        coef = 1 / (positive_scores.size + negative_scores.size)
        compensation_weight = negative_scores.size()/positive_scores.size()
        return -coef * (compensation_weight * torch.sum(torch.log(positive_scores)) +
                     torch.sum(torch.log(negative_scores)))

