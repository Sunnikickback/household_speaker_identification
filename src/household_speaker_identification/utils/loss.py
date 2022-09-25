import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def __call__(self, positive_scores, negative_scores):
        coef = 1 / (positive_scores.size()[0] + negative_scores.size()[0])
        compensation_weight = negative_scores.size()[0] / positive_scores.size()[0]
        return -coef * (compensation_weight * torch.sum(torch.log(positive_scores)) +
                        torch.sum(torch.log(1-negative_scores)))
