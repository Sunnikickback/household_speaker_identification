import math

import torch
from torch import nn


class SyncDropout(nn.Module):
    def __init__(self, dropout_rate, in_features):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.in_features = in_features

    def forward(self, emb1, emb2):
        indices = torch.randperm(self.in_features)[:int((1-self.dropout_rate) * self.in_features)]
        emb1[indices] = 0
        emb2[indices] = 0
        return emb1, emb2


class OriginalDropout(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, emb1, emb2):
        emb1 = self.dropout(emb1)
        emb2 = self.dropout(emb2)
        return emb1, emb2


class LinearScorer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()
        weights_cosine = torch.Tensor(1, 1)
        self.weights_cosine = nn.Parameter(weights_cosine)  # nn.Parameter is a Tensor that's a module parameter.
        weights_euclidian = torch.Tensor(1, 1)
        self.weights_euclidian = nn.Parameter(weights_euclidian)
        bias = torch.Tensor(1)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights_cosine, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_euclidian, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights_cosine)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights_euclidian)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, cosine_score, euclidian_score):
        w_times_cosine_score = torch.mm(cosine_score, self.weights_cosine.t())
        w_times_euclidian_score = torch.mm(euclidian_score, self.weights_euclidian.t())
        weighted_score_sum = torch.add(w_times_cosine_score, w_times_euclidian_score)
        return torch.add(weighted_score_sum, self.bias)