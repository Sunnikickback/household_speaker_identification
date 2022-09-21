import torch
from torch import nn

from src.household_speaker_identification.utils.params import ScoringModelParams


class Model(nn.Module):
    def __init__(self, params: ScoringModelParams):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.input_dropout = nn.Dropout(params.dropout_rate)
        self.adaptation_net = nn.Linear(in_features=params.in_features, out_features=params.out_features, bias=params.use_bias)
        self.cosine_similarity = nn.CosineSimilarity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb1, emb2):
        cosine_sim_scores = self.cosine_similarity(emb1, emb2)
        emb1_after_dropout = self.input_dropout(emb1)
        emb2_after_dropout = self.input_dropout(emb2)
        emb1_after_adaptation_net = self.adaptation_net(emb1_after_dropout)
        emb2_after_adaptation_net = self.adaptation_net(emb2_after_dropout)
        euclidean_dists = torch.cdist(emb1_after_adaptation_net, emb2_after_adaptation_net)
        scores = self.sigmoid(torch.cat([cosine_sim_scores, euclidean_dists], dim=1))
        return scores
