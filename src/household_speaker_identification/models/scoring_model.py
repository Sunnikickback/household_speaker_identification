import torch
from torch import nn

from household_speaker_identification.models.utils.layers import LinearScorer
from household_speaker_identification.utils.params import ScoringModelParams


class ScoringModel(nn.Module):
    def __init__(self, params: ScoringModelParams):
        super(ScoringModel, self).__init__()
        self.relu = nn.ReLU()
        self.input_dropout = nn.Dropout(params.input_dropout_rate)
        self.adaptation_net = nn.Linear(in_features=params.adaptation_input_features,
                                        out_features=params.adaptation_output_features, bias=params.use_bias)
        self.cosine_similarity = nn.CosineSimilarity()
        self.scorer = LinearScorer()
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb1, emb2):
        batch_size = emb1.shape[0]

        cosine_sim_scores = self.cosine_similarity(emb1, emb2).reshape(-1, 1)
        emb1_after_dropout = self.input_dropout(emb1)
        emb2_after_dropout = self.input_dropout(emb2)
        emb1_after_adaptation_net = self.adaptation_net(emb1_after_dropout)
        emb2_after_adaptation_net = self.adaptation_net(emb2_after_dropout)
        euclidean_dists = torch.cdist(emb1_after_adaptation_net.view(batch_size, 1, -1),
                                      emb2_after_adaptation_net.reshape(batch_size, 1, -1)).reshape(-1, 1)
        scores = self.sigmoid(self.scorer(cosine_sim_scores, euclidean_dists))
        return scores
