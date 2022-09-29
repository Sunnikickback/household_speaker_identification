import torch
from torch import nn

from household_speaker_identification.models.utils.layers import LinearScorer, SyncDropout, OriginalDropout
from household_speaker_identification.utils.params import ScoringModelParams


class ScoringModel(nn.Module):
    def __init__(self, params: ScoringModelParams):
        super(ScoringModel, self).__init__()
        self.relu = nn.ReLU()
        self.in_features = params.adaptation_input_features
        self.input_dropout_rate = params.input_dropout_rate
        if params.dropout_type == "sync":
            self.dropout = SyncDropout(self.input_dropout_rate, self.in_features)
        else:
            self.dropout = OriginalDropout(self.input_dropout_rate)

        self.adaptation_net = nn.Linear(in_features=params.adaptation_input_features,
                                        out_features=params.adaptation_output_features, bias=params.use_bias)
        self.cosine_similarity = nn.CosineSimilarity()
        self.scorer = LinearScorer()
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb1, emb2, training=True):
        batch_size = emb1.shape[0]

        cosine_sim_scores = self.cosine_similarity(emb1, emb2).reshape(-1, 1)

        if training:
            emb1, emb2 = self.dropout(emb1, emb2)

        emb1_after_adaptation_net = self.relu(self.adaptation_net(emb1))
        emb2_after_adaptation_net = self.relu(self.adaptation_net(emb2))
        euclidean_dists = torch.cdist(emb1_after_adaptation_net.view(batch_size, 1, -1),
                                      emb2_after_adaptation_net.reshape(batch_size, 1, -1)).reshape(-1, 1)
        scores = self.sigmoid(self.scorer(cosine_sim_scores, euclidean_dists))
        return scores
