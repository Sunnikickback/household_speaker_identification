from torch import nn


class CosineModel(nn.Module):
    def __init__(self):
        super(CosineModel, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, emb1, emb2, training=True):
        cosine_sim_scores = self.cosine_similarity(emb1, emb2).reshape(-1, 1)
        return cosine_sim_scores
