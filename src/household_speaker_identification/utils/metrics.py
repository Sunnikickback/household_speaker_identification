import torch


def compute_eer(positive_scores, negative_scores):
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    min_index = (FAR - FRR).abs().argmin()

    EER = (FAR[min_index] + FRR[min_index]) / 2

    return float(EER), float(thresholds[min_index])
