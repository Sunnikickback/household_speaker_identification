from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def compute_eer(labels, scores):
    assert len(scores) == len(labels)
    far, tpr, thresholds = roc_curve(labels, scores, pos_label=2)
    eer = brentq(lambda x: 1.0 - x - interp1d(far, tpr)(x), 0.0, 1.0)
    thresh = interp1d(far, thresholds)(eer)
    return eer, thresh
