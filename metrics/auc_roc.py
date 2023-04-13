import numpy as np
from sklearn import metrics


def dtc_roc(y_preds: np.ndarray, y_trues: np.ndarray) -> np.float64:
    auc_roc = metrics.roc_auc_score(y_score=y_preds, y_true=y_trues)
    return np.round(auc_roc, decimals=5)


def cls_roc(y_preds: np.ndarray, y_trues: np.ndarray) -> np.float64:
    avg_roc = 0.0
    for i in range(y_preds.shape[-1]):
        y_preds_ = y_preds[:, i]
        y_trues_ = (y_trues == i).astype(int)
        auc_roc = metrics.roc_auc_score(y_score=y_preds_, y_true=y_trues_)
        avg_roc += auc_roc * (sum(y_trues_) / len(y_trues))

    return np.round(avg_roc, decimals=5)
