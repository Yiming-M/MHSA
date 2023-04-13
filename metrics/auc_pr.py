import numpy as np
from sklearn import metrics


def dtc_pr(y_preds: np.ndarray, y_trues: np.ndarray) -> np.float64:
    precision, recall, _ = metrics.precision_recall_curve(y_true=y_trues, probas_pred=y_preds)
    auc_pr = metrics.auc(x=recall, y=precision)
    return np.round(auc_pr, decimals=5)


def cls_pr(y_preds: np.ndarray, y_trues: np.ndarray) -> np.float64:
    avg_pr = 0.0
    for i in range(y_preds.shape[-1]):
        y_preds_ = y_preds[:, i]
        y_trues_ = (y_trues == i).astype(int)
        precision, recall, _ = metrics.precision_recall_curve(y_true=y_trues_, probas_pred=y_preds_)
        auc_pr = metrics.auc(x=recall, y=precision)
        avg_pr += auc_pr * (sum(y_trues_) / len(y_trues))

    return np.round(avg_pr, decimals=5)
