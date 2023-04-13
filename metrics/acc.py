import numpy as np
from sklearn import metrics

from typing import Tuple


def dtc_acc(y_preds: np.ndarray, y_trues: np.ndarray) -> Tuple[np.float64, np.float64]:
    acc = 0.0
    for threshold in np.arange(0, 1, 0.001):
        y_preds_ = (y_preds >= threshold).astype(int)
        acc_ = metrics.accuracy_score(y_pred=y_preds_, y_true=y_trues)
        if acc_ > acc:
            acc = acc_
    return np.round(acc, decimals=5)


def cls_acc(y_preds: np.ndarray, y_trues: np.ndarray) -> Tuple[np.float64, np.float64]:
    y_preds_ = np.argmax(y_preds, axis=1)
    acc = metrics.accuracy_score(y_pred=y_preds_, y_true=y_trues)

    return np.round(acc, decimals=5)
