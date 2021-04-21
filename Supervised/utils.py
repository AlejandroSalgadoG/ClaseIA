import sklearn.metrics as mt
import numpy as np

def get_metrics(y_true, y_hat, y_proba=None):
    n = len(y_true)
    tn, fp, fn, tp = mt.confusion_matrix(y_true, y_hat).ravel()
    metrics = {}
    metrics["accuracy"] = (tn + tp) / n
    metrics["sensitivity"] = tn / (tn + fp)
    metrics["specificity"] = tp / (tp + fn)
    if y_proba is not None:
        metrics["roc-auc"] = mt.roc_auc_score(y_true, y_proba)
    else:
        metrics["roc-auc"] = np.nan
    return metrics