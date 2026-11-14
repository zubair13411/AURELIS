import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same non-zero length.')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_pneumonia': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall_pneumonia': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1_pneumonia': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'specificity': float(tn / (tn + fp)) if tn + fp else 0.0,
        'classification_report': classification_report(
            y_true, y_pred, labels=[0, 1], target_names=['NORMAL', 'PNEUMONIA'], zero_division=0
        ),
    }
