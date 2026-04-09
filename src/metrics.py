"""
Evaluation metrics: OA, Precision, Recall, F1, mIoU.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


def compute_iou_per_class(y_true, y_pred, num_classes):
    """Compute IoU for each class."""
    ious = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        if tp + fp + fn == 0:
            ious.append(1.0)
        else:
            ious.append(tp / (tp + fp + fn))
    return ious


def evaluate(y_true, y_pred, class_names=None):
    """
    Compute all metrics and return as dict.
    """
    num_classes = len(set(y_true))
    oa = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ious = compute_iou_per_class(np.array(y_true), np.array(y_pred), num_classes)
    miou = np.mean(ious)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0)

    return {
        "oa": oa,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miou": miou,
        "iou_per_class": ious,
        "confusion_matrix": cm,
        "report": report,
    }
