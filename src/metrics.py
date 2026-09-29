"""
Metrics for one binary label (y: ground truth, s: continuous score).
"""

import numpy as np


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """
    P(score of a random positive > score of a random negative), ties = 1/2.
    """
    pos, neg = s[y], s[~y]
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """
    Area under the precision-recall curve (mean precision at each positive).
    """
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """
    Accuracy, precision, recall, F1 and error counts of boolean predictions.
    """
    accuracy = float((y == pred).mean())
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "fp": fp, "fn": fn,
    }
