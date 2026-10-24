"""Evaluation metrics (numpy only, so they are cheap to unit test).

Convention: label 1 = AI-generated (positive class), 0 = real.
False positive = a real image flagged as AI, which is the costly error for a trust-and-safety tool.
"""
import numpy as np


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def div(a, b):
        return a / b if b else 0.0

    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "fpr": div(fp, fp + tn),
        "f1": div(2 * precision * recall, precision + recall),
    }


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), tie-aware."""
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(len(a), dtype=float)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_prob)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_bins(y_true, y_prob, n_bins: int = 10):
    """Return (mean_predicted_prob, observed_positive_rate, count) per non-empty bin."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    idx = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append((float(y_prob[m].mean()), float(y_true[m].mean()), int(m.sum())))
    return out


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    bins = calibration_bins(y_true, y_prob, n_bins)
    total = sum(c for _, _, c in bins)
    return float(sum(abs(obs - pred) * c for pred, obs, c in bins) / total) if total else 0.0
