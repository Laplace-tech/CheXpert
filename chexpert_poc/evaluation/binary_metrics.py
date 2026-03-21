from __future__ import annotations

from typing import Any

import numpy as np


def safe_div(n: float, d: float) -> float:
    if d == 0:
        return 0.0
    return float(n / d)


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def compute_binary_metrics_from_counts(
    counts: dict[str, int],
) -> dict[str, float]:
    tp = int(counts["tp"])
    tn = int(counts["tn"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = 0.5 * (recall + specificity)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
    }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}"
        )

    if y_true.ndim != 1:
        raise ValueError(f"Expected 1D arrays, got y_true.ndim={y_true.ndim}")

    unique = np.unique(y_true)
    allowed = {0.0, 1.0}
    for v in unique.tolist():
        if float(v) not in allowed:
            raise ValueError(f"Non-binary y_true detected: {unique.tolist()}")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0,1], got {threshold}")

    y_pred = (y_prob >= threshold).astype(np.int64)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)

    metrics = compute_binary_metrics_from_counts(
        {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    )

    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        **metrics,
    }