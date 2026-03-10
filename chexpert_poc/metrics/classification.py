from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _to_float32_numpy(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _safe_binary_ranking_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """
    이진 분류 ranking metric(AUROC/AUPRC)을 안전하게 계산한다.

    계산 불가 상황:
    - y_true와 y_prob 길이 불일치
    - 클래스가 하나만 존재
    - sklearn 내부 예외 발생

    위 경우에는 nan을 반환한다.
    """
    y_true = _to_float32_numpy(y_true)
    y_prob = _to_float32_numpy(y_prob)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")

    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan")

    try:
        return float(metric_fn(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return _safe_binary_ranking_metric(y_true, y_prob, roc_auc_score)


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return _safe_binary_ranking_metric(y_true, y_prob, average_precision_score)


def _safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return float("nan")

    valid = ~np.isnan(arr)
    if not np.any(valid):
        return float("nan")

    return float(arr[valid].mean())


def _validate_binary_targets(y_true: np.ndarray, label_name: str) -> None:
    unique = np.unique(y_true)
    allowed = {0.0, 1.0}

    for v in unique.tolist():
        if float(v) not in allowed:
            raise ValueError(
                f"Non-binary target detected for label={label_name}: {unique.tolist()}"
            )


def compute_multilabel_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    loss_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    멀티라벨 분류 결과에 대해 class별 AUROC/AUPRC와 전체 평균을 계산한다.

    Args:
        y_true: shape [N, C], expected binary values in {0, 1}
        y_prob: shape [N, C], probability scores in [0, 1] expected
        label_names: length C
        loss_mask: shape [N, C] or None
            - None이면 모든 항목을 valid로 간주
            - 0이면 해당 위치는 metric 계산에서 제외

    Returns:
        per_class_results, summary
    """
    y_true = _to_float32_numpy(y_true)
    y_prob = _to_float32_numpy(y_prob)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")

    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y_true.ndim={y_true.ndim}")

    n_samples, n_classes = y_true.shape

    if len(label_names) != n_classes:
        raise ValueError(
            f"label_names length mismatch: len(label_names)={len(label_names)} vs classes={n_classes}"
        )

    if loss_mask is None:
        loss_mask = np.ones_like(y_true, dtype=np.float32)
    else:
        loss_mask = _to_float32_numpy(loss_mask)
        if loss_mask.shape != y_true.shape:
            raise ValueError(
                f"loss_mask shape mismatch: {loss_mask.shape} vs {y_true.shape}"
            )

    per_class_results: list[dict[str, Any]] = []
    aurocs: list[float] = []
    auprcs: list[float] = []

    for class_idx, label_name in enumerate(label_names):
        valid = loss_mask[:, class_idx] > 0.5

        y_true_c = y_true[valid, class_idx]
        y_prob_c = y_prob[valid, class_idx]

        if y_true_c.size > 0:
            _validate_binary_targets(y_true_c, label_name)

        auroc = safe_auroc(y_true_c, y_prob_c)
        auprc = safe_auprc(y_true_c, y_prob_c)

        positives = int((y_true_c == 1.0).sum())
        negatives = int((y_true_c == 0.0).sum())
        num_valid = int(valid.sum())

        per_class_results.append(
            {
                "label": label_name,
                "num_valid": num_valid,
                "positives": positives,
                "negatives": negatives,
                "auroc": auroc,
                "auprc": auprc,
            }
        )

        aurocs.append(auroc)
        auprcs.append(auprc)

    summary = {
        "mean_auroc": _safe_nanmean(aurocs),
        "mean_auprc": _safe_nanmean(auprcs),
        "num_samples": int(n_samples),
        "num_classes": int(n_classes),
    }

    return per_class_results, summary


def format_classification_metrics_table(per_class_results: list[dict[str, Any]]) -> str:
    lines = []
    lines.append(
        f"{'label':17s} | {'valid':>6s} | {'pos':>6s} | {'neg':>6s} | {'AUROC':>8s} | {'AUPRC':>8s}"
    )
    lines.append("-" * 72)

    for row in per_class_results:
        auroc = row["auroc"]
        auprc = row["auprc"]

        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "nan"
        auprc_str = f"{auprc:.4f}" if not np.isnan(auprc) else "nan"

        lines.append(
            f"{row['label']:17s} | "
            f"{row['num_valid']:6d} | "
            f"{row['positives']:6d} | "
            f"{row['negatives']:6d} | "
            f"{auroc_str:>8s} | "
            f"{auprc_str:>8s}"
        )

    return "\n".join(lines)