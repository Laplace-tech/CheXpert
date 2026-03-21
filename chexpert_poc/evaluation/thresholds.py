from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from chexpert_poc.evaluation.binary_metrics import compute_binary_metrics


VALID_CRITERIA = {"f1", "balanced_accuracy", "recall"}


def validate_threshold_grid(
    th_min: float,
    th_max: float,
    th_step: float,
) -> list[float]:
    th_min = float(th_min)
    th_max = float(th_max)
    th_step = float(th_step)

    if not (0.0 <= th_min <= 1.0):
        raise ValueError(f"th-min must be in [0,1], got {th_min}")
    if not (0.0 <= th_max <= 1.0):
        raise ValueError(f"th-max must be in [0,1], got {th_max}")
    if th_min > th_max:
        raise ValueError(f"th-min must be <= th-max, got {th_min} > {th_max}")
    if th_step <= 0.0:
        raise ValueError(f"th-step must be > 0, got {th_step}")

    thresholds = np.arange(th_min, th_max + 1e-12, th_step, dtype=np.float64)
    thresholds = [float(x) for x in thresholds]

    if not thresholds:
        raise RuntimeError("Threshold grid is empty")

    return thresholds


def choose_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
    criterion: str = "f1",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if criterion not in VALID_CRITERIA:
        raise ValueError(f"Unsupported criterion: {criterion}")

    if not thresholds:
        raise ValueError("thresholds must not be empty")

    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}"
        )

    if y_true.size == 0:
        raise ValueError("Cannot choose threshold from empty inputs")

    all_rows: list[dict[str, Any]] = []
    for th in thresholds:
        row = compute_binary_metrics(y_true, y_prob, threshold=th)
        all_rows.append(row)

    def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
        primary = float(row[criterion])
        secondary = float(row["balanced_accuracy"])
        tertiary = -abs(float(row["threshold"]) - 0.5)
        return (primary, secondary, tertiary)

    best_row = max(all_rows, key=sort_key)
    return best_row, all_rows


def build_infer_thresholds_payload(
    *,
    criterion: str,
    recommended_thresholds: list[float],
    label_names: list[str],
    pred_csv_path: Path,
    th_min: float,
    th_max: float,
    th_step: float,
) -> dict[str, Any]:
    threshold_string = ",".join(f"{x:.2f}" for x in recommended_thresholds)

    return {
        "criterion": criterion,
        "thresholds": recommended_thresholds,
        "thresholds_str_for_infer": threshold_string,
        "labels": label_names,
        "prediction_csv": str(pred_csv_path),
        "threshold_grid": {
            "th_min": float(th_min),
            "th_max": float(th_max),
            "th_step": float(th_step),
        },
        "warning": (
            "These thresholds are tuned on the same validation predictions used "
            "to measure performance. Treat them as validation-tuned operating points, "
            "not unbiased generalization estimates."
        ),
    }


def find_thresholds_json_near_eval(pred_csv_path: str | Path) -> Path | None:
    pred_csv_path = Path(pred_csv_path)
    candidate = pred_csv_path.parent / "threshold_tuning" / "infer_thresholds.json"
    if candidate.exists():
        return candidate
    return None


def parse_thresholds_from_arg(
    thresholds_str: str,
    expected_len: int,
) -> list[float]:
    parts = [x.strip() for x in thresholds_str.split(",") if x.strip()]
    if len(parts) != expected_len:
        raise ValueError(
            f"Expected {expected_len} thresholds, got {len(parts)}: {thresholds_str}"
        )

    thresholds = [float(x) for x in parts]
    for t in thresholds:
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"Threshold must be in [0,1], got {t}")

    return thresholds


def load_thresholds(
    pred_csv_path: str | Path,
    label_names: list[str],
    thresholds_arg: str | None,
) -> tuple[list[float], str]:
    if thresholds_arg is not None:
        return parse_thresholds_from_arg(thresholds_arg, len(label_names)), "cli"

    threshold_json_path = find_thresholds_json_near_eval(pred_csv_path)
    if threshold_json_path is not None:
        with open(threshold_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        thresholds = data.get("thresholds", None)
        if thresholds is None or len(thresholds) != len(label_names):
            raise ValueError(f"Invalid thresholds json: {threshold_json_path}")

        thresholds = [float(x) for x in thresholds]
        for t in thresholds:
            if not (0.0 <= t <= 1.0):
                raise ValueError(f"Threshold must be in [0,1], got {t}")

        json_labels = data.get("labels", None)
        if json_labels is not None and list(json_labels) != list(label_names):
            raise ValueError(
                f"Threshold label order mismatch: {threshold_json_path}"
            )

        return thresholds, str(threshold_json_path)

    return [0.5] * len(label_names), "default_0.5"