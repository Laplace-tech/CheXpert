from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def find_latest_study_predictions_csv(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    candidates = list(output_root.glob("train_runs/*/eval/study_predictions.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No study_predictions.csv found under: {output_root / 'train_runs'}"
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def load_prediction_rows(csv_path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    return rows


def validate_required_columns(
    rows: list[dict[str, str]],
    required_columns: list[str],
) -> None:
    if not rows:
        raise RuntimeError("Prediction rows are empty")

    missing = [c for c in required_columns if c not in rows[0]]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def get_label_column_names(label: str) -> tuple[str, str, str]:
    return f"{label}_target", f"{label}_prob", f"{label}_mask"


def extract_valid_label_arrays(
    rows: list[dict[str, str]],
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_col, prob_col, mask_col = get_label_column_names(label)
    validate_required_columns(rows, [target_col, prob_col, mask_col])

    y_true = np.asarray([float(r[target_col]) for r in rows], dtype=np.float32)
    y_prob = np.asarray([float(r[prob_col]) for r in rows], dtype=np.float32)
    y_mask = np.asarray([float(r[mask_col]) for r in rows], dtype=np.float32)

    return y_true, y_prob, y_mask