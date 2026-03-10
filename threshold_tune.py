from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from chexpert_poc.utils.train_utils import load_config


VALID_CRITERIA = {"f1", "balanced_accuracy", "recall"}


def find_latest_study_predictions_csv(output_root: str | Path) -> Path:
    """
    output_root/train_runs 아래에서 가장 최근의 eval/study_predictions.csv를 찾는다.
    """
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


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def safe_div(n: float, d: float) -> float:
    if d == 0:
        return 0.0
    return float(n / d)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")

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
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
    }


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
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")

    if y_true.size == 0:
        raise ValueError("Cannot choose threshold from empty inputs")

    all_rows: list[dict[str, Any]] = []
    for th in thresholds:
        row = compute_binary_metrics(y_true, y_prob, threshold=th)
        all_rows.append(row)

    def sort_key(row: dict[str, Any]):
        primary = row[criterion]
        secondary = row["balanced_accuracy"]
        tertiary = -abs(row["threshold"] - 0.5)
        return (primary, secondary, tertiary)

    best_row = max(all_rows, key=sort_key)
    return best_row, all_rows


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_threshold_grid_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--pred-csv",
        type=str,
        default=None,
        help="explicit study_predictions.csv path; if omitted, latest eval file is used",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="f1",
        choices=sorted(VALID_CRITERIA),
        help="metric used to choose best threshold",
    )
    parser.add_argument("--th-min", type=float, default=0.05)
    parser.add_argument("--th-max", type=float, default=0.95)
    parser.add_argument("--th-step", type=float, default=0.01)
    args = parser.parse_args()

    config = load_config(args.config)
    label_names = list(config["data"]["target_labels"])
    output_root = Path(config["paths"]["output_root"])

    pred_csv_path = (
        Path(args.pred_csv)
        if args.pred_csv is not None
        else find_latest_study_predictions_csv(output_root)
    )

    rows = load_prediction_rows(pred_csv_path)
    thresholds = validate_threshold_grid(
        th_min=args.th_min,
        th_max=args.th_max,
        th_step=args.th_step,
    )

    eval_dir = pred_csv_path.parent
    output_dir = eval_dir / "threshold_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("threshold_tune.py start")
    print("=" * 100)
    print(f"prediction_csv : {pred_csv_path}")
    print(f"criterion      : {args.criterion}")
    print(f"output_dir     : {output_dir}")
    print(f"threshold_grid : [{args.th_min}, {args.th_max}] step={args.th_step}")

    best_summary: list[dict[str, Any]] = []
    all_grid_rows: list[dict[str, Any]] = []
    recommended_thresholds: list[float] = []

    for label in label_names:
        target_col = f"{label}_target"
        prob_col = f"{label}_prob"
        mask_col = f"{label}_mask"

        if target_col not in rows[0] or prob_col not in rows[0] or mask_col not in rows[0]:
            raise KeyError(f"Missing required columns for label={label}")

        y_true = np.asarray([float(r[target_col]) for r in rows], dtype=np.float32)
        y_prob = np.asarray([float(r[prob_col]) for r in rows], dtype=np.float32)
        y_mask = np.asarray([float(r[mask_col]) for r in rows], dtype=np.float32)

        valid = y_mask > 0.5
        y_true = y_true[valid]
        y_prob = y_prob[valid]

        positives = int((y_true == 1.0).sum())
        negatives = int((y_true == 0.0).sum())
        num_valid = int(valid.sum())
        unique = np.unique(y_true)

        if num_valid == 0:
            raise RuntimeError(f"No valid samples for label={label}")
        if unique.size < 2:
            print(
                f"[warn] label={label}: only one class present in valid set "
                f"(unique={unique.tolist()}). Threshold tuning is not informative."
            )

        best_row, grid_rows = choose_best_threshold(
            y_true=y_true,
            y_prob=y_prob,
            thresholds=thresholds,
            criterion=args.criterion,
        )

        best_row = {
            "label": label,
            "num_valid": num_valid,
            "positives": positives,
            "negatives": negatives,
            **best_row,
        }
        best_summary.append(best_row)
        recommended_thresholds.append(float(best_row["threshold"]))

        for row in grid_rows:
            all_grid_rows.append(
                {
                    "label": label,
                    "num_valid": num_valid,
                    "positives": positives,
                    "negatives": negatives,
                    **row,
                }
            )

    save_json(best_summary, output_dir / "best_thresholds_by_class.json")
    save_threshold_grid_csv(best_summary, output_dir / "best_thresholds_by_class.csv")
    save_threshold_grid_csv(all_grid_rows, output_dir / "threshold_grid_metrics.csv")

    threshold_string = ",".join(f"{x:.2f}" for x in recommended_thresholds)

    save_json(
        {
            "criterion": args.criterion,
            "thresholds": recommended_thresholds,
            "thresholds_str_for_infer": threshold_string,
            "labels": label_names,
            "prediction_csv": str(pred_csv_path),
            "threshold_grid": {
                "th_min": float(args.th_min),
                "th_max": float(args.th_max),
                "th_step": float(args.th_step),
            },
            "warning": (
                "These thresholds are tuned on the same validation predictions used "
                "to measure performance. Treat them as validation-tuned operating points, "
                "not unbiased generalization estimates."
            ),
        },
        output_dir / "infer_thresholds.json",
    )

    print("\n[best thresholds]")
    print(
        f"{'label':17s} | {'th':>6s} | {'F1':>8s} | {'Prec':>8s} | {'Rec':>8s} | {'Spec':>8s} | {'BalAcc':>8s}"
    )
    print("-" * 86)
    for row in best_summary:
        print(
            f"{row['label']:17s} | "
            f"{row['threshold']:6.2f} | "
            f"{row['f1']:8.4f} | "
            f"{row['precision']:8.4f} | "
            f"{row['recall']:8.4f} | "
            f"{row['specificity']:8.4f} | "
            f"{row['balanced_accuracy']:8.4f}"
        )

    print("\n[recommended --thresholds string for infer.py]")
    print(threshold_string)

    print("\nartifacts saved to:")
    print(output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()