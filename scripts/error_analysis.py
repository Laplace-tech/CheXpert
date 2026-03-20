from __future__ import annotations

# eval.py가 만든 study_predictions.csv를 읽어서,
# 라벨별 threshold 기준으로 TP/TN/FP/FN을 다시 분류하고
# FP/FN/TP/TN 대표 사례와 클래스별 요약 통계를 저장하는 파일

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chexpert_poc.common.config import load_config
from chexpert_poc.common.io import save_json

def find_latest_study_predictions_csv(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    candidates = list(output_root.glob("train_runs/*/eval/study_predictions.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No study_predictions.csv found under: {output_root / 'train_runs'}"
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def find_thresholds_json_near_eval(pred_csv_path: str | Path) -> Path | None:
    pred_csv_path = Path(pred_csv_path)
    candidate = pred_csv_path.parent / "threshold_tuning" / "infer_thresholds.json"
    if candidate.exists():
        return candidate
    return None


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


def parse_thresholds_from_arg(thresholds_str: str, expected_len: int) -> list[float]:
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


def save_rows_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        actual_fieldnames = fieldnames or list(rows[0].keys())
    else:
        actual_fieldnames = fieldnames or []

    with open(path, "w", encoding="utf-8", newline="") as f:
        if not actual_fieldnames:
            f.write("")
            return

        writer = csv.DictWriter(f, fieldnames=actual_fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def sort_fp_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda x: float(x["prob"]), reverse=True)


def sort_fn_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda x: float(x["prob"]))


def sort_tp_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 가장 확신 높은 TP를 위로
    return sorted(rows, key=lambda x: float(x["prob"]), reverse=True)


def sort_tn_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 가장 확신 높은 TN(확률이 가장 낮은 음성)을 위로
    return sorted(rows, key=lambda x: float(x["prob"]))


def compute_confusion_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    tp = sum(1 for r in rows if r["error_type"] == "TP")
    tn = sum(1 for r in rows if r["error_type"] == "TN")
    fp = sum(1 for r in rows if r["error_type"] == "FP")
    fn = sum(1 for r in rows if r["error_type"] == "FN")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_div(n: float, d: float) -> float:
    if d == 0:
        return 0.0
    return float(n / d)


def compute_binary_metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

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
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }


def validate_top_n(top_n: int) -> int:
    top_n = int(top_n)
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")
    return top_n


def validate_required_columns(rows: list[dict[str, str]], required_columns: list[str]) -> None:
    if not rows:
        raise RuntimeError("Prediction rows are empty")

    missing = [c for c in required_columns if c not in rows[0]]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def build_case_rows_for_label(
    rows: list[dict[str, str]],
    label: str,
    threshold: float,
) -> list[dict[str, Any]]:
    target_col = f"{label}_target"
    prob_col = f"{label}_prob"
    mask_col = f"{label}_mask"

    validate_required_columns(
        rows,
        ["path", "study_id", target_col, prob_col, mask_col],
    )

    detailed_rows: list[dict[str, Any]] = []

    for row in rows:
        mask = float(row[mask_col])
        if mask <= 0.5:
            continue

        target = int(float(row[target_col]))
        prob = float(row[prob_col])
        pred = int(prob >= threshold)

        if target == 1 and pred == 1:
            error_type = "TP"
        elif target == 0 and pred == 0:
            error_type = "TN"
        elif target == 0 and pred == 1:
            error_type = "FP"
        elif target == 1 and pred == 0:
            error_type = "FN"
        else:
            raise RuntimeError("Unexpected confusion state")

        detailed_rows.append(
            {
                "label": label,
                "path": row["path"],
                "study_id": row["study_id"],
                "target": target,
                "prob": prob,
                "threshold": float(threshold),
                "pred": pred,
                "error_type": error_type,
            }
        )

    return detailed_rows


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
        "--thresholds",
        type=str,
        default=None,
        help="comma-separated thresholds; if omitted, threshold_tuning/infer_thresholds.json is used when available",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="number of top FP/FN/TP/TN samples to save per class",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    label_names = list(config["data"]["target_labels"])
    output_root = Path(config["paths"]["output_root"])
    top_n = validate_top_n(args.top_n)

    pred_csv_path = (
        Path(args.pred_csv)
        if args.pred_csv is not None
        else find_latest_study_predictions_csv(output_root)
    )

    rows = load_prediction_rows(pred_csv_path)

    thresholds, threshold_source = load_thresholds(
        pred_csv_path=pred_csv_path,
        label_names=label_names,
        thresholds_arg=args.thresholds,
    )

    if len(thresholds) != len(label_names):
        raise ValueError(
            f"Threshold count mismatch: len(thresholds)={len(thresholds)} "
            f"vs len(label_names)={len(label_names)}"
        )

    output_dir = pred_csv_path.parent / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("error_analysis.py start")
    print("=" * 100)
    print(f"prediction_csv    : {pred_csv_path}")
    print(f"threshold_source  : {threshold_source}")
    print(f"thresholds        : {thresholds}")
    print(f"output_dir        : {output_dir}")
    print(f"top_n             : {top_n}")

    summary_rows: list[dict[str, Any]] = []

    row_fieldnames = [
        "label",
        "path",
        "study_id",
        "target",
        "prob",
        "threshold",
        "pred",
        "error_type",
    ]

    for label, threshold in zip(label_names, thresholds):
        detailed_rows = build_case_rows_for_label(
            rows=rows,
            label=label,
            threshold=threshold,
        )

        counts = compute_confusion_counts(detailed_rows)
        metrics = compute_binary_metrics_from_counts(counts)

        positives = sum(1 for r in detailed_rows if r["target"] == 1)
        negatives = sum(1 for r in detailed_rows if r["target"] == 0)

        summary_row = {
            "label": label,
            "threshold": float(threshold),
            "num_valid": len(detailed_rows),
            "positives": positives,
            "negatives": negatives,
            **counts,
            **metrics,
        }
        summary_rows.append(summary_row)

        fp_rows = sort_fp_rows([r for r in detailed_rows if r["error_type"] == "FP"])
        fn_rows = sort_fn_rows([r for r in detailed_rows if r["error_type"] == "FN"])
        tp_rows = sort_tp_rows([r for r in detailed_rows if r["error_type"] == "TP"])
        tn_rows = sort_tn_rows([r for r in detailed_rows if r["error_type"] == "TN"])

        class_dir = output_dir / label.replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

        save_rows_csv(detailed_rows, class_dir / "all_cases.csv", fieldnames=row_fieldnames)
        save_rows_csv(fp_rows[:top_n], class_dir / f"top_{top_n}_fp.csv", fieldnames=row_fieldnames)
        save_rows_csv(fn_rows[:top_n], class_dir / f"top_{top_n}_fn.csv", fieldnames=row_fieldnames)
        save_rows_csv(tp_rows[:top_n], class_dir / f"top_{top_n}_tp.csv", fieldnames=row_fieldnames)
        save_rows_csv(tn_rows[:top_n], class_dir / f"top_{top_n}_tn.csv", fieldnames=row_fieldnames)

    analysis_metadata = {
        "prediction_csv": str(pred_csv_path),
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "top_n": top_n,
        "labels": label_names,
    }

    save_json(analysis_metadata, output_dir / "analysis_metadata.json")
    save_json(summary_rows, output_dir / "summary_by_class.json")
    save_rows_csv(summary_rows, output_dir / "summary_by_class.csv")

    print("\n[summary by class]")
    print(
        f"{'label':17s} | {'th':>5s} | {'FP':>4s} | {'FN':>4s} | {'Prec':>7s} | {'Rec':>7s} | {'F1':>7s}"
    )
    print("-" * 72)
    for row in summary_rows:
        print(
            f"{row['label']:17s} | "
            f"{row['threshold']:5.2f} | "
            f"{row['fp']:4d} | "
            f"{row['fn']:4d} | "
            f"{row['precision']:7.4f} | "
            f"{row['recall']:7.4f} | "
            f"{row['f1']:7.4f}"
        )

    print("\nartifacts saved to:")
    print(output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()