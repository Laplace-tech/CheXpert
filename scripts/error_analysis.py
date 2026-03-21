from __future__ import annotations

# eval.py가 만든 study_predictions.csv를 읽어서,
# 라벨별 threshold 기준으로 TP/TN/FP/FN을 다시 분류하고
# FP/FN/TP/TN 대표 사례와 클래스별 요약 통계를 저장하는 파일

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chexpert_poc.common.config import load_config
from chexpert_poc.common.io import save_json
from chexpert_poc.evaluation.artifact_io import save_rows_csv
from chexpert_poc.evaluation.binary_metrics import compute_binary_metrics_from_counts
from chexpert_poc.evaluation.error_analysis import (
    build_case_rows_for_label,
    compute_confusion_counts_from_case_rows,
    sort_case_rows,
    validate_top_n,
)
from chexpert_poc.evaluation.prediction_table import (
    find_latest_study_predictions_csv,
    load_prediction_rows,
)
from chexpert_poc.evaluation.thresholds import load_thresholds


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

        counts = compute_confusion_counts_from_case_rows(detailed_rows)
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

        fp_rows = sort_case_rows(detailed_rows, "FP")
        fn_rows = sort_case_rows(detailed_rows, "FN")
        tp_rows = sort_case_rows(detailed_rows, "TP")
        tn_rows = sort_case_rows(detailed_rows, "TN")

        class_dir = output_dir / label.replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

        save_rows_csv(
            detailed_rows,
            class_dir / "all_cases.csv",
            fieldnames=row_fieldnames,
        )
        save_rows_csv(
            fp_rows[:top_n],
            class_dir / f"top_{top_n}_fp.csv",
            fieldnames=row_fieldnames,
        )
        save_rows_csv(
            fn_rows[:top_n],
            class_dir / f"top_{top_n}_fn.csv",
            fieldnames=row_fieldnames,
        )
        save_rows_csv(
            tp_rows[:top_n],
            class_dir / f"top_{top_n}_tp.csv",
            fieldnames=row_fieldnames,
        )
        save_rows_csv(
            tn_rows[:top_n],
            class_dir / f"top_{top_n}_tn.csv",
            fieldnames=row_fieldnames,
        )

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