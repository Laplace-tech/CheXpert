from __future__ import annotations

# eval.py가 만든 study_predictions.csv를 읽어서
# 각 클래스별 최적 threshold를 고른다.

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chexpert_poc.common.config import load_config
from chexpert_poc.common.io import save_json
from chexpert_poc.evaluation.artifact_io import save_threshold_grid_csv
from chexpert_poc.evaluation.prediction_table import (
    extract_valid_label_arrays,
    find_latest_study_predictions_csv,
    load_prediction_rows,
)
from chexpert_poc.evaluation.thresholds import (
    VALID_CRITERIA,
    build_infer_thresholds_payload,
    choose_best_threshold,
    validate_threshold_grid,
)


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

    output_dir = pred_csv_path.parent / "threshold_tuning"
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
        y_true, y_prob, y_mask = extract_valid_label_arrays(rows, label)

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

    infer_thresholds_payload = build_infer_thresholds_payload(
        criterion=args.criterion,
        recommended_thresholds=recommended_thresholds,
        label_names=label_names,
        pred_csv_path=pred_csv_path,
        th_min=args.th_min,
        th_max=args.th_max,
        th_step=args.th_step,
    )
    save_json(infer_thresholds_payload, output_dir / "infer_thresholds.json")

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
    print(infer_thresholds_payload["thresholds_str_for_infer"])

    print("\nartifacts saved to:")
    print(output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()