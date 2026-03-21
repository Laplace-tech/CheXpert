from __future__ import annotations

from typing import Any

from chexpert_poc.evaluation.prediction_table import (
    get_label_column_names,
    validate_required_columns,
)


def validate_top_n(top_n: int) -> int:
    top_n = int(top_n)
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")
    return top_n


def build_case_rows_for_label(
    rows: list[dict[str, str]],
    label: str,
    threshold: float,
) -> list[dict[str, Any]]:
    target_col, prob_col, mask_col = get_label_column_names(label)

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


def compute_confusion_counts_from_case_rows(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    tp = sum(1 for r in rows if r["error_type"] == "TP")
    tn = sum(1 for r in rows if r["error_type"] == "TN")
    fp = sum(1 for r in rows if r["error_type"] == "FP")
    fn = sum(1 for r in rows if r["error_type"] == "FN")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def sort_case_rows(
    rows: list[dict[str, Any]],
    error_type: str,
) -> list[dict[str, Any]]:
    if error_type not in {"FP", "FN", "TP", "TN"}:
        raise ValueError(f"Unsupported error_type: {error_type}")

    filtered = [r for r in rows if r["error_type"] == error_type]

    if error_type in {"FP", "TP"}:
        return sorted(filtered, key=lambda x: float(x["prob"]), reverse=True)

    return sorted(filtered, key=lambda x: float(x["prob"]))