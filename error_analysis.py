from __future__ import annotations

# eval.py가 저장한 study_predictions.csv를 읽어서,
# 각 라벨(class)별로
#   - TP / TN / FP / FN 을 다시 분류하고
#   - FP / FN 상위 사례를 뽑고
#   - 클래스별 precision / recall / F1 같은 요약 통계를 저장하는 파일
#
# eval.py: 전체 성능이 얼마나 좋은가?
# error_analysis.py: 어떤 케이스에서 틀렸는가?
#
# 저장되는 것:
# - summary_by_class.json / csv
# - label별 all_cases.csv
# - label별 top_N_fp.csv / top_N_fn.csv / top_N_tp.csv / top_N_tn.csv
#
# 이미 eval.py가 만든 study_predictions.csv를 사후 분석하는 파일이다.
# =========================================================

import argparse          # CLI 인자 처리
import csv               # study_predictions.csv 읽기 / 분석 결과 csv 저장
import json              # threshold json / summary json 저장
from pathlib import Path # 파일 경로 처리
from typing import Any

# [연계: train_utils.py]
# - base.yaml 로드용
from chexpert_poc.utils.train_utils import load_config


def find_latest_study_predictions_csv(output_root: str | Path) -> Path:
    """
    output_root/train_runs 아래에서 가장 최근의 eval/study_predictions.csv를 찾는다.
    """
    # [역할]
    # - 사용자가 --pred-csv를 직접 안 주면
    #   outputs/train_runs/*/eval/study_predictions.csv 중
    #   가장 최근 파일을 자동 선택
    #
    # [연계: eval.py]
    # - eval.py가 저장한 study_predictions.csv를 입력으로 사용
    output_root = Path(output_root)
    candidates = list(output_root.glob("train_runs/*/eval/study_predictions.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No study_predictions.csv found under: {output_root / 'train_runs'}"
        )

    # 가장 최근 수정된 파일을 선택
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def find_thresholds_json_near_eval(pred_csv_path: str | Path) -> Path | None:
    # [역할]
    # - 현재 prediction csv와 같은 eval 폴더 밑에서
    #   threshold_tuning/infer_thresholds.json 파일을 찾음
    #
    # [의미]
    # - threshold_tune.py를 이미 돌렸으면
    #   error_analysis도 그 tuned threshold를 그대로 재사용할 수 있음
    pred_csv_path = Path(pred_csv_path)
    candidate = pred_csv_path.parent / "threshold_tuning" / "infer_thresholds.json"
    if candidate.exists():
        return candidate
    return None


def load_prediction_rows(csv_path: str | Path) -> list[dict[str, str]]:
    # [역할]
    # - study_predictions.csv를 읽어서
    #   list[dict] 형태로 메모리에 올림
    #
    # [입력 예시 컬럼]
    # path, study_id,
    # Atelectasis_target, Atelectasis_prob, Atelectasis_mask,
    # Cardiomegaly_target, ...
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
    # [역할]
    # - CLI로 받은 threshold 문자열
    #   "0.5,0.5,0.5,0.5,0.5"
    #   를 float 리스트로 변환
    #
    # [중요]
    # - threshold 개수는 라벨 개수와 같아야 함
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
    # =====================================================
    # error analysis에 사용할 threshold를 결정
    # =====================================================
    #
    # 우선순위:
    # 1) CLI --thresholds 직접 입력
    # 2) 같은 eval 폴더의 threshold_tuning/infer_thresholds.json
    # 3) 없으면 전부 0.5
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

        return thresholds, str(threshold_json_path)

    # threshold tuning 결과가 없으면 기본 0.5
    return [0.5] * len(label_names), "default_0.5"


def save_json(data: Any, path: str | Path) -> None:
    # [역할]
    # - 분석 결과를 json으로 저장
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_rows_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> None:
    # [역할]
    # - dict row 리스트를 CSV로 저장
    #
    # [사용처]
    # - all_cases.csv
    # - top_N_fp.csv
    # - top_N_fn.csv
    # - top_N_tp.csv
    # - top_N_tn.csv
    # - summary_by_class.csv
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
    # [FP 정렬 기준]
    # - 실제 음성인데 높게 예측한 순서
    # - 즉 "확신을 가지고 틀린 false positive"가 위로 올라옴
    return sorted(rows, key=lambda x: float(x["prob"]), reverse=True)


def sort_fn_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # [FN 정렬 기준]
    # - 실제 양성인데 낮게 예측한 순서
    # - 즉 "놓친 false negative 중에서도 확률이 특히 낮았던 사례"가 위로 올라옴
    return sorted(rows, key=lambda x: float(x["prob"]))


def compute_confusion_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    # [역할]
    # - detailed row 리스트에서 TP/TN/FP/FN 개수 집계
    tp = sum(1 for r in rows if r["error_type"] == "TP")
    tn = sum(1 for r in rows if r["error_type"] == "TN")
    fp = sum(1 for r in rows if r["error_type"] == "FP")
    fn = sum(1 for r in rows if r["error_type"] == "FN")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_div(n: float, d: float) -> float:
    # [역할]
    # - 0으로 나누기 방지
    if d == 0:
        return 0.0
    return float(n / d)


def compute_binary_metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    # =====================================================
    # confusion count에서 precision / recall / F1 등 계산
    # =====================================================
    #
    # [입력]
    # counts = {"tp": ..., "tn": ..., "fp": ..., "fn": ...}
    #
    # [출력]
    # - precision
    # - recall
    # - specificity
    # - accuracy
    # - balanced_accuracy
    # - f1
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
    # [역할]
    # - top_n이 1 이상인지 검증
    top_n = int(top_n)
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")
    return top_n


def validate_required_columns(rows: list[dict[str, str]], required_columns: list[str]) -> None:
    # [역할]
    # - prediction csv에 필요한 컬럼이 실제로 있는지 검사
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
    # =====================================================
    # 특정 라벨 1개에 대해 case row들을 만드는 함수
    # =====================================================
    #
    # [입력]
    # - rows: study_predictions.csv 전체 row
    # - label: 예) "Pleural Effusion"
    # - threshold: 예) 0.37
    #
    # [출력]
    # - 각 case마다
    #   path, study_id, target, prob, threshold, pred, error_type
    #   를 담은 detailed row 리스트
    #
    # [핵심]
    # - 여기서 prob >= threshold 로 pred를 다시 계산
    # - 그리고 TP/TN/FP/FN(error_type)을 부여
    target_col = f"{label}_target"
    prob_col = f"{label}_prob"
    mask_col = f"{label}_mask"

    validate_required_columns(
        rows,
        ["path", "study_id", target_col, prob_col, mask_col],
    )

    detailed_rows: list[dict[str, Any]] = []

    for row in rows:
        # mask <= 0.5면 valid sample이 아니므로 분석 제외
        mask = float(row[mask_col])
        if mask <= 0.5:
            continue

        target = int(float(row[target_col]))
        prob = float(row[prob_col])
        pred = int(prob >= threshold)

        # confusion type 결정
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
    # -----------------------------------------------------
    # 1) CLI 인자
    # -----------------------------------------------------
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
        help="number of top FP/FN samples to save per class",
    )
    args = parser.parse_args()

    # -----------------------------------------------------
    # 2) config 로드
    # -----------------------------------------------------
    config = load_config(args.config)
    label_names = list(config["data"]["target_labels"])
    output_root = Path(config["paths"]["output_root"])
    top_n = validate_top_n(args.top_n)

    # -----------------------------------------------------
    # 3) prediction csv 경로 결정
    # -----------------------------------------------------
    pred_csv_path = (
        Path(args.pred_csv)
        if args.pred_csv is not None
        else find_latest_study_predictions_csv(output_root)
    )

    # study_predictions.csv 전체 로드
    rows = load_prediction_rows(pred_csv_path)

    # -----------------------------------------------------
    # 4) threshold 결정
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 5) 출력 폴더 준비
    # -----------------------------------------------------
    # 예:
    # outputs/train_runs/run_xxx/eval/error_analysis/
    eval_dir = pred_csv_path.parent
    output_dir = eval_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("error_analysis.py start")
    print("=" * 100)
    print(f"prediction_csv    : {pred_csv_path}")
    print(f"threshold_source  : {threshold_source}")
    print(f"thresholds        : {thresholds}")
    print(f"output_dir        : {output_dir}")
    print(f"top_n             : {top_n}")

    # summary_rows:
    # - 클래스별 confusion count / metric 요약
    summary_rows: list[dict[str, Any]] = []

    # detailed row csv 공통 필드 순서
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

    # -----------------------------------------------------
    # 6) 라벨별 상세 에러 분석
    # -----------------------------------------------------
    for label, threshold in zip(label_names, thresholds):
        # 현재 라벨 하나에 대해
        # 모든 케이스를 TP/TN/FP/FN으로 분류
        detailed_rows = build_case_rows_for_label(
            rows=rows,
            label=label,
            threshold=threshold,
        )

        # confusion count
        counts = compute_confusion_counts(detailed_rows)

        # precision / recall / F1 등 계산
        metrics = compute_binary_metrics_from_counts(counts)

        positives = sum(1 for r in detailed_rows if r["target"] == 1)
        negatives = sum(1 for r in detailed_rows if r["target"] == 0)

        # 클래스별 summary row 생성
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

        # -------------------------------------------------
        # error_type별 row 분리
        # -------------------------------------------------
        fp_rows = sort_fp_rows([r for r in detailed_rows if r["error_type"] == "FP"])
        fn_rows = sort_fn_rows([r for r in detailed_rows if r["error_type"] == "FN"])
        tp_rows = [r for r in detailed_rows if r["error_type"] == "TP"]
        tn_rows = [r for r in detailed_rows if r["error_type"] == "TN"]

        # 라벨별 폴더
        # 예:
        # error_analysis/Pleural_Effusion/
        class_dir = output_dir / label.replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

        # 모든 케이스 저장
        save_rows_csv(detailed_rows, class_dir / "all_cases.csv", fieldnames=row_fieldnames)

        # 상위 FP/FN/TP/TN 저장
        save_rows_csv(fp_rows[:top_n], class_dir / f"top_{top_n}_fp.csv", fieldnames=row_fieldnames)
        save_rows_csv(fn_rows[:top_n], class_dir / f"top_{top_n}_fn.csv", fieldnames=row_fieldnames)
        save_rows_csv(tp_rows[:top_n], class_dir / f"top_{top_n}_tp.csv", fieldnames=row_fieldnames)
        save_rows_csv(tn_rows[:top_n], class_dir / f"top_{top_n}_tn.csv", fieldnames=row_fieldnames)

    # -----------------------------------------------------
    # 7) 전체 분석 메타데이터 저장
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 8) 콘솔 출력
    # -----------------------------------------------------
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