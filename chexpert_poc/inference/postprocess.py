# threshold / 결과 조립

from __future__ import annotations

import json              # infer_thresholds.json 읽을 때 사용
from pathlib import Path # 체크포인트 경로, threshold json 경로를 OS 독립적으로 다루기 위해 사용 
from typing import Any   # build_prediction_result의 반환 dict 내부의 값 타입이 섞여 있어서 사용

# 사용자가 CLI에서 직접 넣은 threshold 문자열을 실제 float 리스트로 바꾸는 함수
def parse_thresholds_from_string(
    thresholds_str: str,
    label_names: list[str],
) -> list[float]:
    # CLI 문자열 "0.5,0.5,0.5,0.5,0.5" -> float 리스트
    parts = [x.strip() for x in thresholds_str.split(",") if x.strip()]
    if len(parts) != len(label_names):
        raise ValueError(
            f"Expected {len(label_names)} thresholds, got {len(parts)}. "
            f"Example: 0.5,0.5,0.5,0.5,0.5"
        )

    thresholds = [float(x) for x in parts]
    for t in thresholds:
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"Threshold must be in [0,1], got {t}")

    return thresholds

# 특정 checkpoint(best.pt)에 대응되는 threshold tuning 결과 json 파일 경로를 추정해서 찾는 함수
def find_thresholds_json_for_checkpoint(checkpoint_path: str | Path) -> Path | None:
    """
    같은 run의 eval/threshold_tuning/infer_thresholds.json을 찾는다.

    기대 구조:
    .../train_runs/<run_name>/checkpoints/best.pt
    -> .../train_runs/<run_name>/eval/threshold_tuning/infer_thresholds.json
    """
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent.parent
    candidate = run_dir / "eval" / "threshold_tuning" / "infer_thresholds.json"
    if candidate.exists():
        return candidate
    return None


# 실제 추론에 사용할 threshold들을 최종 결정하는 함수
def load_thresholds(
    checkpoint_path: str | Path,
    label_names: list[str],
    thresholds_arg: str | None,
) -> tuple[list[float], str]:
    # threshold 우선순위
    # 1) CLI --thresholds
    # 2) 같은 run의 infer_thresholds.json
    # 3) 없으면 전부 0.5

    if thresholds_arg is not None:
        return parse_thresholds_from_string(thresholds_arg, label_names), "cli"

    threshold_json_path = find_thresholds_json_for_checkpoint(checkpoint_path)
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

        # threshold 파일의 label 순서가 현재 infer label 순서와 같은지 검증
        json_labels = data.get("labels", None)
        if json_labels is not None and list(json_labels) != list(label_names):
            raise ValueError(
                f"Threshold label order mismatch: {threshold_json_path}"
            )

        return thresholds, str(threshold_json_path)

    return [0.5] * len(label_names), "default_0.5"

# 모델이 낸 확률(prob list) + threshold list를 받아서 
# 최종 예측 결과를 저장/출력하기 좋은 dict 형태로 만드는 함수
def build_prediction_result(
    *,
    input_path: str,
    label_names: list[str],
    probs: list[float],
    thresholds: list[float],
) -> dict[str, Any]:
    if len(label_names) != len(probs):
        raise ValueError(
            f"label_names/probs length mismatch: {len(label_names)} vs {len(probs)}"
        )
    if len(label_names) != len(thresholds):
        raise ValueError(
            "label_names/thresholds length mismatch: "
            f"{len(label_names)} vs {len(thresholds)}"
        )

    predictions: list[dict[str, Any]] = []
    positive_labels: list[str] = []

    for label_name, prob, threshold in zip(label_names, probs, thresholds):
        pred = int(prob >= threshold)

        predictions.append(
            {
                "label": label_name,
                "prob": float(prob),
                "threshold": float(threshold),
                "pred": pred,
            }
        )

        if pred == 1:
            positive_labels.append(label_name)

    return {
        "input_path": input_path,
        "positive_labels": positive_labels,
        "predictions": predictions,
    }