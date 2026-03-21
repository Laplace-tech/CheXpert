from __future__ import annotations


import csv               # 표쥰 라이브러리 csv 모듈
from pathlib import Path # 파일 경로를 문자열 대신 객체처럼 다루기 위한 모듈
from typing import Any   # 딕셔너리 내부 값 타입이 제각각일 때 Any로 받기 위해 사용


# predictions.json과 별도로 csv 저장
def save_inference_predictions_csv(
    predictions: list[dict[str, Any]], # 예측 결과 여러 개를 담은 리스트
    output_csv_path: str | Path,       # 저장할 CSV 파일 경로
    label_names: list[str],            # 클래스 이름 목록
) -> None:
    
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["input_path", "positive_labels"]
    for label_name in label_names:
        fieldnames.extend(
            [
                f"{label_name}_prob",
                f"{label_name}_threshold",
                f"{label_name}_pred",
            ]
        )

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in predictions:
            row = {
                "input_path": item["input_path"],
                "positive_labels": ",".join(item["positive_labels"]),
            }

            pred_map = {x["label"]: x for x in item["predictions"]}
            for label_name in label_names:
                row[f"{label_name}_prob"] = pred_map[label_name]["prob"]
                row[f"{label_name}_threshold"] = pred_map[label_name]["threshold"]
                row[f"{label_name}_pred"] = pred_map[label_name]["pred"]

            writer.writerow(row)