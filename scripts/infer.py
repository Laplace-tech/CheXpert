from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chexpert_poc.common.config import get_config_bool, get_section, load_config
from chexpert_poc.common.io import ensure_dir, save_json
from chexpert_poc.common.model_config import resolve_num_classes
from chexpert_poc.common.runtime import get_device
from chexpert_poc.datasets.chexpert_dataset import build_image_transform

from chexpert_poc.inference.artifact_io import save_inference_predictions_csv
from chexpert_poc.inference.checkpoint import (
    load_checkpoint,
    resolve_checkpoint_path,
    validate_checkpoint_config,
)
from chexpert_poc.inference.input_io import collect_input_paths
from chexpert_poc.inference.postprocess import load_thresholds
from chexpert_poc.inference.predictor import (
    build_model_from_checkpoint,
    predict_one_image,
)

# =========================================================
# infer.py 역할
# ---------------------------------------------------------
# 학습된 체크포인트(best.pt)를 불러와서
# 입력 이미지(단일 파일 또는 폴더 전체)에 대해
# 확률 예측을 수행하고,
# threshold를 적용해 0/1 판정을 만들고,
# 결과를 JSON / CSV로 저장하는 파일
#
# 리팩토링 반영 사항
# 1) threshold / result build / csv save 로직을
#    chexpert_poc.inference.* 로 분리
# 2) predictor / checkpoint 로직도 분리
# 3) checkpoint 중복 로드 제거
# =========================================================


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =========================================================
# pipeline
# =========================================================

def run_inference_pipeline(
    *,
    config: dict[str, Any],
    input_path: str,
    checkpoint: str | None = None,
    recursive: bool = False,
    thresholds_arg: str | None = None,
) -> dict[str, Any]:
    # -----------------------------------------------------
    # 1) config / device 로드
    # -----------------------------------------------------
    device = get_device()

    # -----------------------------------------------------
    # 2) 주요 설정 읽기
    # -----------------------------------------------------
    data_cfg = get_section(config, "data")
    paths_cfg = get_section(config, "paths")

    label_names = list(data_cfg["target_labels"])
    image_size = int(data_cfg["image_size"])
    channels_last = get_config_bool(config, "train", "channels_last", default=True)
    num_classes = resolve_num_classes(config)

    # -----------------------------------------------------
    # 3) checkpoint 경로 결정 및 로드
    # -----------------------------------------------------
    checkpoint_path = resolve_checkpoint_path(
        output_root=paths_cfg["output_root"],
        checkpoint=checkpoint,
    )
    checkpoint_meta = load_checkpoint(checkpoint_path)

    # -----------------------------------------------------
    # 4) checkpoint config와 현재 config 최소 검증
    # -----------------------------------------------------
    validate_checkpoint_config(checkpoint_meta, config)

    # -----------------------------------------------------
    # 5) threshold 결정
    # -----------------------------------------------------
    thresholds, threshold_source = load_thresholds(
        checkpoint_path=checkpoint_path,
        label_names=label_names,
        thresholds_arg=thresholds_arg,
    )

    # -----------------------------------------------------
    # 6) 입력 이미지 목록 수집
    # -----------------------------------------------------
    input_paths = collect_input_paths(input_path, recursive=recursive)

    # -----------------------------------------------------
    # 7) 전처리 transform 생성
    # -----------------------------------------------------
    transform = build_image_transform(image_size=image_size)

    # -----------------------------------------------------
    # 8) 모델 로드
    # -----------------------------------------------------
    model = build_model_from_checkpoint(
        checkpoint=checkpoint_meta,
        num_classes=num_classes,
        device=device,
        channels_last=channels_last,
    )

    # -----------------------------------------------------
    # 9) 출력 폴더 준비
    # -----------------------------------------------------
    run_name = datetime.now().strftime("infer_%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(paths_cfg["output_root"]) / "infer_runs" / run_name)

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "num_inputs": len(input_paths),
        "recursive": bool(recursive),
        "labels": label_names,
        "num_classes": num_classes,
        "device": str(device),
        "channels_last": channels_last,
        "pretrained": False,  # checkpoint load 전제
    }

    print("=" * 100)
    print("infer.py start")
    print("=" * 100)
    print(f"device          : {device}")
    print(f"checkpoint_path : {checkpoint_path}")
    print(f"threshold_source: {threshold_source}")
    print(f"num_inputs      : {len(input_paths)}")
    print(f"output_dir      : {output_dir}")
    print(f"channels_last   : {channels_last}")
    print(f"num_classes     : {num_classes}")
    print(f"thresholds      : {thresholds}")

    # -----------------------------------------------------
    # 10) 이미지별 추론
    # -----------------------------------------------------
    all_predictions: list[dict[str, Any]] = []
    for image_path_obj in input_paths:
        result = predict_one_image(
            model=model,
            image_path=image_path_obj,
            transform=transform,
            device=device,
            label_names=label_names,
            thresholds=thresholds,
            channels_last=channels_last,
        )
        all_predictions.append(result)

    # -----------------------------------------------------
    # 11) 결과 저장
    # -----------------------------------------------------
    save_json(metadata, output_dir / "infer_metadata.json")
    save_json(all_predictions, output_dir / "predictions.json")
    save_inference_predictions_csv(
        predictions=all_predictions,
        output_csv_path=output_dir / "predictions.csv",
        label_names=label_names,
    )

    return {
        "output_dir": output_dir,
        "metadata": metadata,
        "predictions": all_predictions,
    }


# =========================================================
# main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="single image file or folder path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="optional explicit checkpoint path; if omitted, latest best.pt is used",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="when input is a folder, search recursively",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="comma-separated per-class thresholds, e.g. 0.5,0.5,0.5,0.5,0.5",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    result = run_inference_pipeline(
        config=config,
        input_path=args.input,
        checkpoint=args.checkpoint,
        recursive=args.recursive,
        thresholds_arg=args.thresholds,
    )

    output_dir = result["output_dir"]
    all_predictions = result["predictions"]

    print("\n[first 3 predictions]")
    for item in all_predictions[:3]:
        print(f"- input_path: {item['input_path']}")
        print(f"  positive_labels: {item['positive_labels']}")
        for pred in item["predictions"]:
            print(
                f"  {pred['label']}: "
                f"prob={pred['prob']:.4f}, "
                f"threshold={pred['threshold']:.2f}, "
                f"pred={pred['pred']}"
            )

    print("\nartifacts saved to:")
    print(output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()