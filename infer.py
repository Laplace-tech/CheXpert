from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from dotenv import load_dotenv
from PIL import Image

from chexpert_poc.datasets.chexpert_dataset import build_image_transform
from chexpert_poc.datasets.labels import CHEXPERT_5_LABELS
from chexpert_poc.models.densenet import build_densenet121
from chexpert_poc.utils.losses import logits_to_probs
from chexpert_poc.utils.train_utils import ensure_dir, get_device, load_config, save_json


# =========================================================
# infer.py 역할
# ---------------------------------------------------------
# 학습된 체크포인트(best.pt)를 불러와서
# 입력 이미지(단일 파일 또는 폴더 전체)에 대해
# 확률 예측을 수행하고,
# threshold를 적용해 0/1 판정을 만들고,
# 결과를 JSON / CSV로 저장하는 파일
#
# 핵심 반영 사항
# 1) bool(config[...]) 패턴 제거
# 2) num_classes는 len(data.target_labels) 기준으로 일원화
# 3) checkpoint 저장 당시 target_labels와 현재 config의 target_labels 최소 검증
# =========================================================


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================================================
# config helper
# =========================================================

def get_section(config: dict, section: str) -> dict[str, Any]:
    section_value = config.get(section)
    if not isinstance(section_value, dict):
        raise TypeError(
            f"config['{section}'] must be dict, got {type(section_value).__name__}"
        )
    return section_value


def require_bool(name: str, value: Any) -> bool:
    # 문자열 "false"를 bool("false") == True 로 잘못 해석하지 않게
    # 진짜 bool만 허용
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be bool, got {type(value).__name__}: {value!r}")


def get_config_bool(
    config: dict,
    section: str,
    key: str,
    default: bool | None = None,
) -> bool:
    section_dict = get_section(config, section)

    if key in section_dict:
        return require_bool(f"{section}.{key}", section_dict[key])

    if default is not None:
        return default

    raise KeyError(f"Missing required bool config: {section}.{key}")


def resolve_num_classes(config: dict) -> int:
    # num_classes의 source of truth를 data.target_labels 길이로 맞춤
    data_cfg = get_section(config, "data")
    model_cfg = get_section(config, "model")

    target_labels = data_cfg.get("target_labels")
    if not isinstance(target_labels, Sequence) or isinstance(target_labels, (str, bytes)):
        raise TypeError("data.target_labels must be a non-string sequence")
    if len(target_labels) == 0:
        raise ValueError("data.target_labels must not be empty")

    derived_num_classes = len(target_labels)

    # model.num_classes가 있으면 일치 여부만 검증
    if "num_classes" in model_cfg:
        configured_num_classes = int(model_cfg["num_classes"])
        if configured_num_classes != derived_num_classes:
            raise ValueError(
                "model.num_classes does not match len(data.target_labels): "
                f"{configured_num_classes} vs {derived_num_classes}"
            )

    return derived_num_classes


def validate_checkpoint_config(
    checkpoint: dict[str, Any],
    current_config: dict[str, Any],
) -> None:
    # checkpoint 저장 당시 config와 현재 infer config가 너무 다르면
    # label 순서 / 클래스 수 해석이 꼬일 수 있으므로 최소 검증
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return

    if not isinstance(checkpoint_config, dict):
        raise TypeError(
            f"checkpoint['config'] must be dict when present, got "
            f"{type(checkpoint_config).__name__}"
        )

    current_labels = list(get_section(current_config, "data")["target_labels"])
    checkpoint_labels = checkpoint_config.get("data", {}).get("target_labels")

    if checkpoint_labels is not None and list(checkpoint_labels) != current_labels:
        raise ValueError(
            "Current config data.target_labels does not match checkpoint config. "
            f"current={current_labels}, checkpoint={list(checkpoint_labels)}"
        )


# =========================================================
# tensor helper
# =========================================================

def move_tensor(
    x: torch.Tensor,
    device: torch.device,
    channels_last: bool = False,
) -> torch.Tensor:
    # image tensor([B,C,H,W])이고 channels_last=True면
    # memory_format=torch.channels_last로 device 이동
    if channels_last and x.ndim == 4:
        return x.to(
            device=device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return x.to(device=device, non_blocking=True)


# =========================================================
# checkpoint / threshold helper
# =========================================================

def find_latest_best_checkpoint(output_root: str | Path) -> Path:
    # outputs/train_runs/*/checkpoints/best.pt 중 최신 파일 하나 선택
    output_root = Path(output_root)
    candidates = list(output_root.glob("train_runs/*/checkpoints/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt found under: {output_root / 'train_runs'}"
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def resolve_checkpoint_path(
    output_root: str | Path,
    checkpoint: str | Path | None,
) -> Path:
    # --checkpoint를 직접 줬으면 그걸 사용
    # 안 줬으면 최신 best.pt 자동 선택
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    return find_latest_best_checkpoint(output_root)


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


# =========================================================
# input helper
# =========================================================

def collect_input_paths(input_path: str | Path, recursive: bool = False) -> list[Path]:
    # 입력이 파일이면 단일 이미지로 처리
    # 입력이 폴더면 이미지 파일들 전부 수집
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path.resolve()]

    if input_path.is_dir():
        if recursive:
            files = [p for p in input_path.rglob("*") if p.is_file()]
        else:
            files = [p for p in input_path.glob("*") if p.is_file()]

        image_files = [p.resolve() for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
        image_files = sorted(image_files)

        if not image_files:
            raise RuntimeError(f"No image files found under: {input_path}")

        return image_files

    raise ValueError(f"Unsupported input path: {input_path}")


# =========================================================
# threshold parsing / loading
# =========================================================

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


# =========================================================
# model helper
# =========================================================

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    num_classes: int,
    device: torch.device,
    channels_last: bool,
) -> torch.nn.Module:
    # checkpoint 로드 -> 모델 구조 생성 -> state_dict 로드 -> device 이동
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    # pretrained=False인 이유:
    # 곧바로 checkpoint 가중치로 덮어쓸 것이기 때문
    model = build_densenet121(
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    model.eval()

    return model


# =========================================================
# single-image prediction
# =========================================================

@torch.no_grad()
def predict_one_image(
    model: torch.nn.Module,
    image_path: Path,
    transform,
    device: torch.device,
    label_names: list[str],
    thresholds: list[float],
    channels_last: bool,
) -> dict[str, Any]:
    # 이미지 1장 열기 -> 전처리 -> 모델 추론 -> threshold 적용
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

    tensor = move_tensor(tensor, device=device, channels_last=channels_last)

    # model 출력은 raw logits [1, C]
    logits = model(tensor)

    # sigmoid 적용해서 확률 [C]로 변환
    probs = logits_to_probs(logits)[0].detach().cpu().tolist()

    predictions: list[dict[str, Any]] = []
    positive_labels: list[str] = []

    # 클래스별로 threshold 적용
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
        "input_path": str(image_path),
        "positive_labels": positive_labels,
        "predictions": predictions,
    }


# =========================================================
# saving helper
# =========================================================

def save_predictions_csv(
    predictions: list[dict[str, Any]],
    output_csv_path: str | Path,
    label_names: list[str],
) -> None:
    # predictions.json과 별도로 csv 저장
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


# =========================================================
# main
# =========================================================

def main() -> None:
    load_dotenv()

    # -----------------------------------------------------
    # 1) CLI 인자
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 2) config / device 로드
    # -----------------------------------------------------
    config = load_config(args.config)
    device = get_device()

    # -----------------------------------------------------
    # 3) 주요 설정 읽기
    # -----------------------------------------------------
    # label_names / num_classes는 data.target_labels 기준
    label_names = list(get_section(config, "data").get("target_labels", list(CHEXPERT_5_LABELS)))
    image_size = int(get_section(config, "data")["image_size"])
    channels_last = get_config_bool(config, "train", "channels_last", default=True)
    num_classes = resolve_num_classes(config)

    # -----------------------------------------------------
    # 4) checkpoint 경로 결정
    # -----------------------------------------------------
    checkpoint_path = resolve_checkpoint_path(
        output_root=get_section(config, "paths")["output_root"],
        checkpoint=args.checkpoint,
    )

    # -----------------------------------------------------
    # 5) checkpoint config와 현재 config 최소 검증
    # -----------------------------------------------------
    checkpoint_meta = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint_meta:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
    validate_checkpoint_config(checkpoint_meta, config)

    # -----------------------------------------------------
    # 6) threshold 결정
    # -----------------------------------------------------
    thresholds, threshold_source = load_thresholds(
        checkpoint_path=checkpoint_path,
        label_names=label_names,
        thresholds_arg=args.thresholds,
    )

    # -----------------------------------------------------
    # 7) 입력 이미지 목록 수집
    # -----------------------------------------------------
    input_paths = collect_input_paths(args.input, recursive=args.recursive)

    # -----------------------------------------------------
    # 8) 전처리 transform 생성
    # -----------------------------------------------------
    transform = build_image_transform(image_size=image_size)

    # -----------------------------------------------------
    # 9) 모델 로드
    # -----------------------------------------------------
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
        channels_last=channels_last,
    )

    # -----------------------------------------------------
    # 10) 출력 폴더 준비
    # -----------------------------------------------------
    run_name = datetime.now().strftime("infer_%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(get_section(config, "paths")["output_root"]) / "infer_runs" / run_name)

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "num_inputs": len(input_paths),
        "recursive": bool(args.recursive),
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
    # 11) 이미지별 추론
    # -----------------------------------------------------
    all_predictions: list[dict[str, Any]] = []
    for image_path in input_paths:
        result = predict_one_image(
            model=model,
            image_path=image_path,
            transform=transform,
            device=device,
            label_names=label_names,
            thresholds=thresholds,
            channels_last=channels_last,
        )
        all_predictions.append(result)

    # -----------------------------------------------------
    # 12) 결과 저장
    # -----------------------------------------------------
    save_json(metadata, output_dir / "infer_metadata.json")
    save_json(all_predictions, output_dir / "predictions.json")
    save_predictions_csv(
        predictions=all_predictions,
        output_csv_path=output_dir / "predictions.csv",
        label_names=label_names,
    )

    # -----------------------------------------------------
    # 13) 콘솔에 일부 결과 미리보기
    # -----------------------------------------------------
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