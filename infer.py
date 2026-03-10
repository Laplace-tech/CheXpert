from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from PIL import Image

from chexpert_poc.datasets.chexpert_dataset import build_image_transform
from chexpert_poc.datasets.labels import CHEXPERT_5_LABELS
from chexpert_poc.models.densenet import build_densenet121
from chexpert_poc.utils.losses import logits_to_probs
from chexpert_poc.utils.train_utils import ensure_dir, get_device, load_config, save_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def move_tensor(
    x: torch.Tensor,
    device: torch.device,
    channels_last: bool = False,
) -> torch.Tensor:
    if channels_last and x.ndim == 4:
        return x.to(
            device=device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return x.to(device=device, non_blocking=True)


def find_latest_best_checkpoint(output_root: str | Path) -> Path:
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


def collect_input_paths(input_path: str | Path, recursive: bool = False) -> list[Path]:
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


def parse_thresholds_from_string(
    thresholds_str: str,
    label_names: list[str],
) -> list[float]:
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

        json_labels = data.get("labels", None)
        if json_labels is not None and list(json_labels) != list(label_names):
            raise ValueError(
                f"Threshold label order mismatch: {threshold_json_path}"
            )

        return thresholds, str(threshold_json_path)

    return [0.5] * len(label_names), "default_0.5"


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    num_classes: int,
    device: torch.device,
    channels_last: bool,
) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

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
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0)

    tensor = move_tensor(tensor, device=device, channels_last=channels_last)

    logits = model(tensor)
    probs = logits_to_probs(logits)[0].detach().cpu().tolist()

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
        "input_path": str(image_path),
        "positive_labels": positive_labels,
        "predictions": predictions,
    }


def save_predictions_csv(
    predictions: list[dict[str, Any]],
    output_csv_path: str | Path,
    label_names: list[str],
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


def main() -> None:
    load_dotenv()

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
    device = get_device()

    label_names = list(config["data"].get("target_labels", list(CHEXPERT_5_LABELS)))
    image_size = int(config["data"]["image_size"])
    channels_last = bool(config["train"].get("channels_last", True))

    checkpoint_path = resolve_checkpoint_path(
        output_root=config["paths"]["output_root"],
        checkpoint=args.checkpoint,
    )

    thresholds, threshold_source = load_thresholds(
        checkpoint_path=checkpoint_path,
        label_names=label_names,
        thresholds_arg=args.thresholds,
    )

    input_paths = collect_input_paths(args.input, recursive=args.recursive)
    transform = build_image_transform(image_size=image_size)

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=int(config["model"]["num_classes"]),
        device=device,
        channels_last=channels_last,
    )

    run_name = datetime.now().strftime("infer_%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(config["paths"]["output_root"]) / "infer_runs" / run_name)

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "num_inputs": len(input_paths),
        "recursive": bool(args.recursive),
        "labels": label_names,
        "device": str(device),
        "channels_last": channels_last,
    }

    print("=" * 100)
    print("infer.py start")
    print("=" * 100)
    print(f"device          : {device}")
    print(f"checkpoint_path : {checkpoint_path}")
    print(f"threshold_source: {threshold_source}")
    print(f"num_inputs      : {len(input_paths)}")
    print(f"output_dir      : {output_dir}")
    print(f"thresholds      : {thresholds}")

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

    save_json(metadata, output_dir / "infer_metadata.json")
    save_json(all_predictions, output_dir / "predictions.json")
    save_predictions_csv(
        predictions=all_predictions,
        output_csv_path=output_dir / "predictions.csv",
        label_names=label_names,
    )

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