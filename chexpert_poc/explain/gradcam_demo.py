from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from PIL import Image

from chexpert_poc.datasets.chexpert_dataset import build_image_transform
from chexpert_poc.datasets.labels import CHEXPERT_5_LABELS
from chexpert_poc.explain.gradcam import (
    GradCAM,
    build_heatmap_rgb,
    make_triptych,
    overlay_heatmap_on_image,
    resize_cam_to_image,
    save_rgb_image,
)
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
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent.parent
    candidate = run_dir / "eval" / "threshold_tuning" / "infer_thresholds.json"
    if candidate.exists():
        return candidate
    return None


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


def validate_input_image(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input must be a file: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    return path.resolve()


def choose_target_label(
    label_names: list[str],
    probs: list[float],
    thresholds: list[float],
    label_arg: str | None,
) -> tuple[str, int]:
    if label_arg is not None:
        if label_arg not in label_names:
            raise ValueError(
                f"Unknown label: {label_arg}. Expected one of {label_names}"
            )
        return label_arg, label_names.index(label_arg)

    # 기본은 가장 확률이 높은 라벨
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return label_names[best_idx], best_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="single image path")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="explicit checkpoint path; if omitted, latest best.pt is used",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Grad-CAM을 만들 라벨 이름. 없으면 최고 확률 라벨 사용",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="comma-separated thresholds, e.g. 0.47,0.22,0.36,0.39,0.60",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="overlay 강도 (0~1)",
    )
    args = parser.parse_args()

    load_dotenv()

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

    image_path = validate_input_image(args.input)
    transform = build_image_transform(image_size=image_size)

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=int(config["model"]["num_classes"]),
        device=device,
        channels_last=channels_last,
    )

    original_pil = Image.open(image_path).convert("RGB")
    original_rgb = np_image = None  # lint appeasement
    try:
        original_rgb = __import__("numpy").array(original_pil, dtype=__import__("numpy").uint8)
    finally:
        original_pil.close()

    image_for_model = Image.fromarray(original_rgb)
    tensor = transform(image_for_model).unsqueeze(0)
    tensor = move_tensor(tensor, device=device, channels_last=channels_last)

    gradcam = GradCAM(model=model, target_layer=model.features)
    try:
        # 먼저 한 번 forward/backward 해서 CAM과 logits를 같이 얻는다.
        # label 미지정이면 logits로 최고 확률 라벨을 고른 후 다시 backward 1회 수행.
        initial_out = gradcam.generate(tensor, class_idx=0)
        probs = logits_to_probs(initial_out.logits)[0].detach().cpu().tolist()

        target_label, target_idx = choose_target_label(
            label_names=label_names,
            probs=probs,
            thresholds=thresholds,
            label_arg=args.label,
        )

        # label이 class 0이 아닌 경우 실제 target label로 CAM 재계산
        if target_idx != 0:
            cam_out = gradcam.generate(tensor, class_idx=target_idx)
            logits = cam_out.logits
            cam = cam_out.cam
            probs = logits_to_probs(logits)[0].detach().cpu().tolist()
        else:
            logits = initial_out.logits
            cam = initial_out.cam

    finally:
        gradcam.remove_hooks()

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

    cam_resized = resize_cam_to_image(cam, image_size=(original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_rgb = build_heatmap_rgb(cam_resized)
    overlay_rgb = overlay_heatmap_on_image(
        image_rgb=original_rgb,
        cam_resized=cam_resized,
        alpha=float(args.alpha),
    )
    panel = make_triptych(original_rgb, heatmap_rgb, overlay_rgb)

    run_name = datetime.now().strftime("gradcam_%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(config["paths"]["output_root"]) / "gradcam_runs" / run_name)

    result = {
        "input_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "threshold_source": threshold_source,
        "positive_labels": positive_labels,
        "target_label": target_label,
        "target_prob": float(probs[target_idx]),
        "predictions": predictions,
        "artifacts": {
            "original_image": str(output_dir / "original.png"),
            "heatmap_image": str(output_dir / f"heatmap_{target_label.replace(' ', '_')}.png"),
            "overlay_image": str(output_dir / f"overlay_{target_label.replace(' ', '_')}.png"),
            "panel_image": str(output_dir / f"panel_{target_label.replace(' ', '_')}.png"),
        },
    }

    save_json(result, output_dir / "gradcam_result.json")
    save_rgb_image(original_rgb, output_dir / "original.png")
    save_rgb_image(
        heatmap_rgb,
        output_dir / f"heatmap_{target_label.replace(' ', '_')}.png",
    )
    save_rgb_image(
        overlay_rgb,
        output_dir / f"overlay_{target_label.replace(' ', '_')}.png",
    )
    panel.save(output_dir / f"panel_{target_label.replace(' ', '_')}.png")

    print("=" * 100)
    print("gradcam_demo.py start")
    print("=" * 100)
    print(f"device          : {device}")
    print(f"checkpoint_path : {checkpoint_path}")
    print(f"threshold_source: {threshold_source}")
    print(f"input_path      : {image_path}")
    print(f"target_label    : {target_label}")
    print(f"target_prob     : {probs[target_idx]:.4f}")
    print(f"positive_labels : {positive_labels}")

    print("\n[predictions]")
    for pred in predictions:
        print(
            f"- {pred['label']}: "
            f"prob={pred['prob']:.4f}, "
            f"threshold={pred['threshold']:.2f}, "
            f"pred={pred['pred']}"
        )

    print("\nartifacts saved to:")
    print(output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()