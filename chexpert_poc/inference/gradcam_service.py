from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from chexpert_poc.explain.gradcam import (
    GradCAM,
    build_heatmap_rgb,
    make_triptych,
    overlay_heatmap_on_image,
    resize_cam_to_image,
    save_rgb_image,
)
from chexpert_poc.inference.postprocess import build_prediction_result


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
    label_arg: str | None,
) -> tuple[str, int]:
    if label_arg is not None:
        if label_arg not in label_names:
            raise ValueError(
                f"Unknown label: {label_arg}. Expected one of {label_names}"
            )
        return label_arg, label_names.index(label_arg)

    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return label_names[best_idx], best_idx


def load_original_rgb(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        return np.array(rgb, dtype=np.uint8)


def build_input_tensor(
    image_rgb: np.ndarray,
    transform,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    image_for_model = Image.fromarray(image_rgb)
    tensor = transform(image_for_model).unsqueeze(0)

    if channels_last and tensor.ndim == 4:
        tensor = tensor.to(
            device=device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    else:
        tensor = tensor.to(device=device, non_blocking=True)

    return tensor


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> list[float]:
    
    logits = model(tensor)
    
    """
    Logits를 멀티라벨 확률로 변환한다.
    - model은 sigmoid 미적용된 raw logits를 출력한다.
    - eval/infer 시에는 획률이 필요하므로 sigmoid가 적용됨.
    - 다중 분류가 아니라 이진 멀티라벨 분류이므로 각 클래스가 독립적으로 0~1 확률을 가짐.
    """
    return torch.sigmoid(logits)[0].detach().cpu().tolist()


def generate_gradcam_result(
    *,
    model: torch.nn.Module,
    image_path: str | Path,
    transform,
    device: torch.device,
    label_names: list[str],
    thresholds: list[float],
    channels_last: bool,
    target_label_arg: str | None = None,
    alpha: float = 0.75,
) -> dict[str, Any]:
    image_path = validate_input_image(image_path)
    original_rgb = load_original_rgb(image_path)
    tensor = build_input_tensor(
        image_rgb=original_rgb,
        transform=transform,
        device=device,
        channels_last=channels_last,
    )

    probs_before_cam = predict_probs(model, tensor)
    target_label, target_idx = choose_target_label(
        label_names=label_names,
        probs=probs_before_cam,
        label_arg=target_label_arg,
    )

    gradcam = GradCAM(model=model, target_layer=model.features)
    try:
        cam_out = gradcam.generate(tensor, class_idx=target_idx)
        logits = cam_out.logits
        cam = cam_out.cam
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()
    finally:
        gradcam.remove_hooks()

    prediction_result = build_prediction_result(
        input_path=str(image_path),
        label_names=label_names,
        probs=probs,
        thresholds=thresholds,
    )

    cam_resized = resize_cam_to_image(
        cam,
        image_size=(original_rgb.shape[1], original_rgb.shape[0]),
    )
    heatmap_rgb = build_heatmap_rgb(cam_resized, cmap_name="turbo")
    overlay_rgb = overlay_heatmap_on_image(
        image_rgb=original_rgb,
        cam_resized=cam_resized,
        alpha=float(alpha),
    )
    panel = make_triptych(original_rgb, heatmap_rgb, overlay_rgb)

    return {
        "input_path": str(image_path),
        "target_label": target_label,
        "target_idx": target_idx,
        "target_prob": float(probs[target_idx]),
        "positive_labels": prediction_result["positive_labels"],
        "predictions": prediction_result["predictions"],
        "images": {
            "original_rgb": original_rgb,
            "heatmap_rgb": heatmap_rgb,
            "overlay_rgb": overlay_rgb,
            "panel": panel,
        },
    }


def save_gradcam_artifacts(
    *,
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_label = result["target_label"]
    safe_label = target_label.replace(" ", "_")

    original_path = output_dir / "original.png"
    heatmap_path = output_dir / f"heatmap_{safe_label}.png"
    overlay_path = output_dir / f"overlay_{safe_label}.png"
    panel_path = output_dir / f"panel_{safe_label}.png"

    save_rgb_image(result["images"]["original_rgb"], original_path)
    save_rgb_image(result["images"]["heatmap_rgb"], heatmap_path)
    save_rgb_image(result["images"]["overlay_rgb"], overlay_path)
    result["images"]["panel"].save(panel_path)

    return {
        "original_image": str(original_path),
        "heatmap_image": str(heatmap_path),
        "overlay_image": str(overlay_path),
        "panel_image": str(panel_path),
    }