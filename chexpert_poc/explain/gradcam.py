from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


@dataclass
class GradCAMOutput:
    logits: torch.Tensor
    cam: np.ndarray  # shape [H, W], float32 in [0, 1]


class GradCAM:
    """
    DenseNet121 같은 CNN backbone에 대해 Grad-CAM을 생성한다.

    사용 방식:
    1) target_layer에 hook 등록
    2) 특정 class_idx logit에 대해 backward
    3) gradient global average pooling -> channel weight
    4) weighted sum + ReLU -> CAM
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )
        self._backward_handle = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if not grad_output or grad_output[0] is None:
            self.gradients = None
            return
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> GradCAMOutput:
        """
        Args:
            input_tensor: shape [1, 3, H, W]
            class_idx: Grad-CAM을 만들 라벨 index

        Returns:
            GradCAMOutput(logits, cam)
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected input_tensor shape [1, 3, H, W], got {tuple(input_tensor.shape)}"
            )

        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logits = self.model(input_tensor)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"Expected logits shape [1, C], got {tuple(logits.shape)}")

        if not (0 <= class_idx < logits.shape[1]):
            raise ValueError(
                f"class_idx out of range: {class_idx}, num_classes={logits.shape[1]}"
            )

        target_logit = logits[0, class_idx]
        target_logit.backward(retain_graph=False)

        if self.activations is None:
            raise RuntimeError("Forward hook did not capture activations")
        if self.gradients is None:
            raise RuntimeError("Backward hook did not capture gradients")

        # activations / gradients shape: [1, C, H, W]
        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * activations).sum(dim=1)[0]  # [H, W]
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy().astype(np.float32)
        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return GradCAMOutput(logits=logits.detach(), cam=cam)


def denormalize_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    """
    ImageNet normalize 된 tensor [3, H, W] -> uint8 RGB image [H, W, 3]
    """
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(
            f"Expected image_tensor shape [3, H, W], got {tuple(image_tensor.shape)}"
        )

    mean = IMAGENET_MEAN.to(device=image_tensor.device)[:, None, None]
    std = IMAGENET_STD.to(device=image_tensor.device)[:, None, None]

    x = image_tensor.detach().float() * std + mean
    x = x.clamp(0.0, 1.0)
    x = x.permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return x


def resize_cam_to_image(cam: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """
    cam: [H, W] in [0,1]
    image_size: (width, height)
    """
    if cam.ndim != 2:
        raise ValueError(f"Expected 2D cam, got shape={cam.shape}")

    cam_img = Image.fromarray((cam * 255.0).round().astype(np.uint8), mode="L")
    cam_img = cam_img.resize(image_size, resample=Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    return cam_resized


def build_heatmap_rgb(cam_resized: np.ndarray) -> np.ndarray:
    """
    간단한 red-yellow 계열 heatmap RGB 생성.
    """
    if cam_resized.ndim != 2:
        raise ValueError(
            f"Expected 2D cam_resized, got shape={cam_resized.shape}"
        )

    red = cam_resized
    green = cam_resized * 0.45
    blue = np.zeros_like(cam_resized, dtype=np.float32)

    heatmap = np.stack([red, green, blue], axis=-1)
    heatmap = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
    return heatmap


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    cam_resized: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    원본 RGB 이미지 위에 CAM heatmap을 overlay 한다.
    alpha는 전체 강도이고, 실제 mixing은 cam intensity에 비례한다.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={image_rgb.shape}")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    image_f = image_rgb.astype(np.float32)
    heatmap_rgb = build_heatmap_rgb(cam_resized).astype(np.float32)

    alpha_map = (cam_resized[..., None] * alpha).astype(np.float32)
    overlay = image_f * (1.0 - alpha_map) + heatmap_rgb * alpha_map
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def make_triptych(
    original_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    gap: int = 12,
) -> Image.Image:
    """
    original / heatmap / overlay 를 한 장으로 붙인다.
    """
    original = Image.fromarray(original_rgb)
    heatmap = Image.fromarray(heatmap_rgb)
    overlay = Image.fromarray(overlay_rgb)

    width, height = original.size
    canvas = Image.new("RGB", (width * 3 + gap * 2, height), color=(255, 255, 255))

    canvas.paste(original, (0, 0))
    canvas.paste(heatmap, (width + gap, 0))
    canvas.paste(overlay, (width * 2 + gap * 2, 0))
    return canvas


def save_rgb_image(image_rgb: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)