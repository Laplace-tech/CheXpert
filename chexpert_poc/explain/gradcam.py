from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class GradCAMOutput:
    logits: torch.Tensor
    cam: np.ndarray  # [H, W], float32 in [0, 1]


class GradCAM:
    """
    DenseNet121 같은 CNN backbone에 대해 Grad-CAM을 생성한다.

    핵심:
    - module-level full backward hook를 쓰지 않는다.
    - forward hook에서 output tensor에 직접 register_hook()를 걸어
      gradient를 잡는다.
    - DenseNet의 inplace ReLU와 충돌을 피하기 위한 구현이다.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output

        def _tensor_grad_hook(grad: torch.Tensor) -> None:
            self.gradients = grad

        if output.requires_grad:
            output.register_hook(_tensor_grad_hook)

    def remove_hooks(self) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> GradCAMOutput:
        """
        Args:
            input_tensor: [1, 3, H, W]
            class_idx: 타겟 라벨 인덱스

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
            raise RuntimeError("Tensor hook did not capture gradients")

        # activations / gradients: [1, C, H, W]
        activations = self.activations.detach()
        gradients = self.gradients.detach()

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


def resize_cam_to_image(cam: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """
    cam: [H, W] in [0, 1]
    image_size: (width, height)
    """
    if cam.ndim != 2:
        raise ValueError(f"Expected 2D cam, got shape={cam.shape}")

    cam_img = Image.fromarray((cam * 255.0).round().astype(np.uint8), mode="L")
    cam_img = cam_img.resize(image_size, resample=Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    return cam_resized


def build_heatmap_rgb(cam_resized: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """
    matplotlib colormap 기반 heatmap RGB 생성.
    cmap_name 예:
    - "jet"
    - "turbo"
    - "magma"
    - "inferno"
    """
    if cam_resized.ndim != 2:
        raise ValueError(f"Expected 2D cam_resized, got shape={cam_resized.shape}")

    import matplotlib.cm as cm

    cmap = cm.get_cmap(cmap_name)
    heatmap = cmap(cam_resized)[..., :3]  # RGBA -> RGB
    heatmap = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
    return heatmap

def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    cam_resized: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={image_rgb.shape}")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    image_f = image_rgb.astype(np.float32)
    heatmap_rgb = build_heatmap_rgb(cam_resized, cmap_name="turbo").astype(np.float32)

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