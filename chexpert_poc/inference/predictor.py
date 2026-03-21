# 모델 로드 / 단일 추론

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from chexpert_poc.inference.postprocess import build_prediction_result
from chexpert_poc.models.densenet import build_densenet121
from chexpert_poc.common.runtime import move_tensor

def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    num_classes: int,
    device: torch.device,
    channels_last: bool,
) -> torch.nn.Module:
    # checkpoint dict -> 모델 구조 생성 -> state_dict 로드 -> device 이동
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing model_state_dict")

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
    # 이미지 1장 열기 -> 전처리 -> 모델 추론 -> threshold 적용
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

    tensor = move_tensor(tensor, device=device, channels_last=channels_last)

    # model 출력은 raw logits [1, C]
    logits = model(tensor)
    
    """
    Logits를 멀티라벨 확률로 변환한다.
    - model은 sigmoid 미적용된 raw logits를 출력한다.
    - eval/infer 시에는 획률이 필요하므로 sigmoid가 적용됨.
    - 다중 분류가 아니라 이진 멀티라벨 분류이므로 각 클래스가 독립적으로 0~1 확률을 가짐.
    """
    # sigmoid 적용해서 확률 [C]로 변환
    probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

    return build_prediction_result(
        input_path=str(image_path),
        label_names=label_names,
        probs=probs,
        thresholds=thresholds,
    )