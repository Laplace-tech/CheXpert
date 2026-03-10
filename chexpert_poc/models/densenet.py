from __future__ import annotations

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


def _resolve_densenet121_weights(pretrained: bool):
    """
    torchvision DenseNet121 가중치 선택.

    현재 프로젝트 baseline은
    - pretrained=True  -> ImageNet pretrained 사용
    - pretrained=False -> random init
    """
    if pretrained:
        return DenseNet121_Weights.IMAGENET1K_V1
    return None


def build_densenet121(
    num_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    """
    DenseNet121 멀티라벨 분류 모델을 생성한다.

    동작:
    - torchvision DenseNet121 백본 생성
    - 마지막 classifier를 num_classes 출력으로 교체

    Args:
        num_classes: 최종 출력 클래스 수
        pretrained: ImageNet pretrained weights 사용 여부

    Returns:
        nn.Module
    """
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    weights = _resolve_densenet121_weights(pretrained=pretrained)
    model = densenet121(weights=weights)

    if not hasattr(model, "classifier"):
        raise AttributeError("DenseNet121 model does not have classifier attribute")

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model