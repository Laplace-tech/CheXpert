from __future__ import annotations

import torch.nn as nn # classifier를 nn.Linear(...)로 교체할 때 사용
from torchvision.models import DenseNet121_Weights, densenet121


# [DenseNet121 멀티라벨 분류 모델을 생성]
#
#  1. num_classes(최종 출력 클래스 수):
#  2. ImageNet pretrained weights 사용 여부:
def build_densenet121(
    num_classes: int = 5,    # num_classes=int(config["model"]["num_classes"])
    pretrained: bool = True, # pretrained=bool(config["model"]["pretrained"])
) -> nn.Module:
    """
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

    # pretrained 여부에 따라 weights 결정
    weights = _resolve_densenet121_weights(pretrained=pretrained)
    
    # torchvision DenseNet121 생성
    # - feature extractor + classifier가 포함된 기본 DenseNet121
    # - classifier는 원래 ImageNet 1000-class 분류용 마지막 층일 것이므로 그대로 가져다 못씀.
    model = densenet121(weights=weights)

    if not hasattr(model, "classifier"):
        raise AttributeError("DenseNet121 model does not have classifier attribute")

    # [classifier(마지막 분류층) 교체]
    # 1) 원래 ImageNet 1000-class classifier를 버리고, 현재 프로젝트용 num_classes 출력 선형층으로 교체
    # 2) 여기서 softmax/sigmoid는 안 붙이고, "raw logits"(sigmoid 적용 전 원시 점수)만 출력
    # 3) 나중에:
    #   * train.py / losses.py 에서는 BCEWithLogits 계열 손실에 바로 넣고
    #   * eval.py / infer.py 에서는 sigmoid로 확률 변환
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)


    # 최종 모델 반환
    # - train.py 에서 optimizer 붙이고 학습
    # - eval.py / infer.py 에서 checkpoint load 후 사용
    return model



# 가중치 끌어오기: DenseNet121_Weights.IMAGENET1K_V1
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
