from __future__ import annotations

import torch
import torch.nn.functional as F

# [용어 정리]
# ==================================================
# logits: 
# - 모델이 마지막에 내놓는 원시 점수(raw score)
#
# sigmoid: 
# - logits를 0~1 사이 값으로 바꿔주는 함수
# - 평가/추론 때 "확률처럼 해석"할 수 있게 해줌
#
# BCE(Binary Cross Entropy):
# - 이진분류에서 정답과 예측 확률을 비교해서 손실을 계산
# - 이때 입력으로 확률값을 기대한다. (logits값 X)
#
# BCEWithLogits
# - 입력으로 확률값이 아닌 logits를 받아서 처리하는 BCE
# ==================================================


# loss를 최종적으로 어떻게 줄일지(reduction 방식) 허용 목록
# - "none" : 원소별 loss 그대로 반환 [N, C]
# - "sum"  : 전부 더해서 scalar 반환
# - "mean" : valid한 원소 수로 나눠 평균 scalar 반환
VALID_REDUCTIONS = {"none", "sum", "mean"}


# [입력 텐서 의미]
# - logits:    모델이 마지막에 내놓는 예측 출력값, shape [N, C]
#              N = 배치 크기(이미지 개수), C = 클래스 수
# - targets:   정답 라벨, shape [N, C]
#              각 값은 보통 0 또는 1
# - loss_mask: 이 칸을 학습에 포함할지 말지 정하는 스위치, shape [N, C]
#              1 = loss 계산에 포함
#              0 = loss 계산에서 제외
# - pos_weight: 클래스별 양성 가중치, shape [C] 또는 None
#               양성 샘플이 매우 적은 클래스에서 "양성을 틀렸을 때" loss를 더 크게 부과
def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Masked BCEWithLogits loss: 멀티라벨 분류용 BCEWithLogit loss를 계산

    Args: N(batch size), C(number of Class)
        logits:    shape [N, C]
        targets:   shape [N, C], values expected in {0, 1}
        loss_mask: shape [N, C], values typically in {0, 1}
        pos_weight: optional shape [C]
        reduction:
            - "none": elementwise masked loss [N, C]
            - "sum":  masked loss sum
            - "mean": masked loss sum / number of valid entries
    """
    reduction = _validate_reduction(reduction)

    # [shape 검증]
    # - logits, targets, loss_mask가 전부 같은 shape여야 함
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits.shape != targets.shape: {logits.shape} vs {targets.shape}"
        )
    if logits.shape != loss_mask.shape:
        raise ValueError(
            f"logits.shape != loss_mask.shape: {logits.shape} vs {loss_mask.shape}"
        )

    targets = targets.to(device=logits.device, dtype=logits.dtype)
    loss_mask = loss_mask.to(device=logits.device, dtype=logits.dtype)
    pos_weight = _validate_pos_weight(pos_weight, logits)

    # ********** 원소별 BCEWithLogits 계산 **********
    # - reduction="none"으로 먼저 원소별 loss [N, C]를 얻음 (mask 적용 전)
    # - 모델이 sigmoid를 안 붙인 raw logits를 내기 때문에 BCEWithLogits를 사용,
    #   densenet.py 에서 classfier만 바꿨고 sigmoid는 안 붙였음
    element_loss = F.binary_cross_entropy_with_logits(
        input=logits,
        target=targets,
        pos_weight=pos_weight,
        reduction="none",
    )

    # *************** loss_mask 적용 ***************
    # labels.py의 불확실성 처리 정책이 여기서 적용된다.
    masked_loss = element_loss * loss_mask

    if reduction == "none":
        return masked_loss

    if reduction == "sum":
        return masked_loss.sum()

    valid_count = loss_mask.sum().clamp_min(1.0)
    return masked_loss.sum() / valid_count


# 지원하지 않는 reduction 문자열이면 즉시 실패
def _validate_reduction(reduction: str) -> str:
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"Invalid reduction: {reduction}. "
            f"Expected one of {sorted(VALID_REDUCTIONS)}"
        )
    return reduction


def _validate_pos_weight(
    pos_weight: torch.Tensor | None,
    logits: torch.Tensor,
) -> torch.Tensor | None:
    """
    BCEWithLogitsLoss용 pos_weight shape을 검증/정규화한다.

    기대 형태:
    - None
    - shape [C] where logits.shape == [N, C]

    현재 프로젝트는 멀티라벨 분류 [batch, num_classes]만 다루므로
    1D class-wise pos_weight만 허용한다.
    """
    if pos_weight is None:
        return None

    pos_weight = torch.as_tensor(
        pos_weight,
        device=logits.device,
        dtype=logits.dtype,
    )

    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits to be 2D [N, C] when using pos_weight, "
            f"got logits.shape={tuple(logits.shape)}"
        )

    if pos_weight.ndim != 1:
        raise ValueError(
            f"pos_weight must be 1D with shape [C], got shape={tuple(pos_weight.shape)}"
        )

    num_classes = logits.shape[1]
    if pos_weight.shape[0] != num_classes:
        raise ValueError(
            f"pos_weight length mismatch: len(pos_weight)={pos_weight.shape[0]} "
            f"vs num_classes={num_classes}"
        )

    return pos_weight



@torch.no_grad()
def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Logits를 멀티라벨 확률로 변환한다.
    
    - model은 sigmoid 미적용된 raw logits를 출력한다.
    - eval/infer 시에는 획률이 필요하므로 sigmoid가 적용됨.
    - 다중 분류가 아니라 이진 멀티라벨 분류이므로 각 클래스가 독립적으로 0~1 확률을 가짐.
    """
    return torch.sigmoid(logits)