from __future__ import annotations

import torch
import torch.nn.functional as F


VALID_REDUCTIONS = {"none", "sum", "mean"}


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    멀티라벨 분류용 Masked BCEWithLogits loss.

    Args:
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

    element_loss = F.binary_cross_entropy_with_logits(
        input=logits,
        target=targets,
        pos_weight=pos_weight,
        reduction="none",
    )

    # label_policy에서 정의한 uncertainty 처리 결과(loss_mask)가 여기서 반영된다.
    masked_loss = element_loss * loss_mask

    if reduction == "none":
        return masked_loss

    if reduction == "sum":
        return masked_loss.sum()

    valid_count = loss_mask.sum().clamp_min(1.0)
    return masked_loss.sum() / valid_count


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