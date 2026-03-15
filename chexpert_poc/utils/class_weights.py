from __future__ import annotations

from typing import Any, Sequence

import pandas as pd  # DataFrame 다루기
import torch  # 최종 pos_weight를 텐서로 변환

# 라벨 정책 가져다 쓰기
from chexpert_poc.datasets.labels import (
    CHEXPERT_5_LABELS,
    encode_chexpert_label,
    validate_uncertainty_strategy,
)


# target_labels가 비어 있지 않은지 검사
# 호출부:
# - compute_pos_weight_from_dataframe(...)
# - compute_pos_weight_from_dataset(...)
def _validate_target_labels(
    target_labels: Sequence[str],
) -> list[str]:
    labels = list(target_labels)
    if not labels:
        raise ValueError("target_labels must not be empty")
    return labels


# pos_weight 상한 잘라버리기
# pos_weight = neg / pos 값이 과하게 커질 수 있음.
#  그 값을 그대로 학습에 반영하면 loss가 크게 휘청일 수 있으니 상한을 두어 제한함.
def _validate_clip_max(clip_max: float | None) -> float | None:
    if clip_max is None:
        return None

    clip_max = float(clip_max)
    if clip_max <= 0.0:
        raise ValueError(f"clip_max must be > 0, got {clip_max}")
    return clip_max


# 원본 DataFrame을 받아서 라벨별 pos_weight를 계산
#
# [입력]
# - df: 보통 train.csv를 읽은 DataFrame 또는 dataset.df
# - target_labels: 현재 프로젝트의 타겟 라벨들
# - uncertainty_strategy: U-Ignore / U-Ones
# - clip_max: pos_weight 상한
#
# [출력]
# - pos_weight_tensor: shape [C]
# - stats: 각 클래스별 pos/neg/ignored 통계
def compute_pos_weight_from_dataframe(
    df: pd.DataFrame,
    target_labels: list[str] | tuple[str, ...] = CHEXPERT_5_LABELS,
    uncertainty_strategy: str = "U-Ignore",
    clip_max: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """
    Raw dataframe 기준으로 라벨별 pos_weight를 계산한다.

    계산 규칙:
    - uncertainty_strategy를 적용해 (label_value, loss_mask_value)로 변환
    - loss_mask == 0 인 항목은 ignored_count로만 집계
    - valid 항목에 대해 pos_weight = negative_count / max(positive_count, 1)

    Returns:
        pos_weight_tensor: shape [C]
        stats: class별 집계 정보
    """
    strategy = validate_uncertainty_strategy(uncertainty_strategy)
    labels = _validate_target_labels(target_labels)
    clip_max = _validate_clip_max(clip_max)

    stats: list[dict[str, Any]] = []
    pos_weights: list[float] = []

    # 라벨별 반복:
    # 각 클래스마다 positive_count / negative_count / ignored_count 를 센다.
    for label_name in labels:
        if label_name not in df.columns:
            raise ValueError(f"Missing target label column: {label_name}")

        positive_count = 0
        negative_count = 0
        ignored_count = 0

        for raw_value in df[label_name].tolist():
            # 각각의 raw value를 라벨 정책에 따라서 해석함 (label.py)
            label_value, loss_mask_value = encode_chexpert_label(raw_value, strategy)

            # loss_mask = 0이면 해당 샘플은 학습에서 제외시킨다
            if loss_mask_value == 0.0:
                ignored_count += 1
                continue

            if label_value == 1.0:
                positive_count += 1
            elif label_value == 0.0:
                negative_count += 1
            else:
                raise ValueError(
                    f"Unexpected encoded label value: {label_value} for {label_name}"
                )

        # pos_weight 계산
        safe_positive_count = max(positive_count, 1)
        pos_weight = float(negative_count / safe_positive_count)

        # pos_weight 상한선 제한
        if clip_max is not None:
            pos_weight = min(pos_weight, clip_max)

        # 최종 결과 누적
        pos_weights.append(pos_weight)
        stats.append(
            {
                "label": label_name,
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "ignored_count": int(ignored_count),
                "pos_weight": float(pos_weight),
            }
        )

    # 최종적으로 [C] 텐서로 상한값 반환
    # ex: tensor([2.31, 5.82, 7.44, 1.90, 3.12])
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32)
    return pos_weight_tensor, stats


# [역할]
# - dataset 안에 이미 계산된 label_matrix / loss_mask_matrix가 있으면
#   그걸 직접 써서 빠르고 일관되게 pos_weight 계산
# - 없으면 fallback으로 dataset.df를 써서 다시 계산
def compute_pos_weight_from_dataset(
    dataset,    # 보통 chexpert_dataset.py 의 CheXpertDataset 인스턴스
    clip_max: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """
    Dataset 기준으로 라벨별 pos_weight를 계산한다.

    우선순위:
    1) dataset.label_matrix / dataset.loss_mask_matrix가 있으면 그걸 사용
    2) 없으면 dataset.df를 사용해 다시 계산

    기대 속성:
    - dataset.target_labels
    - dataset.uncertainty_strategy
    - (optional) dataset.label_matrix, dataset.loss_mask_matrix
    - (fallback) dataset.df
    """
    clip_max = _validate_clip_max(clip_max)

    if not hasattr(dataset, "target_labels"):
        raise AttributeError("dataset must have target_labels")
    if not hasattr(dataset, "uncertainty_strategy"):
        raise AttributeError("dataset must have uncertainty_strategy")

    target_labels = _validate_target_labels(dataset.target_labels)

    # chexpert_dataset.py 는 __init__에서
    #  self.label_matrix, self.loss_mask_matrix 를 이미 만들어둠
    #  따라서, raw df를 다시 encode할 필요 없이 바로 그 텐서를 기준으로 pos_weight를 계산
    has_precomputed = hasattr(dataset, "label_matrix") and hasattr(
        dataset, "loss_mask_matrix"
    )

    if has_precomputed:
        labels = torch.as_tensor(dataset.label_matrix, dtype=torch.float32)
        loss_masks = torch.as_tensor(dataset.loss_mask_matrix, dtype=torch.float32)

        # shape 검증
        # labels / loss_masks 모두 [N, C] 이어야 함
        if labels.ndim != 2 or loss_masks.ndim != 2:
            raise ValueError(
                f"Expected 2D label/mask tensors, got labels={labels.shape}, "
                f"loss_masks={loss_masks.shape}"
            )
        if labels.shape != loss_masks.shape:
            raise ValueError(
                f"labels.shape != loss_masks.shape: {labels.shape} vs {loss_masks.shape}"
            )
        if labels.shape[1] != len(target_labels):
            raise ValueError(
                f"Number of dataset classes ({labels.shape[1]}) does not match "
                f"len(target_labels) ({len(target_labels)})"
            )

        stats: list[dict[str, Any]] = []
        pos_weights: list[float] = []

        for class_idx, label_name in enumerate(target_labels):
            valid = loss_masks[:, class_idx] > 0.5
            y = labels[:, class_idx]

            positive_count = int(((y == 1.0) & valid).sum().item())
            negative_count = int(((y == 0.0) & valid).sum().item())
            ignored_count = int((~valid).sum().item())

            safe_positive_count = max(positive_count, 1)
            pos_weight = float(negative_count / safe_positive_count)

            if clip_max is not None:
                pos_weight = min(pos_weight, clip_max)

            pos_weights.append(pos_weight)
            stats.append(
                {
                    "label": label_name,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "ignored_count": ignored_count,
                    "pos_weight": float(pos_weight),
                }
            )

        return torch.tensor(pos_weights, dtype=torch.float32), stats

    if not hasattr(dataset, "df"):
        raise AttributeError(
            "dataset must have either (label_matrix, loss_mask_matrix) or df"
        )

    return compute_pos_weight_from_dataframe(
        df=dataset.df,
        target_labels=target_labels,
        uncertainty_strategy=dataset.uncertainty_strategy,
        clip_max=clip_max,
    )


# - train.py 로그 출력용 / 콘솔 확인용
def format_pos_weight_stats(stats: list[dict[str, Any]]) -> str:
    """
    pos_weight 통계를 사람이 읽기 쉬운 표 문자열로 만든다.
    """
    lines = []
    lines.append(
        f"{'label':17s} | {'pos':>8s} | {'neg':>8s} | {'ignored':>8s} | {'pos_weight':>10s}"
    )
    lines.append("-" * 68)

    for item in stats:
        lines.append(
            f"{item['label']:17s} | "
            f"{item['positive_count']:8d} | "
            f"{item['negative_count']:8d} | "
            f"{item['ignored_count']:8d} | "
            f"{item['pos_weight']:10.4f}"
        )

    return "\n".join(lines)
