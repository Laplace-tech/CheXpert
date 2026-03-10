from __future__ import annotations

import math
from typing import Final

# 이번 프로젝트에서 사용하는 5개 타겟 라벨
CHEXPERT_5_LABELS: Final[list[str]] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# 현재 허용하는 uncertain 처리 정책
VALID_UNCERTAINTY_STRATEGIES: Final[set[str]] = {"U-Ignore", "U-Ones"}

# Path 문자열만 보고 frontal view 여부를 판별할 때 쓰는 키워드
FRONTAL_VIEW_KEYWORDS: Final[tuple[str, ...]] = (
    "view1_frontal",
    "view2_frontal",
    "frontal",
)


def validate_uncertainty_strategy(strategy: str) -> str:
    # 설정값으로 들어온 uncertain 정책이 허용 범위 안인지 검증
    if strategy not in VALID_UNCERTAINTY_STRATEGIES:
        raise ValueError(
            f"Invalid uncertainty strategy: {strategy}. "
            f"Expected one of {sorted(VALID_UNCERTAINTY_STRATEGIES)}"
        )
    return strategy


def _is_nan_like(value: object) -> bool:
    # NaN, np.nan, float("nan") 같은 값을 안전하게 판별
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _normalize_label_value(value: object) -> float:
    # 라벨 값을 float 기준으로 통일
    # NaN 계열은 float("nan"), 숫자는 float로 변환
    if _is_nan_like(value):
        return float("nan")

    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Label value is not numeric: {value}") from e


# CheXpert raw label의 값을 *학습용 (label_value, loss_mask_value)*로 바꿈
def encode_chexpert_label(value: object, strategy: str) -> tuple[float, float]:
    """
    NaN  -> (0.0, 1.0)
    0    -> (0.0, 1.0)
    1    -> (1.0, 1.0)
    -1   -> strategy에 따라 처리 
    """
    strategy = validate_uncertainty_strategy(strategy)
    normalized = _normalize_label_value(value)

    if math.isnan(normalized):
        return 0.0, 1.0  # NaN은 음성으로 두고 loss는 포함

    if normalized == 1.0:
        return 1.0, 1.0  # 양성

    if normalized == 0.0:
        return 0.0, 1.0  # 음성

    if normalized == -1.0:
        if strategy == "U-Ignore":
            return 0.0, 0.0  # uncertain은 학습에서 제외
        if strategy == "U-Ones":
            return 1.0, 1.0  # uncertain을 양성으로 취급

    raise ValueError(f"Unexpected label value: {value}")


def is_frontal_view(path_value: object) -> bool:
    # 이미지 path 문자열을 보고 frontal view인지 판별
    # 별도 view 메타데이터가 아니라 파일명 규칙에 의존
    if not isinstance(path_value, str):
        return False

    path_lower = path_value.strip().replace("\\", "/").lower()
    if not path_lower:
        return False

    return any(keyword in path_lower for keyword in FRONTAL_VIEW_KEYWORDS)