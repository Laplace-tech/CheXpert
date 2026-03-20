# - 외부에서 호출하는 함수:
# (function) def encode_chexpert_label(value: object, strategy: str) -> tuple[float, float]
# (function) def validate_uncertainty_strategy(strategy: str) -> str
# (function) def is_frontal_view(path_value: object) -> bool

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

# 현재 허용하는 uncertain(-1) 처리 정책
VALID_UNCERTAINTY_STRATEGIES: Final[set[str]] = {"U-Ignore", "U-Ones"}

# Path 문자열만 보고 frontal view 여부를 판별할 때 쓰는 키워드
FRONTAL_VIEW_KEYWORDS: Final[tuple[str, ...]] = (
    "view1_frontal",
    "view2_frontal",
    "frontal",
)

# CheXpert raw label의 값을 *학습용 (label_value, loss_mask_value)*로 바꿈
def encode_chexpert_label(value: object, strategy: str) -> tuple[float, float]:
    
    strategy = validate_uncertainty_strategy(strategy) # strategy 값이 유효한지 검증
    normalized = _normalize_label_value(value)         # raw 값을 float/NaN 기준으로 통일

    # 1) NaN:
    if math.isnan(normalized):
        return 0.0, 1.0  # NaN은 음성으로 두고 loss는 포함

    # 2) Positive:
    if normalized == 1.0:
        return 1.0, 1.0 # 양성으로 두고 loss는 포함

    # 3) Negative:
    if normalized == 0.0:
        return 0.0, 1.0 # 음성으로 두고 loss는 포함

    # 4) Uncertain (-1):
    if normalized == -1.0:
        
        if strategy == "U-Ignore":
            # (label_value = 0, loss_mask = 0)
            #  uncertain을 0으로 놓되, mask=0으로 loss(학습 손실)에서 제외시킴
            # - 라벨 품질이 애매한 데이터들을 억지로 학습하지 않는다
            # - uncertain 샘플이 많으면 데이터 낭비가 커질 수 있음. 
            return 0.0, 0.0 
        
        if strategy == "U-Ones":
            # (label_value = 1, loss_mask = 1) 
            #  uncertain을 양성으로 판정한다.
            # - uncertain도 학습에 활용 가능함
            # - 양성이 아닌 샘플까지 양성처럼 주입할 수 있어서 노이즈가 증가 (precision 박살남)
            return 1.0, 1.0  
        
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


def validate_uncertainty_strategy(strategy: str) -> str:
    # 설정에 들어온 uncertainty_strategy 값이 유효한지 검사
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

