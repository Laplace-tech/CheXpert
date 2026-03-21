from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Final

CHEXPERT_5_LABELS: Final[tuple[str, ...]] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)

VALID_UNCERTAINTY_STRATEGIES: Final[frozenset[str]] = frozenset(
    {"U-Ignore", "U-Ones"}
)


def encode_chexpert_label(value: object, strategy: str) -> tuple[float, float]:
    strategy = validate_uncertainty_strategy(strategy)
    normalized = _normalize_label_value(value)

    # 현재 프로젝트 정책:
    # NaN/blank는 negative(0)로 두고 loss 계산에는 포함한다.
    if math.isnan(normalized):
        return 0.0, 1.0

    if normalized == 1.0:
        return 1.0, 1.0

    if normalized == 0.0:
        return 0.0, 1.0

    if normalized == -1.0:
        if strategy == "U-Ignore":
            return 0.0, 0.0
        if strategy == "U-Ones":
            return 1.0, 1.0

    raise ValueError(f"Unexpected label value: {value!r}")


def is_frontal_view(path_value: object) -> bool:
    if not isinstance(path_value, str):
        return False

    path_str = path_value.strip().replace("\\", "/").lower()
    if not path_str:
        return False

    filename = PurePosixPath(path_str).name
    return "frontal" in filename


def validate_uncertainty_strategy(strategy: str) -> str:
    if strategy not in VALID_UNCERTAINTY_STRATEGIES:
        raise ValueError(
            f"Invalid uncertainty strategy: {strategy}. "
            f"Expected one of {sorted(VALID_UNCERTAINTY_STRATEGIES)}"
        )
    return strategy


def _is_nan_like(value: object) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _normalize_label_value(value: object) -> float:
    if _is_nan_like(value):
        return float("nan")

    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Label value is not numeric: {value!r}") from e