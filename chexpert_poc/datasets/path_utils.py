from __future__ import annotations

from pathlib import Path
from typing import Final, Optional

import pandas as pd

# CSV의 Path 컬럼 값 앞에 붙을 수 있는 prefix 목록
# : resolve_image_path()에서 실제 로컬 파일 경로를 찾을 때 사용
# ex)
# - CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
# - CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg

KNOWN_CSV_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "CheXpert-v1.0-small/",
    "CheXpert-v1.0/",
    "./",
)


def resolve_image_path(raw_root: Path, csv_path_value: object) -> Optional[Path]:
    """
    CSV의 Path 값을 실제 로컬 이미지 경로로 해석한다.

    설계 원칙:
    - Path 값이 절대경로면 그대로 먼저 시도
    - raw_root 기준 경로 시도
    - CheXpert prefix 제거 후 다시 시도
    - raw_root.parent 기준 fallback 시도

    주의:
    - basename(view1_frontal.jpg)만으로 찾는 fallback은 넣지 않는다.
      같은 파일명이 여러 patient/study에 반복될 수 있어서
      잘못된 파일을 조용히 집어올 위험이 있다.
    """
    if pd.isna(csv_path_value):
        return None

    path_str = str(csv_path_value).strip().replace("\\", "/")
    if not path_str:
        return None

    original = Path(path_str)
    candidates: list[Path] = []

    if original.is_absolute():
        candidates.append(original)

    # raw_root 기준 원본 문자열 그대로
    candidates.append(raw_root / path_str)

    stripped = path_str
    for prefix in KNOWN_CSV_PATH_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    stripped = stripped.lstrip("/")

    # prefix 제거 후 재시도
    candidates.append(raw_root / stripped)

    # raw_root.parent 기준 fallback
    candidates.append(raw_root.parent / path_str)
    candidates.append(raw_root.parent / stripped)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None