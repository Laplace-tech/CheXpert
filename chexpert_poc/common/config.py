# chexpert_poc/common/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PATH_KEYS_TO_NORMALIZE = ("chexpert_root", "output_root")
SPLIT_TO_CSV_KEY = {
    "train": "train_csv",
    "valid": "valid_csv",
    "test": "test_csv",
}


def get_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """
    config에서 특정 section(dict)을 꺼낸다.

    예:
    - get_section(config, "data")
    - get_section(config, "paths")
    """
    
    section_value = config.get(section)
    if not isinstance(section_value, dict):
        raise TypeError(
            f"config['{section}'] must be dict, got {type(section_value).__name__}"
        )
    return section_value


def require_bool(name: str, value: Any) -> bool:
    """
    config 값이 '진짜 bool'인지 검증

    문자열 "false"는 bool("false") == True 같은 위험한 동작을 만들 수 있으므로
    여기서는 bool 타입만 허용
    """
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be bool, got {type(value).__name__}: {value!r}")


def get_config_bool(
    config: dict[str, Any],
    section: str,
    key: str,
    default: bool | None = None,
) -> bool:
    """
    config[section][key]에서 bool 값을 안전하게 읽는다.

    동작:
    - 키가 있으면 require_bool()로 타입 검증 후 반환
    - 키가 없고 default가 있으면 default 반환
    - 둘 다 아니면 KeyError
    """
    section_dict = get_section(config, section)

    if key in section_dict:
        return require_bool(f"{section}.{key}", section_dict[key])

    if default is not None:
        return default

    raise KeyError(f"Missing required bool config: {section}.{key}")


def get_split_csv_path(config: dict[str, Any], split: str) -> Path:
    """
    train / valid / test split에 대응하는 CSV 절대경로를 반환한다.

    전제:
    - paths.chexpert_root 는 load_config() 이후 절대경로로 정규화되어 있음
    - data.train_csv / valid_csv / test_csv 는 파일명 또는 상대경로 문자열임

    예:
    - get_split_csv_path(config, "train")
    - get_split_csv_path(config, "valid")
    - get_split_csv_path(config, "test")
    """
    if split not in SPLIT_TO_CSV_KEY:
        raise ValueError(f"Unsupported split: {split}. Expected one of {tuple(SPLIT_TO_CSV_KEY)}")

    paths_cfg = get_section(config, "paths")
    data_cfg = get_section(config, "data")

    raw_root = Path(paths_cfg["chexpert_root"])
    csv_value = data_cfg.get(SPLIT_TO_CSV_KEY[split])

    if csv_value is None:
        raise KeyError(f"Missing required config: data.{SPLIT_TO_CSV_KEY[split]}")

    csv_path = Path(str(csv_value))
    if csv_path.is_absolute():
        return csv_path.resolve()

    return (raw_root / csv_path).resolve()


def _normalize_path_fields(
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    """
    config 안의 paths.* 상대경로를 프로젝트 루트 기준 절대경로로 바꾼다.

    현재 전제:
    - config 파일은 보통 <project_root>/configs/base.yaml 에 위치
    - 따라서 project_root = config_path.parent.parent
    """
    paths_cfg = config.get("paths")
    if not isinstance(paths_cfg, dict):
        return config

    project_root = config_path.parent.parent

    for key in ("chexpert_root", "output_root"):
        raw_value = paths_cfg.get(key)
        if raw_value is None:
            continue

        path_value = Path(str(raw_value))
        if not path_value.is_absolute():
            paths_cfg[key] = str((project_root / path_value).resolve())
        else:
            paths_cfg[key] = str(path_value.resolve())

    return config


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    YAML config를 로드하고, 공통 path 정규화를 수행한 뒤 반환한다.
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(config, dict):
        raise TypeError(
            f"Config root must be a dict, got {type(config).__name__}: {config_path}"
        )

    config = _normalize_path_fields(config, config_path)
    return config