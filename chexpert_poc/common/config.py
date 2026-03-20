from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_section(config: dict, section: str) -> dict[str, Any]:
    section_value = config.get(section)
    if not isinstance(section_value, dict):
        raise TypeError(
            f"config['{section}'] must be dict, got {type(section_value).__name__}"
        )
    return section_value


def require_bool(name: str, value: Any) -> bool:
    # 문자열 "false" 같은 걸 bool("false") -> True 로 잘못 처리하지 않도록
    # 진짜 bool만 허용
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be bool, got {type(value).__name__}: {value!r}")


def get_config_bool(
    config: dict,
    section: str,
    key: str,
    default: bool | None = None,
) -> bool:
    section_dict = get_section(config, section)

    if key in section_dict:
        return require_bool(f"{section}.{key}", section_dict[key])

    if default is not None:
        return default

    raise KeyError(f"Missing required bool config: {section}.{key}")


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