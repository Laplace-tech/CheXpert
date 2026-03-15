from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from chexpert_poc.datasets.chexpert_dataset import build_chexpert_dataset


# ---------------------------------------------------------
# config helper
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# validation helper
# ---------------------------------------------------------

def _validate_positive_int(name: str, value: Any) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _validate_nonnegative_int(name: str, value: Any) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


# ---------------------------------------------------------
# config / seed / device / file I/O
# ---------------------------------------------------------

# configs/base.yaml 같은 설정 파일을 읽어서 dict로 반환
def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # yaml.safe_load(...)는 YAML의 true/false를 파이썬 bool로 읽어줌
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(config, dict):
        raise TypeError(
            f"Config root must be a dict, got {type(config).__name__}: {config_path}"
        )

    return config


def set_seed(seed: int) -> None:
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    # worker_id 인자는 PyTorch 시그니처용
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# GPU 가능하면 cuda, 아니면 cpu 선택
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 출력 폴더가 없으면 생성
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# JSON 파일 저장. 부모 디렉터리가 없으면 자동 생성
def save_json(data: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 학습 중 best.pt / last.pt 같은 체크포인트를 저장
def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ---------------------------------------------------------
# dataloader / optimizer
# ---------------------------------------------------------

# train/valid DataLoader를 생성
def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    현재 정책:
    - train shuffle=True
    - valid shuffle=False
    - seed 기반 generator/worker_init_fn으로 재현성 보강
    """
    # 1) chexpert_dataset.py 의 build_chexpert_dataset(...)로 train / valid dataset 생성
    train_dataset = build_chexpert_dataset(config=config, split="train")
    valid_dataset = build_chexpert_dataset(config=config, split="valid")

    # 2) base.yaml 에서 batch_size, num_workers 읽어서 검증
    data_cfg = get_section(config, "data")

    batch_size = _validate_positive_int("data.batch_size", data_cfg["batch_size"])
    num_workers = _validate_nonnegative_int("data.num_workers", data_cfg["num_workers"])

    # bool(...) 제거: 진짜 bool만 허용
    pin_memory = get_config_bool(
        config,
        "data",
        "pin_memory",
        default=torch.cuda.is_available(),
    )
    persistent_workers = get_config_bool(
        config,
        "data",
        "persistent_workers",
        default=(num_workers > 0),
    )
    drop_last = get_config_bool(
        config,
        "data",
        "drop_last",
        default=False,
    )

    # PyTorch 제약: persistent_workers는 num_workers > 0 일 때만 가능
    if num_workers == 0:
        persistent_workers = False

    project_cfg = config.get("project", {})
    if not isinstance(project_cfg, dict):
        raise TypeError(
            f"config['project'] must be dict, got {type(project_cfg).__name__}"
        )

    seed = int(project_cfg.get("seed", 42))
    generator = torch.Generator()
    generator.manual_seed(seed)

    # train loader 설정
    train_loader_kwargs: dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": drop_last,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }

    # valid loader 설정
    valid_loader_kwargs: dict[str, Any] = {
        "dataset": valid_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": False,
        "worker_init_fn": _seed_worker,
    }

    # prefetch_factor는 num_workers > 0 일 때만 유효
    if num_workers > 0 and "prefetch_factor" in data_cfg:
        prefetch_factor = _validate_positive_int(
            "data.prefetch_factor",
            data_cfg["prefetch_factor"],
        )
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        valid_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)
    valid_loader = DataLoader(**valid_loader_kwargs)

    return train_loader, valid_loader


# model.parameters()에 대해 optimizer 생성
# - build_densenet121(...)으로 만든 model이 여기로 들어옴 (densenet.py)
def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    현재 지원 optimizer:
    - Adam
    - AdamW
    """
    train_cfg = get_section(config, "train")

    optimizer_name = str(train_cfg["optimizer"]).lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])

    if lr <= 0.0:
        raise ValueError(f"train.lr must be > 0, got {lr}")
    if weight_decay < 0.0:
        raise ValueError(f"train.weight_decay must be >= 0, got {weight_decay}")

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']}")