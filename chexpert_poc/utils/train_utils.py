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


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    YAML 설정 파일을 로드한다.

    기대:
    - 최상위가 dict
    - 비어 있지 않을 것
    """
    config_path = Path(config_path)

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

    return config


def set_seed(seed: int) -> None:
    """
    Python / NumPy / PyTorch 전역 seed를 설정한다.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """
    DataLoader worker별 seed를 안정적으로 맞춘다.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict | list, path: str | Path) -> None:
    """
    JSON 파일 저장. 부모 디렉터리가 없으면 자동 생성한다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


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


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    train/valid DataLoader를 생성한다.

    현재 정책:
    - train shuffle=True
    - valid shuffle=False
    - seed 기반 generator/worker_init_fn으로 재현성 보강
    """
    train_dataset = build_chexpert_dataset(config=config, split="train")
    valid_dataset = build_chexpert_dataset(config=config, split="valid")

    batch_size = _validate_positive_int("data.batch_size", config["data"]["batch_size"])
    num_workers = _validate_nonnegative_int(
        "data.num_workers", config["data"]["num_workers"]
    )

    pin_memory = bool(config["data"].get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(
        config["data"].get("persistent_workers", num_workers > 0)
    )
    drop_last = bool(config["data"].get("drop_last", False))

    # PyTorch 제약: persistent_workers는 num_workers > 0 일 때만 가능
    if num_workers == 0:
        persistent_workers = False

    seed = int(config.get("project", {}).get("seed", 42))
    generator = torch.Generator()
    generator.manual_seed(seed)

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
    if num_workers > 0 and "prefetch_factor" in config.get("data", {}):
        prefetch_factor = _validate_positive_int(
            "data.prefetch_factor",
            config["data"]["prefetch_factor"],
        )
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        valid_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)
    valid_loader = DataLoader(**valid_loader_kwargs)

    return train_loader, valid_loader


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    현재 지원 optimizer:
    - Adam
    - AdamW
    """
    optimizer_name = str(config["train"]["optimizer"]).lower()
    lr = float(config["train"]["lr"])
    weight_decay = float(config["train"]["weight_decay"])

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

    raise ValueError(f"Unsupported optimizer: {config['train']['optimizer']}")