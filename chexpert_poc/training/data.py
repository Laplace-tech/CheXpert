from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chexpert_poc.common.config import get_config_bool, get_section
from chexpert_poc.datasets.chexpert_dataset import build_chexpert_dataset


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


def _seed_worker(worker_id: int) -> None:
    # worker_id 인자는 PyTorch 시그니처용
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    현재 정책:
    - train shuffle=True
    - valid shuffle=False
    - seed 기반 generator/worker_init_fn으로 재현성 보강
    """
    train_dataset = build_chexpert_dataset(config=config, split="train")
    valid_dataset = build_chexpert_dataset(config=config, split="valid")

    data_cfg = get_section(config, "data")

    batch_size = _validate_positive_int("data.batch_size", data_cfg["batch_size"])
    num_workers = _validate_nonnegative_int("data.num_workers", data_cfg["num_workers"])

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