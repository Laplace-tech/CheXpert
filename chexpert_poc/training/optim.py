from __future__ import annotations

import torch

from chexpert_poc.common.config import get_section


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