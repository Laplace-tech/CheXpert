from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from chexpert_poc.common.config import get_section


def resolve_num_classes(config: dict[str, Any]) -> int:
    """
    num_classes의 source of truth를 data.target_labels 길이로 맞춘다.

    정책:
    - data.target_labels는 반드시 존재해야 함
    - model.num_classes가 있으면 len(data.target_labels)와 일치하는지 검증만 수행
    """
    data_cfg = get_section(config, "data")
    model_cfg = get_section(config, "model")

    target_labels = data_cfg.get("target_labels")
    if not isinstance(target_labels, Sequence) or isinstance(target_labels, (str, bytes)):
        raise TypeError("data.target_labels must be a non-string sequence")
    if len(target_labels) == 0:
        raise ValueError("data.target_labels must not be empty")

    derived_num_classes = len(target_labels)

    if "num_classes" in model_cfg:
        configured_num_classes = int(model_cfg["num_classes"])
        if configured_num_classes != derived_num_classes:
            raise ValueError(
                "model.num_classes does not match len(data.target_labels): "
                f"{configured_num_classes} vs {derived_num_classes}"
            )

    return derived_num_classes