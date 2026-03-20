from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from chexpert_poc.common.config import get_section


def find_latest_best_checkpoint(output_root: str | Path) -> Path:
    # outputs/train_runs/*/checkpoints/best.pt 중 최신 파일 하나 선택
    output_root = Path(output_root)
    candidates = list(output_root.glob("train_runs/*/checkpoints/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt found under: {output_root / 'train_runs'}"
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def resolve_checkpoint_path(
    output_root: str | Path,
    checkpoint: str | Path | None,
) -> Path:
    # --checkpoint를 직접 줬으면 그걸 사용
    # 안 줬으면 최신 best.pt 자동 선택
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    return find_latest_best_checkpoint(output_root)


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be dict, got {type(checkpoint).__name__}: {checkpoint_path}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    return checkpoint


def validate_checkpoint_config(
    checkpoint: dict[str, Any],
    current_config: dict[str, Any],
) -> None:
    # checkpoint 저장 당시 config와 현재 infer config가 너무 다르면
    # label 순서 / 클래스 수 해석이 꼬일 수 있으므로 최소 검증
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return

    if not isinstance(checkpoint_config, dict):
        raise TypeError(
            f"checkpoint['config'] must be dict when present, got "
            f"{type(checkpoint_config).__name__}"
        )

    current_labels = list(get_section(current_config, "data")["target_labels"])
    checkpoint_labels = checkpoint_config.get("data", {}).get("target_labels")

    if checkpoint_labels is not None and list(checkpoint_labels) != current_labels:
        raise ValueError(
            "Current config data.target_labels does not match checkpoint config. "
            f"current={current_labels}, checkpoint={list(checkpoint_labels)}"
        )