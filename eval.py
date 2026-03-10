from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from chexpert_poc.datasets.chexpert_dataset import build_chexpert_dataset
from chexpert_poc.metrics.classification import (
    compute_multilabel_classification_metrics,
    format_classification_metrics_table,
)
from chexpert_poc.models.densenet import build_densenet121
from chexpert_poc.utils.losses import logits_to_probs
from chexpert_poc.utils.train_utils import ensure_dir, get_device, load_config, save_json


def find_latest_best_checkpoint(output_root: str | Path) -> Path:
    """
    output_root/train_runs 아래에서 가장 최근의 best.pt를 찾는다.
    우선 mtime 기준으로 정렬하고, 없으면 예외를 던진다.
    """
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
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    return find_latest_best_checkpoint(output_root)


def move_tensor(
    x: torch.Tensor,
    device: torch.device,
    channels_last: bool = False,
) -> torch.Tensor:
    if channels_last and x.ndim == 4:
        return x.to(
            device=device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return x.to(device=device, non_blocking=True)


def build_valid_loader(config: dict) -> DataLoader:
    """
    eval 전용 valid DataLoader를 생성한다.

    train과 동일한 dataset 정책을 따르되,
    - shuffle=False
    - drop_last=False
    로 고정한다.
    """
    valid_dataset = build_chexpert_dataset(config=config, split="valid")

    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    pin_memory = bool(config["data"].get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(
        config["data"].get("persistent_workers", num_workers > 0)
    )

    if batch_size <= 0:
        raise ValueError(f"data.batch_size must be > 0, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"data.num_workers must be >= 0, got {num_workers}")

    if num_workers == 0:
        persistent_workers = False

    loader_kwargs: dict[str, Any] = {
        "dataset": valid_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": False,
    }

    if num_workers > 0 and "prefetch_factor" in config.get("data", {}):
        prefetch_factor = int(config["data"]["prefetch_factor"])
        if prefetch_factor <= 0:
            raise ValueError(
                f"data.prefetch_factor must be > 0, got {prefetch_factor}"
            )
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
) -> dict[str, Any]:
    model.eval()

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_loss_masks: list[np.ndarray] = []
    all_paths: list[str] = []
    all_study_ids: list[str] = []

    for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
        images = move_tensor(batch["image"], device=device, channels_last=channels_last)
        logits = model(images)
        probs = logits_to_probs(logits)

        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(batch["label"].detach().cpu().numpy())
        all_loss_masks.append(batch["loss_mask"].detach().cpu().numpy())
        all_paths.extend(list(batch["path"]))
        all_study_ids.extend(list(batch["study_id"]))

    if not all_probs:
        raise RuntimeError("Evaluation loader produced no batches")

    return {
        "probs": np.concatenate(all_probs, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
        "loss_masks": np.concatenate(all_loss_masks, axis=0),
        "paths": all_paths,
        "study_ids": all_study_ids,
    }


def aggregate_by_study_max(
    probs: np.ndarray,
    targets: np.ndarray,
    loss_masks: np.ndarray,
    paths: list[str],
    study_ids: list[str],
) -> dict[str, Any]:
    """
    study-level max aggregation

    - probs: study 내 이미지들에 대해 class별 max
    - targets: 동일 study 내 동일해야 하는 게 일반적이지만, 방어적으로 class별 max
    - loss_masks: study 내 하나라도 valid면 valid
    """
    probs = np.asarray(probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    loss_masks = np.asarray(loss_masks, dtype=np.float32)

    if probs.ndim != 2 or targets.ndim != 2 or loss_masks.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got probs={probs.shape}, "
            f"targets={targets.shape}, loss_masks={loss_masks.shape}"
        )

    if not (probs.shape == targets.shape == loss_masks.shape):
        raise ValueError(
            f"Shape mismatch: probs={probs.shape}, "
            f"targets={targets.shape}, loss_masks={loss_masks.shape}"
        )

    n_rows = probs.shape[0]
    if len(paths) != n_rows or len(study_ids) != n_rows:
        raise ValueError(
            f"Length mismatch: n_rows={n_rows}, len(paths)={len(paths)}, "
            f"len(study_ids)={len(study_ids)}"
        )

    if n_rows == 0:
        raise ValueError("Cannot aggregate empty predictions")

    groups: dict[str, list[int]] = {}
    for i, sid in enumerate(study_ids):
        groups.setdefault(sid, []).append(i)

    agg_probs: list[np.ndarray] = []
    agg_targets: list[np.ndarray] = []
    agg_loss_masks: list[np.ndarray] = []
    agg_paths: list[str] = []
    agg_study_ids: list[str] = []

    for sid, indices in groups.items():
        idx = np.asarray(indices, dtype=np.int64)

        agg_probs.append(probs[idx].max(axis=0))
        agg_targets.append(targets[idx].max(axis=0))
        agg_loss_masks.append(loss_masks[idx].max(axis=0))
        agg_paths.append(paths[indices[0]])
        agg_study_ids.append(sid)

    return {
        "probs": np.stack(agg_probs, axis=0),
        "targets": np.stack(agg_targets, axis=0),
        "loss_masks": np.stack(agg_loss_masks, axis=0),
        "paths": agg_paths,
        "study_ids": agg_study_ids,
    }


def save_predictions_csv(
    path: str | Path,
    probs: np.ndarray,
    targets: np.ndarray,
    loss_masks: np.ndarray,
    paths: list[str],
    study_ids: list[str],
    label_names: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["path", "study_id"]
    for label in label_names:
        fieldnames.extend(
            [
                f"{label}_target",
                f"{label}_prob",
                f"{label}_mask",
            ]
        )

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(paths)):
            row = {
                "path": paths[i],
                "study_id": study_ids[i],
            }
            for j, label in enumerate(label_names):
                row[f"{label}_target"] = float(targets[i, j])
                row[f"{label}_prob"] = float(probs[i, j])
                row[f"{label}_mask"] = float(loss_masks[i, j])
            writer.writerow(row)


def build_eval_metadata(
    config: dict,
    checkpoint_path: Path,
    valid_loader: DataLoader,
    raw_result: dict[str, Any],
    study_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "aggregation": "study_max",
        "num_valid_images": int(raw_result["probs"].shape[0]),
        "num_valid_studies": int(study_result["probs"].shape[0]),
        "batch_size": int(valid_loader.batch_size),
        "num_batches": int(len(valid_loader)),
        "target_labels": list(config["data"]["target_labels"]),
        "dataset_stats": getattr(valid_loader.dataset, "dataset_stats", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="explicit checkpoint path; if omitted, latest best.pt is used",
    )
    args = parser.parse_args()

    load_dotenv()

    config = load_config(args.config)
    device = get_device()

    output_root = config["paths"]["output_root"]
    checkpoint_path = resolve_checkpoint_path(output_root, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    print("=" * 100)
    print("eval.py start")
    print("=" * 100)
    print(f"device         : {device}")
    print(f"checkpoint_path: {checkpoint_path}")

    label_names = list(config["data"]["target_labels"])
    channels_last = bool(config["train"].get("channels_last", True))

    valid_loader = build_valid_loader(config=config)

    model = build_densenet121(
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    raw_result = run_inference(
        model=model,
        loader=valid_loader,
        device=device,
        channels_last=channels_last,
    )

    study_result = aggregate_by_study_max(
        probs=raw_result["probs"],
        targets=raw_result["targets"],
        loss_masks=raw_result["loss_masks"],
        paths=raw_result["paths"],
        study_ids=raw_result["study_ids"],
    )

    per_class_results, summary = compute_multilabel_classification_metrics(
        y_true=study_result["targets"],
        y_prob=study_result["probs"],
        label_names=label_names,
        loss_mask=study_result["loss_masks"],
    )

    run_dir = ensure_dir(checkpoint_path.parent.parent / "eval")

    eval_metadata = build_eval_metadata(
        config=config,
        checkpoint_path=checkpoint_path,
        valid_loader=valid_loader,
        raw_result=raw_result,
        study_result=study_result,
    )

    save_json(per_class_results, run_dir / "per_class_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")
    save_json(eval_metadata, run_dir / "eval_metadata.json")

    save_predictions_csv(
        path=run_dir / "study_predictions.csv",
        probs=study_result["probs"],
        targets=study_result["targets"],
        loss_masks=study_result["loss_masks"],
        paths=study_result["paths"],
        study_ids=study_result["study_ids"],
        label_names=label_names,
    )

    print("\n[summary]")
    print(f"num_studies : {summary['num_samples']}")
    print(f"mean_auroc  : {summary['mean_auroc']:.4f}")
    print(f"mean_auprc  : {summary['mean_auprc']:.4f}")

    print("\n[per-class metrics]")
    print(format_classification_metrics_table(per_class_results))

    print("\nartifacts saved to:")
    print(run_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()