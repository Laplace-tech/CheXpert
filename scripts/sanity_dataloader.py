# scripts/sanity_dataloader.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chexpert_poc.common.config import load_config
from chexpert_poc.training.data import create_dataloaders


def describe_dataset(name: str, dataset) -> None:
    """
    Dataset 객체가 기대한 상태인지 요약해서 확인한다.

    우선순위:
    - dataset.dataset_stats가 있으면 그 값을 사용
    - 없으면 최소 속성만 출력
    """
    print("\n" + "=" * 80)
    print(f"{name} dataset summary")
    print("=" * 80)

    stats = getattr(dataset, "dataset_stats", {}) or {}

    print(f"csv_path: {getattr(dataset, 'csv_path', 'unknown')}")
    print(f"split: {getattr(dataset, 'split', 'unknown')}")
    print(f"view_mode: {getattr(dataset, 'view_mode', 'unknown')}")
    print(f"uncertainty_strategy: {getattr(dataset, 'uncertainty_strategy', 'unknown')}")
    print(f"target_labels: {getattr(dataset, 'target_labels', 'unknown')}")
    print(f"usable_dataset_rows: {len(dataset):,}")

    if stats:
        print("\n[dataset_stats]")
        for key in (
            "raw_rows",
            "after_view_filter_rows",
            "after_path_resolve_rows",
            "dropped_by_view_filter",
            "dropped_by_unresolved_path",
            "usable_rows",
        ):
            if key in stats:
                print(f"{key}: {stats[key]}")

    if len(dataset) == 0:
        print("[ERROR] dataset is empty")
        return

    sample = dataset[0]
    print("\n[first sample]")
    print(f"sample keys: {list(sample.keys())}")
    print(f"sample image shape: {tuple(sample['image'].shape)}")
    print(f"sample label: {sample['label'].tolist()}")
    print(f"sample loss_mask: {sample['loss_mask'].tolist()}")
    print(f"sample path: {sample['path']}")
    print(f"sample resolved_path: {sample['resolved_path']}")
    print(f"sample study_id: {sample['study_id']}")


def inspect_batch(name: str, loader) -> None:
    """
    DataLoader가 실제 학습 가능한 batch를 뽑는지 확인한다.
    """
    print("\n" + "=" * 80)
    print(f"{name} batch summary")
    print("=" * 80)

    if len(loader) == 0:
        print("[ERROR] loader has no batches")
        return

    batch = next(iter(loader))

    images = batch["image"]
    labels = batch["label"]
    loss_masks = batch["loss_mask"]
    paths = batch["path"]
    study_ids = batch["study_id"]

    print(f"images.shape: {tuple(images.shape)}")
    print(f"labels.shape: {tuple(labels.shape)}")
    print(f"loss_masks.shape: {tuple(loss_masks.shape)}")
    print(f"images.dtype: {images.dtype}")
    print(f"labels.dtype: {labels.dtype}")
    print(f"loss_masks.dtype: {loss_masks.dtype}")

    print("\n[first 3 labels]")
    for i in range(min(3, labels.shape[0])):
        print(f"- idx={i}")
        print(f"  label     : {labels[i].tolist()}")
        print(f"  loss_mask : {loss_masks[i].tolist()}")
        print(f"  path      : {paths[i]}")
        print(f"  study_id  : {study_ids[i]}")

    print("\n[basic checks]")
    print(f"image min: {images.min().item():.4f}")
    print(f"image max: {images.max().item():.4f}")

    valid_label_values = torch.isin(
        labels, torch.tensor([0.0, 1.0], dtype=labels.dtype)
    )
    valid_mask_values = torch.isin(
        loss_masks, torch.tensor([0.0, 1.0], dtype=loss_masks.dtype)
    )

    print(f"labels only in {{0,1}}: {bool(valid_label_values.all())}")
    print(f"loss_masks only in {{0,1}}: {bool(valid_mask_values.all())}")

    ignored_count = int((loss_masks == 0).sum().item())
    print(f"num_masked_entries_in_batch: {ignored_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    train_loader, valid_loader = create_dataloaders(config=config)

    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset

    describe_dataset("train", train_dataset)
    describe_dataset("valid", valid_dataset)

    print("\n" + "=" * 80)
    print("dataloader config summary")
    print("=" * 80)
    print(f"train batch_size: {train_loader.batch_size}")
    print(f"valid batch_size: {valid_loader.batch_size}")
    print(f"train num_batches: {len(train_loader):,}")
    print(f"valid num_batches: {len(valid_loader):,}")

    inspect_batch("train", train_loader)
    inspect_batch("valid", valid_loader)

    print("\n" + "=" * 80)
    print("sanity dataloader check finished")
    print("=" * 80)


if __name__ == "__main__":
    main()