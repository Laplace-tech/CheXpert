from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from chexpert_poc.utils.train_utils import create_dataloaders, load_config

# Dataset 객체 자체가 기대한 상태인지 확인 (train.csv, valid.csv)
def describe_dataset(name: str, dataset) -> None:
  
    print("\n" + "=" * 80)
    print(f"{name} dataset summary")
    print("=" * 80)

    # 원본 CSV row 수 확인
    raw_rows = "unknown"
    if hasattr(dataset, "csv_path") and Path(dataset.csv_path).exists():
        raw_rows = f"{len(pd.read_csv(dataset.csv_path)):,}"

    print(f"csv_path: {dataset.csv_path}")                          # /home/anna/datasets/cxr/chexpert_small/raw/train.csv
    print(f"raw_csv_rows: {raw_rows}")                              # 223,414
    print(f"usable_dataset_rows: {len(dataset):,}")                 # 191,027
    print(f"view_mode: {dataset.view_mode}")                        # frontal_only
    print(f"uncertainty_strategy: {dataset.uncertainty_strategy}")  # U-Ignore
    print(f"target_labels: {dataset.target_labels}")                # ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    if len(dataset) == 0:
        print("[ERROR] dataset is empty")
        return

    sample = dataset[0]
    print("\n[first sample]")
    print(f"sample keys: {list(sample.keys())}")                    # ['image', 'label', 'loss_mask', 'path', 'resolved_path', 'study_id']
    print(f"sample image shape: {tuple(sample['image'].shape)}")    # (3, 320, 320)
    print(f"sample label: {sample['label'].tolist()}")              # [0.0, 0.0, 0.0, 0.0, 0.0]
    print(f"sample loss_mask: {sample['loss_mask'].tolist()}")      # [1.0, 1.0, 1.0, 1.0, 1.0]
    print(f"sample path: {sample['path']}")                         # CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
    print(f"sample resolved_path: {sample['resolved_path']}")       # /home/anna/datasets/cxr/chexpert_small/raw/train/patient00001/study1/view1_frontal.jpg
    print(f"sample study_id: {sample['study_id']}")                 # patient00001/study1


def inspect_batch(name: str, loader) -> None:
    # DataLoader가 실제 학습 가능한 batch를 뽑는지 확인
    print("\n" + "=" * 80)
    print(f"{name} batch summary")
    print("=" * 80)

    batch = next(iter(loader))

    images = batch["image"]
    labels = batch["label"]
    loss_masks = batch["loss_mask"]
    paths = batch["path"]
    study_ids = batch["study_id"]

    print(f"images.shape: {tuple(images.shape)}")           # (32, 3, 320, 320)
    print(f"labels.shape: {tuple(labels.shape)}")           # (32, 5)
    print(f"loss_masks.shape: {tuple(loss_masks.shape)}")   # (32, 5)
    print(f"images.dtype: {images.dtype}")                  # torch.float32
    print(f"labels.dtype: {labels.dtype}")                  # torch.float32
    print(f"loss_masks.dtype: {loss_masks.dtype}")          # torch.float32

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
    print(f"num_masked_entries_in_batch: {ignored_count}")  # U-Ignore로 빠진 라벨 수


def main():
    load_dotenv()

    config_path = "configs/base.yaml"
    config = load_config(config_path)

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