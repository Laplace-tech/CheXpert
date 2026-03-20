from __future__ import annotations

import argparse          # CLI 인자 처리 (--config, --checkpoint)
import csv               # 예측 결과 CSV 저장
from pathlib import Path # 파일 경로 다루기
from typing import Any, Sequence

import numpy as np       # 예측 결과 배열 처리 / aggregation
import torch             # 모델 로드 / 추론
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm    # 진행 표시줄

# [chexpert_dataset.py]: valid.csv를 dataset으로 만드는 진입점
# labels.py 정책 / dataset filtering / image transform이 이미 반영된 상태의 dataset을 만듦
from chexpert_poc.datasets.chexpert_dataset import build_chexpert_dataset

# [metrics/classification.py]:
# study-level y_true / y_prob를 받아서 AUROC / AUPRC 같은 분류 지표 계산
from chexpert_poc.metrics.classification import (
    compute_multilabel_classification_metrics,
    format_classification_metrics_table,
)

# [densenet.py]: DenseNet121 모델 구조 생성
# - 여기에 checkpoint의 model_state_dict를 load해서 "학습된 모델을 만든다"
from chexpert_poc.models.densenet import build_densenet121

# [losses.py]: logits -> sigmoid probability 변환
from chexpert_poc.utils.losses import logits_to_probs

from chexpert_poc.common.config import get_config_bool, get_section, load_config
from chexpert_poc.common.io import ensure_dir, save_json
from chexpert_poc.common.runtime import get_device


def get_section(config: dict, section: str) -> dict[str, Any]:
    # config["data"], config["train"], config["model"] 같은 섹션을 안전하게 꺼냄
    section_value = config.get(section)
    if not isinstance(section_value, dict):
        raise TypeError(
            f"config['{section}'] must be dict, got {type(section_value).__name__}"
        )
    return section_value


def require_bool(name: str, value: Any) -> bool:
    # 문자열 "false"를 bool("false") == True 로 잘못 해석하지 않기 위해
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
    # section/key 위치의 bool 설정을 안전하게 읽음
    section_dict = get_section(config, section)

    if key in section_dict:
        return require_bool(f"{section}.{key}", section_dict[key])

    if default is not None:
        return default

    raise KeyError(f"Missing required bool config: {section}.{key}")


def resolve_num_classes(config: dict) -> int:
    # num_classes의 source of truth를 data.target_labels 길이로 일원화
    data_cfg = get_section(config, "data")
    model_cfg = get_section(config, "model")

    target_labels = data_cfg.get("target_labels")
    if not isinstance(target_labels, Sequence) or isinstance(target_labels, (str, bytes)):
        raise TypeError("data.target_labels must be a non-string sequence")
    if len(target_labels) == 0:
        raise ValueError("data.target_labels must not be empty")

    derived_num_classes = len(target_labels)

    # model.num_classes가 있으면 "검증용"으로만 사용
    # 실제 num_classes는 target_labels 길이 기준으로 맞춤
    if "num_classes" in model_cfg:
        configured_num_classes = int(model_cfg["num_classes"])
        if configured_num_classes != derived_num_classes:
            raise ValueError(
                "model.num_classes does not match len(data.target_labels): "
                f"{configured_num_classes} vs {derived_num_classes}"
            )

    return derived_num_classes


def validate_checkpoint_config(
    checkpoint: dict[str, Any],
    current_config: dict[str, Any],
) -> None:
    # checkpoint 저장 당시 config와 현재 eval config가 너무 다르면
    # label 순서나 클래스 수 해석이 꼬일 수 있으므로 최소 검증
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


# =========================================================
# checkpoint helper
# =========================================================

def find_latest_best_checkpoint(output_root: str | Path) -> Path:
    """
    output_root/train_runs 아래에서 가장 최근의 best.pt를 찾는다.
    """
    output_root = Path(output_root)

    # 예: outputs/train_runs/run_20260310_185145/checkpoints/best.pt
    candidates = list(output_root.glob("train_runs/*/checkpoints/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt found under: {output_root / 'train_runs'}"
        )

    # 가장 최근 수정된 best.pt 하나 선택
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def resolve_checkpoint_path(
    output_root: str | Path,
    checkpoint: str | Path | None,
) -> Path:
    # 사용자가 --checkpoint를 직접 줬으면 그걸 사용
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    # 안 줬으면 outputs/train_runs 아래에서 최신 best.pt 자동 탐색
    return find_latest_best_checkpoint(output_root)


# =========================================================
# tensor / loader helper
# =========================================================

def move_tensor(
    x: torch.Tensor,
    device: torch.device,
    channels_last: bool = False,
) -> torch.Tensor:
    # image tensor가 4D([B, C, H, W])이고 channels_last 사용이면
    # memory_format까지 같이 맞춰서 device로 이동
    if channels_last and x.ndim == 4:
        return x.to(
            device=device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return x.to(device=device, non_blocking=True)


def build_valid_loader(config: dict) -> DataLoader:
    """
    eval 전용 valid DataLoader 생성
    - shuffle=False
    - drop_last=False
    """
    
    # [chexpert_dataset.py]: valid.csv를 읽어서 valid dataset 생성
    # - labels.py 정책 / frontal_only / uncertainty 처리 등 dataset 정책은 train과 동일하게 유지
    valid_dataset = build_chexpert_dataset(config=config, split="valid")

    data_cfg = get_section(config, "data")

    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg["num_workers"])

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

    if batch_size <= 0:
        raise ValueError(f"data.batch_size must be > 0, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"data.num_workers must be >= 0, got {num_workers}")

    # PyTorch 제약
    if num_workers == 0:
        persistent_workers = False

    loader_kwargs: dict[str, Any] = {
        "dataset": valid_dataset,
        "batch_size": batch_size,
        "shuffle": False,   # eval은 셔플 안 함
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": False, # eval은 샘플 버리면 안 됨
    }

    # 선택 옵션
    if num_workers > 0 and "prefetch_factor" in data_cfg:
        prefetch_factor = int(data_cfg["prefetch_factor"])
        if prefetch_factor <= 0:
            raise ValueError(
                f"data.prefetch_factor must be > 0, got {prefetch_factor}"
            )
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


# =========================================================
# inference / aggregation
# =========================================================

@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
) -> dict[str, Any]:
    # 평가 모드
    model.eval()

    # 전체 valid set의 결과를 모아둘 리스트
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_loss_masks: list[np.ndarray] = []
    all_paths: list[str] = []
    all_study_ids: list[str] = []

    # valid loader를 배치 단위로 순회
    for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
        # image만 device로 이동
        images = move_tensor(batch["image"], device=device, channels_last=channels_last)

        # 모델 forward -> raw logits [B, C]
        logits = model(images)

        # sigmoid 적용 -> probability [B, C]
        probs = logits_to_probs(logits)

        # 나중에 전체 valid set을 한 번에 metric 계산할 수 있게 numpy로 모음
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(batch["label"].detach().cpu().numpy())
        all_loss_masks.append(batch["loss_mask"].detach().cpu().numpy())
        all_paths.extend(list(batch["path"]))
        all_study_ids.extend(list(batch["study_id"]))

    if not all_probs:
        raise RuntimeError("Evaluation loader produced no batches")

    # 배치 단위 결과를 하나의 배열로 합침
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
    - targets: 방어적으로 class별 max
    - loss_masks: study 내 하나라도 valid면 valid
    """
    probs = np.asarray(probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    loss_masks = np.asarray(loss_masks, dtype=np.float32)

    # shape 검증
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

    # 같은 study_id끼리 row index를 묶음
    groups: dict[str, list[int]] = {}
    for i, sid in enumerate(study_ids):
        groups.setdefault(sid, []).append(i)

    agg_probs: list[np.ndarray] = []
    agg_targets: list[np.ndarray] = []
    agg_loss_masks: list[np.ndarray] = []
    agg_paths: list[str] = []
    agg_study_ids: list[str] = []

    # study별로 하나의 대표 예측으로 합침
    for sid, indices in groups.items():
        idx = np.asarray(indices, dtype=np.int64)

        # 확률은 class별 max
        agg_probs.append(probs[idx].max(axis=0))

        # target도 방어적으로 class별 max
        agg_targets.append(targets[idx].max(axis=0))

        # mask도 하나라도 valid면 valid
        agg_loss_masks.append(loss_masks[idx].max(axis=0))

        # 대표 path는 첫 번째 것만 보관
        agg_paths.append(paths[indices[0]])
        agg_study_ids.append(sid)

    return {
        "probs": np.stack(agg_probs, axis=0),
        "targets": np.stack(agg_targets, axis=0),
        "loss_masks": np.stack(agg_loss_masks, axis=0),
        "paths": agg_paths,
        "study_ids": agg_study_ids,
    }


# =========================================================
# save / metadata
# =========================================================

def save_predictions_csv(
    path: str | Path,
    probs: np.ndarray,
    targets: np.ndarray,
    loss_masks: np.ndarray,
    paths: list[str],
    study_ids: list[str],
    label_names: list[str],
) -> None:
    # study_predictions.csv 저장
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 컬럼 이름 구성
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
    channels_last: bool,
    num_classes: int,
) -> dict[str, Any]:
    # eval_metadata.json 저장용 메타정보
    return {
        "checkpoint_path": str(checkpoint_path),
        "aggregation": "study_max",
        "num_valid_images": int(raw_result["probs"].shape[0]),
        "num_valid_studies": int(study_result["probs"].shape[0]),
        "batch_size": int(valid_loader.batch_size),
        "num_batches": int(len(valid_loader)),
        "target_labels": list(config["data"]["target_labels"]),
        "num_classes": num_classes,
        "channels_last": channels_last,
        "pretrained": False,  # eval은 checkpoint load 전제
        "dataset_stats": getattr(valid_loader.dataset, "dataset_stats", None),
    }


# =========================================================
# main
# =========================================================

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

    # config / device 준비
    config = load_config(args.config)
    device = get_device()

    # checkpoint 경로 결정 후 로드
    output_root = config["paths"]["output_root"]
    checkpoint_path = resolve_checkpoint_path(output_root, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    # checkpoint와 현재 config의 최소 일관성 검증
    validate_checkpoint_config(checkpoint, config)

    # 현재 eval에 사용할 주요 설정 정리
    label_names = list(config["data"]["target_labels"])
    num_classes = resolve_num_classes(config)
    channels_last = get_config_bool(config, "train", "channels_last", default=True)

    print("=" * 100)
    print("eval.py start")
    print("=" * 100)
    print(f"device         : {device}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"channels_last  : {channels_last}")
    print(f"num_classes    : {num_classes}")
    print(f"pretrained     : False (checkpoint load 전제)")

    # valid loader 생성
    valid_loader = build_valid_loader(config=config)

    # 모델 구조 생성
    # 여기서 pretrained=False인 이유:
    # - 곧바로 checkpoint의 학습된 가중치로 덮어쓸 것이기 때문
    model = build_densenet121(
        num_classes=num_classes,
        pretrained=False,
    )

    # 학습된 가중치 로드
    model.load_state_dict(checkpoint["model_state_dict"])

    # memory_format / device 적용
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    # valid 전체에 대해 inference 수행
    raw_result = run_inference(
        model=model,
        loader=valid_loader,
        device=device,
        channels_last=channels_last,
    )

    # image-level 결과를 study-level로 max aggregation
    study_result = aggregate_by_study_max(
        probs=raw_result["probs"],
        targets=raw_result["targets"],
        loss_masks=raw_result["loss_masks"],
        paths=raw_result["paths"],
        study_ids=raw_result["study_ids"],
    )

    # 최종 study-level classification metric 계산
    per_class_results, summary = compute_multilabel_classification_metrics(
        y_true=study_result["targets"],
        y_prob=study_result["probs"],
        label_names=label_names,
        loss_mask=study_result["loss_masks"],
    )

    # eval 결과 저장 폴더 생성
    # 예: outputs/train_runs/run_xxx/eval/
    run_dir = ensure_dir(checkpoint_path.parent.parent / "eval")

    # eval 메타데이터 생성
    eval_metadata = build_eval_metadata(
        config=config,
        checkpoint_path=checkpoint_path,
        valid_loader=valid_loader,
        raw_result=raw_result,
        study_result=study_result,
        channels_last=channels_last,
        num_classes=num_classes,
    )

    # JSON / CSV 산출물 저장
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

    # 콘솔 출력
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