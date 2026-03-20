# cd /home/anna/projects/chexpert_poc
# source .venv/bin/activate
# export PYTHONPATH=/home/anna/projects/chexpert_poc
#
# python scripts/check_dataset.py --config configs/base.yaml --sample-size 32
# python scripts/sanity_dataloader.py
# python train.py --config configs/base.yaml
# python eval.py --config configs/base.yaml
# python threshold_tune.py --config configs/base.yaml --criterion f1
# python error_analysis.py --config configs/base.yaml

from __future__ import annotations

import argparse
import csv
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from dotenv import load_dotenv
from tqdm import tqdm

# 모델 구조 생성 담당: DenseNet121 backbone + 마지막 classifier 교체
from chexpert_poc.models.densenet import build_densenet121

# train dataset 기준 각 병변 라벨별로 pos_weight 계산
# 콘솔 출력용 포맷 문자열 생성
from chexpert_poc.utils.class_weights import (
    compute_pos_weight_from_dataset,
    format_pos_weight_stats,
)

# 모델 logits + dataset label / loss mask를 실제 학습 손실로 계산
from chexpert_poc.utils.losses import masked_bce_with_logits

# config 로드 / dataloader 생성 / optimizer 생성 / 저장 유틸
from chexpert_poc.utils.train_utils import (
    build_optimizer,
    create_dataloaders,
)

from chexpert_poc.common.config import get_config_bool, get_section, load_config
from chexpert_poc.common.io import ensure_dir, save_json
from chexpert_poc.common.runtime import get_device



# 로그 출력용: 3721초 -> "1h 02m 01s"
def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_peak_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device=device) / (1024**2)


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


def get_autocast_context(device: torch.device, use_amp: bool):
    if device.type == "cuda" and use_amp:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def make_grad_scaler(device: torch.device, use_amp: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        device=device.type,
        enabled=(use_amp and device.type == "cuda"),
    )


def validate_max_grad_norm(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"max_grad_norm must be > 0 when provided, got {value}")
    return value


def save_history_csv(history: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        return

    fieldnames = list(history[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


# ---------------------------------------------------------
# 안전한 config access helper
# ---------------------------------------------------------

def require_bool(name: str, value: Any) -> bool:
    # 문자열 "false" 같은 걸 bool(...)로 잘못 True 처리하지 않도록
    # 진짜 bool만 허용
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be bool, got {type(value).__name__}: {value!r}")


def get_section(config: dict, section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise TypeError(f"config['{section}'] must be dict, got {type(value).__name__}")
    return value


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


def resolve_num_classes(config: dict) -> int:
    # num_classes의 단일 source of truth를 data.target_labels 길이로 맞춤
    data_cfg = get_section(config, "data")
    model_cfg = get_section(config, "model")

    target_labels = data_cfg.get("target_labels")
    if not isinstance(target_labels, Sequence) or isinstance(target_labels, (str, bytes)):
        raise TypeError("data.target_labels must be a non-string sequence")
    if len(target_labels) == 0:
        raise ValueError("data.target_labels must not be empty")

    derived_num_classes = len(target_labels)

    # model.num_classes가 있으면 일치 여부만 검증
    if "num_classes" in model_cfg:
        configured_num_classes = int(model_cfg["num_classes"])
        if configured_num_classes != derived_num_classes:
            raise ValueError(
                "model.num_classes does not match len(data.target_labels): "
                f"{configured_num_classes} vs {derived_num_classes}"
            )

    return derived_num_classes


def maybe_enable_cudnn_benchmark(config: dict) -> None:
    use_benchmark = get_config_bool(config, "train", "cudnn_benchmark", default=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = use_benchmark


def maybe_set_matmul_precision(config: dict) -> None:
    train_cfg = get_section(config, "train")
    precision = str(train_cfg.get("matmul_precision", "high"))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)


def build_epoch_record(
    epoch: int,
    train_result: dict[str, Any],
    valid_result: dict[str, Any],
    best_valid_loss_before_update: float,
    lr: float,
    epoch_elapsed: float,
    total_elapsed: float,
    eta_seconds: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_loss": train_result["loss"],  # weighted if pos_weight is enabled
        "valid_loss": valid_result["loss"],  # always unweighted
        "train_seconds": train_result["seconds"],
        "valid_seconds": valid_result["seconds"],
        "epoch_seconds": epoch_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "eta_seconds": eta_seconds,
        "train_images": train_result["num_images"],
        "valid_images": valid_result["num_images"],
        "train_images_per_sec": train_result["images_per_sec"],
        "valid_images_per_sec": valid_result["images_per_sec"],
        "train_peak_gpu_memory_mb": train_result["peak_gpu_memory_mb"],
        "valid_peak_gpu_memory_mb": valid_result["peak_gpu_memory_mb"],
        "train_valid_label_count": train_result["valid_label_count"],
        "valid_valid_label_count": valid_result["valid_label_count"],
        "best_valid_loss_so_far": min(best_valid_loss_before_update, valid_result["loss"]),
        "lr": lr,
    }


def build_checkpoint_payload(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
    best_valid_loss: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "best_valid_loss": best_valid_loss,
        "selection_metric": "valid_loss_unweighted",
    }


# 실제 훈련 코드
# - image -> model -> logits
# - logits + label + loss_mask + pos_weight -> masked_bce_with_logits
def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    pos_weight: torch.Tensor | None,
    use_amp: bool,
    channels_last: bool,
    max_grad_norm: float | None,
    epoch: int,
    num_epochs: int,
) -> dict[str, Any]:
    model.train()

    total_loss_sum = 0.0
    total_valid_count = 0.0
    total_images = 0
    total_batches = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cuda_synchronize(device)
    start_time = time.perf_counter()

    pbar = tqdm(
        loader,
        desc=f"train {epoch:02d}/{num_epochs:02d}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(pbar, start=1):
        images = move_tensor(batch["image"], device=device, channels_last=channels_last)
        labels = move_tensor(batch["label"], device=device)
        loss_mask = move_tensor(batch["loss_mask"], device=device)

        optimizer.zero_grad(set_to_none=True)

        with get_autocast_context(device=device, use_amp=use_amp):
            # [densenet.py]
            # - 출력은 sigmoid 없는 raw logits [B, C]
            logits = model(images)

            # [losses.py]
            # - BCEWithLogits 계산
            # - loss_mask==0 는 제외
            # - pos_weight 있으면 양성 오차를 더 크게 반영
            loss_sum = masked_bce_with_logits(
                logits=logits,
                targets=labels,
                loss_mask=loss_mask,
                pos_weight=pos_weight,
                reduction="sum",
            )

            # 유효 라벨 수 기준 평균 loss 산출
            valid_count = loss_mask.sum().clamp_min(1.0)
            loss = loss_sum / valid_count

        if not torch.isfinite(loss).all():
            raise RuntimeError(
                f"Non-finite train loss detected at epoch={epoch}, batch={batch_idx}. "
                f"loss={float(loss.detach().item())}"
            )

        scaler.scale(loss).backward()

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        batch_size = int(images.shape[0])
        total_images += batch_size
        total_batches += 1
        total_loss_sum += float(loss_sum.detach().item())
        total_valid_count += float(loss_mask.sum().detach().item())

        running_loss = total_loss_sum / max(total_valid_count, 1.0)
        pbar.set_postfix(
            loss=f"{running_loss:.4f}",
            imgs=total_images,
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    cuda_synchronize(device)
    elapsed = time.perf_counter() - start_time
    epoch_loss = total_loss_sum / max(total_valid_count, 1.0)
    images_per_sec = total_images / max(elapsed, 1e-8)
    peak_gpu_memory_mb = get_peak_gpu_memory_mb(device)

    return {
        "loss": epoch_loss,
        "valid_label_count": int(total_valid_count),
        "num_images": total_images,
        "num_batches": total_batches,
        "seconds": elapsed,
        "images_per_sec": images_per_sec,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }


@torch.inference_mode()
def validate_one_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
    epoch: int,
    num_epochs: int,
) -> dict[str, Any]:
    model.eval()

    total_loss_sum = 0.0
    total_valid_count = 0.0
    total_images = 0
    total_batches = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cuda_synchronize(device)
    start_time = time.perf_counter()

    pbar = tqdm(
        loader,
        desc=f"valid {epoch:02d}/{num_epochs:02d}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        images = move_tensor(batch["image"], device=device, channels_last=channels_last)
        labels = move_tensor(batch["label"], device=device)
        loss_mask = move_tensor(batch["loss_mask"], device=device)

        with get_autocast_context(device=device, use_amp=use_amp):
            logits = model(images)

            # validation은 모델 선택용이므로 pos_weight를 적용하지 않는다.
            loss_sum = masked_bce_with_logits(
                logits=logits,
                targets=labels,
                loss_mask=loss_mask,
                pos_weight=None,
                reduction="sum",
            )

        batch_size = int(images.shape[0])
        total_images += batch_size
        total_batches += 1
        total_loss_sum += float(loss_sum.detach().item())
        total_valid_count += float(loss_mask.sum().detach().item())

        running_loss = total_loss_sum / max(total_valid_count, 1.0)
        pbar.set_postfix(
            loss=f"{running_loss:.4f}",
            imgs=total_images,
        )

    cuda_synchronize(device)
    elapsed = time.perf_counter() - start_time
    epoch_loss = total_loss_sum / max(total_valid_count, 1.0)
    images_per_sec = total_images / max(elapsed, 1e-8)
    peak_gpu_memory_mb = get_peak_gpu_memory_mb(device)

    return {
        "loss": epoch_loss,
        "valid_label_count": int(total_valid_count),
        "num_images": total_images,
        "num_batches": total_batches,
        "seconds": elapsed,
        "images_per_sec": images_per_sec,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    load_dotenv()

    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    maybe_enable_cudnn_benchmark(config)
    maybe_set_matmul_precision(config)

    # bool(...) 캐스팅 제거: 진짜 bool만 허용
    device = get_device()
    requested_amp = get_config_bool(config, "train", "amp", default=True)
    effective_amp = requested_amp and device.type == "cuda"
    channels_last = get_config_bool(config, "train", "channels_last", default=True)
    use_pos_weight = get_config_bool(config, "train", "use_pos_weight", default=True)
    max_grad_norm = validate_max_grad_norm(get_section(config, "train").get("max_grad_norm", None))

    # num_classes 단일 source of truth = len(data.target_labels)
    num_classes = resolve_num_classes(config)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(config["paths"]["output_root"]) / "train_runs" / run_name)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")

    train_loader, valid_loader = create_dataloaders(config=config)

    train_dataset_stats = getattr(train_loader.dataset, "dataset_stats", None)
    valid_dataset_stats = getattr(valid_loader.dataset, "dataset_stats", None)

    pos_weight = None
    pos_weight_stats = None
    if use_pos_weight:
        pos_weight, pos_weight_stats = compute_pos_weight_from_dataset(
            dataset=train_loader.dataset,
            clip_max=get_section(config, "train").get("pos_weight_clip_max", None),
        )
        pos_weight = pos_weight.to(device)

    pretrained = get_config_bool(config, "model", "pretrained", default=True)

    model = build_densenet121(
        num_classes=num_classes,
        pretrained=pretrained,
    )

    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    optimizer = build_optimizer(model=model, config=config)
    scaler = make_grad_scaler(device=device, use_amp=requested_amp)

    num_epochs = int(config["train"]["epochs"])
    if num_epochs <= 0:
        raise ValueError(f"train.epochs must be > 0, got {num_epochs}")

    best_valid_loss = float("inf")
    history: list[dict[str, Any]] = []

    model_info = {
        "model_name": config["model"]["backbone"],
        "num_classes": num_classes,
        "pretrained": pretrained,
        "trainable_parameters": count_trainable_parameters(model),
        "device": str(device),
        "requested_amp": requested_amp,
        "effective_amp": effective_amp,
        "channels_last": channels_last,
    }

    save_json(config, run_dir / "config_snapshot.json")
    save_json(model_info, run_dir / "model_info.json")
    if train_dataset_stats is not None:
        save_json(train_dataset_stats, run_dir / "train_dataset_stats.json")
    if valid_dataset_stats is not None:
        save_json(valid_dataset_stats, run_dir / "valid_dataset_stats.json")
    if pos_weight_stats is not None:
        save_json(pos_weight_stats, run_dir / "pos_weight_stats.json")

    print("=" * 100)
    print("train.py start")
    print("=" * 100)
    print(f"device              : {device}")
    print(f"requested_amp       : {requested_amp}")
    print(f"effective_amp       : {effective_amp}")
    print(f"channels_last       : {channels_last}")
    print(f"use_pos_weight      : {use_pos_weight}")
    print(f"num_classes         : {num_classes}")
    print(f"pretrained          : {pretrained}")
    print(f"run_dir             : {run_dir}")
    print(f"trainable_parameters: {model_info['trainable_parameters']:,}")

    print("\n[dataloader]")
    print(f"train samples       : {len(train_loader.dataset):,}")
    print(f"valid samples       : {len(valid_loader.dataset):,}")
    print(f"train batches       : {len(train_loader):,}")
    print(f"valid batches       : {len(valid_loader):,}")

    if train_dataset_stats is not None:
        print("\n[train dataset stats]")
        print(train_dataset_stats)
    if valid_dataset_stats is not None:
        print("\n[valid dataset stats]")
        print(valid_dataset_stats)

    if pos_weight_stats is not None:
        print("\n[pos_weight]")
        print(format_pos_weight_stats(pos_weight_stats))

    total_start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        cuda_synchronize(device)
        epoch_start_time = time.perf_counter()

        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            pos_weight=pos_weight,
            use_amp=requested_amp,
            channels_last=channels_last,
            max_grad_norm=max_grad_norm,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        valid_result = validate_one_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            use_amp=requested_amp,
            channels_last=channels_last,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        cuda_synchronize(device)
        epoch_elapsed = time.perf_counter() - epoch_start_time
        total_elapsed = time.perf_counter() - total_start_time
        avg_epoch_time = total_elapsed / epoch
        remaining_epochs = num_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs

        row = build_epoch_record(
            epoch=epoch,
            train_result=train_result,
            valid_result=valid_result,
            best_valid_loss_before_update=best_valid_loss,
            lr=optimizer.param_groups[0]["lr"],
            epoch_elapsed=epoch_elapsed,
            total_elapsed=total_elapsed,
            eta_seconds=eta_seconds,
        )
        history.append(row)

        is_best = row["valid_loss"] < best_valid_loss
        if is_best:
            best_valid_loss = row["valid_loss"]

        print("\n" + "-" * 100)
        print(
            f"[epoch {epoch:02d}/{num_epochs:02d}] "
            f"train_loss={row['train_loss']:.6f} | "
            f"valid_loss(unweighted)={row['valid_loss']:.6f} | "
            f"best_valid={best_valid_loss:.6f}"
        )
        print(
            f"train: {format_seconds(row['train_seconds'])} "
            f"({row['train_images_per_sec']:.2f} img/s, "
            f"peak {row['train_peak_gpu_memory_mb']:.1f} MB)"
        )
        print(
            f"valid: {format_seconds(row['valid_seconds'])} "
            f"({row['valid_images_per_sec']:.2f} img/s, "
            f"peak {row['valid_peak_gpu_memory_mb']:.1f} MB)"
        )
        print(
            f"epoch: {format_seconds(row['epoch_seconds'])} | "
            f"elapsed: {format_seconds(row['total_elapsed_seconds'])} | "
            f"ETA: {format_seconds(row['eta_seconds'])}"
        )
        print(f"lr: {row['lr']:.6g}")
        print("-" * 100)

        save_history_csv(history, run_dir / "history.csv")
        save_json(history, run_dir / "history.json")

        checkpoint = build_checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            best_valid_loss=best_valid_loss,
        )

        save_checkpoint(checkpoint, checkpoint_dir / "last.pt")

        if is_best:
            save_checkpoint(checkpoint, checkpoint_dir / "best.pt")
            print(f"best checkpoint updated -> {checkpoint_dir / 'best.pt'}")

    total_training_time = time.perf_counter() - total_start_time

    print("\n" + "=" * 100)
    print("training finished")
    print("=" * 100)
    print(f"best_valid_loss      : {best_valid_loss:.6f}")
    print(f"total_training_time  : {format_seconds(total_training_time)}")
    print(f"artifacts saved to   : {run_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()