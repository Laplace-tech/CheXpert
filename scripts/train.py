from __future__ import annotations

import argparse  # --config 같은 CLI 인자 받기
import csv  # history.csv 저장
import sys
import time  # epoch 시간 측정
from contextlib import nullcontext  # AMP 안 쓸 때 autocast 대신 빈 context로 사용
from datetime import datetime  # run_YYYYmmdd_HHMMSS 형태 실험 디렉토리 이름 만들기
from pathlib import Path  # 경로 처리
from typing import Any  # config / history row 같은 느슨한 dict 타입 처리

import torch  # PyTorch 핵심
from tqdm import tqdm  # progress bar 출력

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chexpert_poc.common.config import get_config_bool, get_section, load_config
from chexpert_poc.common.io import ensure_dir, save_json, save_checkpoint
from chexpert_poc.common.runtime import get_device, set_seed
from chexpert_poc.common.model_config import resolve_num_classes

# 모델 구조 생성 담당
# 최종적으로 DenseNet121 backbone + classifier(num_classes) 구성
from chexpert_poc.models.densenet import build_densenet121

# train dataset 기준 양성/음성 불균형을 계산한 뒤,
# BCEWithLogitsLoss의 pos_weight용 tensor 생성
# 그리고 그 통계 문자열 출력 포맷 생성
from chexpert_poc.training.class_weights import (
    compute_pos_weight_from_dataset,
    format_pos_weight_stats,
)

# logits, targets, loss_mask, pos_weight 받아서 실제 multi-label BCE loss 계산
# uncertainty masking 반영되는 핵심 손실 함수
from chexpert_poc.training.losses import masked_bce_with_logits

# optimizer 생성 로직
from chexpert_poc.training.optim import build_optimizer

# train/valid dataloader 생성 로직
from chexpert_poc.training.data import create_dataloaders


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


# CUDA는 비동기 실행이므로 시간 측정 직전/직후에 synchronize 안 하면
# 실제보다 시간이 짧게 찍힐 수 있음
def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# epoch 동안 최대 GPU 메모리 사용량(MB) 조회
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
        "best_valid_loss_so_far": min(
            best_valid_loss_before_update, valid_result["loss"]
        ),
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
    model: torch.nn.Module,  # 학습할 모델
    loader,  # train dataloader
    optimizer: torch.optim.Optimizer,  # Adam / Adamw 등
    device: torch.device,  # cpu 또는 cuda
    scaler: torch.amp.GradScaler,  # AMP gradient scaler
    pos_weight: torch.Tensor | None,  # 클래스 불균형 보정용 tensor
    use_amp: bool,  # autocast 사용 여부
    channels_last: bool,  # 이미지 텐서를 channels_last로 보낼지 여부
    max_grad_norm: float | None,  # gradient clipping 값 또는 None
    epoch: int,  # 현재 epoch 번호
    num_epochs: int,  # 전체 epoch 수
) -> dict[str, Any]:
    model.train()

    # epoch 전체 통계 누적용 변수
    total_loss_sum = 0.0  # reduction="sum" 기준 누적 손실ㄹ
    total_valid_count = (
        0.0  # 실제 loss에 반영된 "유효 라벨 수" (uncertainty mask 제외시켜야 함)
    )
    total_images = 0  # 처리한 이미지 수
    total_batches = 0  # 처리한 배치 수

    # GPU 메모리: 이번 epoch의 peak memory 측정을 새로 시작
    if device.type == "cuda":
        # 이번 epoch의 peak memory 측정을 새로 시작
        torch.cuda.reset_peak_memory_stats(device)

    # 시간 초기화: time 측정 정확도 위해 synchronize 후 시작
    cuda_synchronize(device)
    start_time = time.perf_counter()

    # progress bar 생성
    pbar = tqdm(
        loader,
        desc=f"train {epoch:02d}/{num_epochs:02d}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(pbar, start=1):
        images = move_tensor(
            batch["image"], device=device, channels_last=channels_last
        )  # 뭔 개소리지
        labels = move_tensor(batch["label"], device=device)
        loss_mask = move_tensor(batch["loss_mask"], device=device)

        # gradient 초기화
        optimizer.zero_grad(set_to_none=True)

        with get_autocast_context(device=device, use_amp=use_amp):
            # [densenet.py]
            # - 출력은 sigmoid 없는 raw logits [B, C]
            logits = model(images)

            # [losses.py]: 손실 계산 철학
            # 1) multi-label BCE 사용
            # 2) uncertainty는 loss_mask로 제외 가능
            # 3) 불균형 클래스는 pos_weight로 양성 오차를 더 크게 반영
            loss_sum = masked_bce_with_logits(
                logits=logits,  # 모델 출력 raw logits
                targets=labels,  # 정답 라벨
                loss_mask=loss_mask,  # uncertain label 무시용 마스크
                pos_weight=pos_weight,  # 양성 희소 클래스 가중치
                reduction="sum",
            )

            # 유효 라벨 수 기준 평균 loss 산출
            valid_count = loss_mask.sum().clamp_min(1.0)
            loss = loss_sum / valid_count

        # NaN / Inf 손실 발생 시 즉시 중단
        if not torch.isfinite(loss).all():
            raise RuntimeError(
                f"Non-finite train loss detected at epoch={epoch}, batch={batch_idx}. "
                f"loss={float(loss.detach().item())}"
            )

        # backward + gradient clipping + optimizer step
        scaler.scale(loss).backward()

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        # epoch 전체 통계 누적
        # 주의: loss가 아니라 loss_sum을 누적한 뒤
        # 마지막에 total_valid_count로 나눔
        batch_size = int(images.shape[0])
        total_images += batch_size
        total_batches += 1
        total_loss_sum += float(loss_sum.detach().item())
        total_valid_count += float(loss_mask.sum().detach().item())

        # running loss 표시
        running_loss = total_loss_sum / max(total_valid_count, 1.0)
        pbar.set_postfix(
            loss=f"{running_loss:.4f}",
            imgs=total_images,
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    # epoch 종료 후 정리
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

    # 실행 예: python train.py --config configs/base.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)  # yaml config 로드
    set_seed(int(config["project"]["seed"]))  # 랜덤 시드 고정
    maybe_enable_cudnn_benchmark(config)  # cudnn benchmark 설정
    maybe_set_matmul_precision(config)  # matmul precision 설정

    # device         : cuda 있으면 cuda, 없으면 cpu
    # requested_amp  : 사용자가 config에서 원한 AMP 사용 여부
    # effective_amp  : 실제로 적용 가능한 AMP 여부 (cuda일 때만 True)
    # channels_last  : image tensor를 channels_last 포맷으로 보낼지
    # use_pos_weight : pos_weight 계산/사용 여부
    # max_grad_norm  : gradient clipping 값
    device = get_device()
    requested_amp = get_config_bool(config, "train", "amp", default=True)
    effective_amp = requested_amp and device.type == "cuda"
    channels_last = get_config_bool(config, "train", "channels_last", default=True)
    use_pos_weight = get_config_bool(config, "train", "use_pos_weight", default=True)
    max_grad_norm = validate_max_grad_norm(
        get_section(config, "train").get("max_grad_norm", None)
    )

    # data.target_labels 길이 기준으로 num_classes 계산
    num_classes = resolve_num_classes(config)

    # 예:
    # outputs/train_runs/run_20260320_185145/
    # outputs/train_runs/run_20260320_185145/checkpoints/
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(config["paths"]["output_root"]) / "train_runs" / run_name)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")

    # train/valid dataset 및 dataloader 생성
    train_loader, valid_loader = create_dataloaders(config=config)

    train_dataset_stats = getattr(train_loader.dataset, "dataset_stats", None)
    valid_dataset_stats = getattr(valid_loader.dataset, "dataset_stats", None)

    # pos_weight 계산 (train dataset 기준으로)
    pos_weight = None
    pos_weight_stats = None
    if use_pos_weight:
        pos_weight, pos_weight_stats = compute_pos_weight_from_dataset(
            dataset=train_loader.dataset,
            clip_max=get_section(config, "train").get("pos_weight_clip_max", None),
        )
        pos_weight = pos_weight.to(device)

    # 모델 생성: DenseNet121
    pretrained = get_config_bool(config, "model", "pretrained", default=True)
    model = build_densenet121(
        num_classes=num_classes,  # classifier 출력 차원 = num_classes
        pretrained=pretrained,    # pretrained 여부는 config로 제어
    )

    # 모델도 channels_last memory format으로 맞춤
    # 이후에 device로 이동
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    optimizer = build_optimizer(model=model, config=config)
    scaler = make_grad_scaler(device=device, use_amp=requested_amp)

    # 필수 학습 epoch 수 검증
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

    # 각종 artifact 저장: 이번 run의 설정과 데이터셋 / 모델 통계
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
