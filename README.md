# CheXpert PoC

CheXpert-small 기반 흉부 X-ray 멀티라벨 분류 PoC 저장소입니다.  
이 리포는 **학습(train) / 평가(eval) / threshold tuning / error analysis / 추론(infer) / Grad-CAM**까지 포함하는 **실험 리포**이며, 향후 `capstone-cxr`의 `ai-service`로 옮길 **최소 추론 코어의 source repo** 역할도 함께 합니다.

---

## 1. 이 리포의 역할

이 프로젝트는 아래 두 목적을 동시에 가집니다.

- **실험 리포**: CheXpert-small 기반 baseline 재현, 비교 실험, 분석 기록
- **이식용 source repo**: 나중에 서비스에 넣을 최소 추론 코어 정리

중요한 원칙은 명확합니다.

- `chexpert_poc`는 **실험 리포**입니다.
- `capstone-cxr`는 **제품 리포**입니다.
- 이 리포 전체를 서비스로 복붙하지 않습니다.
- 서비스에는 여기서 검증된 **최소 추론 코어만** 옮깁니다.

---

## 2. 지금 프로젝트 상태

현재 이 리포는 **리팩토링 1차를 마쳤고**, 이제는 구조를 더 바꾸는 단계가 아니라 **baseline 재확립 및 비교 실험 관리 단계**에 들어와 있습니다.

현재 판단은 아래와 같습니다.

- `common / datasets / training / evaluation / inference / scripts` 구조가 정리됨
- 공용 후처리 로직이 `chexpert_poc/evaluation/*`로 분리됨
- 추론 쪽 최소 코어가 `chexpert_poc/inference/*` 기준으로 정리됨
- `configs/base.yaml` 기준 repo-relative path 정책이 정착됨
- 우선순위는 추가 리팩토링이 아니라 **baseline 확정 → 비교 실험 → 기록 축적**임

---

## 3. 문서 빠른 이동

### 가장 먼저 읽을 것
- [실험노트 모음](docs/experiments/README.md)
- [Baseline 01 실험 기록](docs/experiments/baseline_01_run_20260321_125758.md)
- [실험 기록 템플릿](docs/experiments/_template.md)

### 지금 당장 실행할 때 볼 것
- 아래 **실행 파이프라인** 섹션
- `configs/base.yaml`
- `scripts/`

### 결과가 어디에 저장되는지 볼 것
- `outputs/train_runs/<run_id>/`
- `logs/`
- `docs/experiments/`

---

## 4. 현재 baseline 정책

현재 기준 baseline은 아래 설정을 기본으로 합니다.

### 데이터셋
- Dataset: `CheXpert-small`
- Root path: `data/chexpert_small/raw`

### 타겟 라벨
공식 5-task subset
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion

### 데이터 정책
- View mode: `frontal_only`
- Uncertainty policy: `U-Ignore`
- Evaluation aggregation: `study-level max`

### 모델 / 학습 정책
- Backbone: `DenseNet121`
- Input size: `320`
- Optimizer: `Adam`
- Learning rate: `1e-4`
- `pos_weight_clip_max`: `8`
- AMP: `True`
- Channels-last: `True`

### 평가 정책
- Class-wise `AUROC`, `AUPRC`
- Mean `AUROC`, Mean `AUPRC`
- Threshold tuning sweep: `0.05 ~ 0.95`

---

## 5. 기준 baseline 결과 요약

현재 1차 baseline 기록은 아래 run을 기준으로 정리되어 있습니다.

- Run ID: `run_20260321_125758`
- Best epoch: `3`
- Selection rule: **best validation loss checkpoint (`best.pt`)**
- Mean AUROC: `0.8811`
- Mean AUPRC: `0.7387`

핵심 해석:

- 학습 파이프라인은 정상 동작함
- epoch 3 이후 과적합 패턴이 보임
- baseline 모델은 마지막 epoch가 아니라 **best.pt** 기준으로 봐야 함
- F1 기준 threshold tuning 결과 대부분의 클래스가 `0.5`보다 낮은 threshold를 선호함
- 현재 baseline은 성능은 준수하지만, threshold 기준으로는 **recall 쪽으로 약간 기운 운영점**임

자세한 내용은 아래 문서를 봅니다.

- [Baseline 01 실험 기록](docs/experiments/baseline_01_run_20260321_125758.md)

---

## 6. 현재 리포 구조

```text
.
├── README.md
├── checkpoints
├── chexpert_poc
│   ├── common
│   ├── datasets
│   ├── evaluation
│   ├── explain
│   ├── inference
│   ├── metrics
│   ├── models
│   └── training
├── configs
│   └── base.yaml
├── data
│   └── chexpert_small/raw
├── docs
│   └── experiments
├── logs
├── outputs
│   ├── gradcam_runs
│   ├── infer_runs
│   └── train_runs
└── scripts
    ├── check_dataset.py
    ├── sanity_dataloader.py
    ├── train.py
    ├── eval.py
    ├── threshold_tune.py
    ├── error_analysis.py
    ├── infer.py
    └── gradcam_demo.py
```

---

## 7. 패키지 역할 요약

### `chexpert_poc/common`
공용 설정, IO, runtime 유틸리티를 둡니다.

### `chexpert_poc/datasets`
CSV 로드, 경로 해석, label/loss mask 생성, view policy 적용을 담당합니다.

### `chexpert_poc/training`
DataLoader 생성, loss, optimizer, class weight 계산을 담당합니다.

### `chexpert_poc/evaluation`
`eval.py`, `threshold_tune.py`, `error_analysis.py`에서 쓰는 후처리 로직을 담당합니다.

### `chexpert_poc/inference`
checkpoint 로드, 입력 이미지 검증, 확률 후처리, 추론 결과 저장, 서비스 이식용 entry를 담당합니다.

### `scripts`
실행용 엔트리포인트만 둡니다.

---

## 8. 실행 파이프라인

기본 실행 순서는 아래와 같습니다.

```text
check_dataset
-> sanity_dataloader
-> train
-> eval
-> threshold_tune
-> error_analysis
-> infer
-> gradcam_demo
```

실제 명령어는 다음과 같습니다.

### 8.1 환경 진입

```bash
cd /home/anna/projects/chexpert_poc
source .venv/bin/activate
export PYTHONPATH=/home/anna/projects/chexpert_poc
```

### 8.2 데이터셋 / 로더 점검

```bash
python scripts/check_dataset.py --config configs/base.yaml
python scripts/sanity_dataloader.py --config configs/base.yaml
```

### 8.3 baseline 학습

```bash
python scripts/train.py --config configs/base.yaml
```

### 8.4 평가

```bash
python scripts/eval.py --config configs/base.yaml
```

### 8.5 threshold tuning

```bash
python scripts/threshold_tune.py --config configs/base.yaml --criterion f1
```

비교 실험용 예시:

```bash
python scripts/threshold_tune.py --config configs/base.yaml --criterion balanced_accuracy
python scripts/threshold_tune.py --config configs/base.yaml --criterion recall
```

### 8.6 error analysis

```bash
python scripts/error_analysis.py --config configs/base.yaml
```

### 8.7 inference / Grad-CAM smoke test

```bash
python scripts/infer.py --config configs/base.yaml --input data/chexpert_small/raw/valid/patient64541/study1/view1_frontal.jpg
python scripts/gradcam_demo.py --config configs/base.yaml --input data/chexpert_small/raw/valid/patient64541/study1/view1_frontal.jpg
```

---

## 9. 산출물 위치

### 학습 산출물
- `outputs/train_runs/<run_id>/checkpoints/best.pt`
- `outputs/train_runs/<run_id>/checkpoints/last.pt`
- `outputs/train_runs/<run_id>/history.json`
- `outputs/train_runs/<run_id>/config_snapshot.json`
- `outputs/train_runs/<run_id>/model_info.json`
- `outputs/train_runs/<run_id>/pos_weight_stats.json`

### 평가 산출물
- `outputs/train_runs/<run_id>/eval/per_class_metrics.json`
- `outputs/train_runs/<run_id>/eval/summary_metrics.json`
- `outputs/train_runs/<run_id>/eval/eval_metadata.json`
- `outputs/train_runs/<run_id>/eval/study_predictions.csv`

### threshold tuning 산출물
- `outputs/train_runs/<run_id>/eval/threshold_tuning/infer_thresholds.json`
- `outputs/train_runs/<run_id>/eval/threshold_tuning/best_thresholds_by_class.json`
- `outputs/train_runs/<run_id>/eval/threshold_tuning/threshold_grid_metrics.csv`

### error analysis 산출물
- `outputs/train_runs/<run_id>/eval/error_analysis/`

### 실험 기록 문서
- `docs/experiments/README.md`
- `docs/experiments/baseline_01_run_20260321_125758.md`
- `docs/experiments/_template.md`

---

## 10. 실험 기록 운영 원칙

이 프로젝트는 논문/보고서까지 염두에 두고 있으므로, 실험은 반드시 **재현 가능하게 기록**해야 합니다.

기본 원칙:

- baseline / 비교실험 / 실패실험까지 기록
- 마지막 epoch가 아니라 **best checkpoint 기준**으로 기록
- config, dataset policy, run 목적, threshold, 결과 해석을 같이 기록
- 비교 실험에서는 한 번에 하나의 변수만 바꿈

실험이 끝나면 반드시 아래 중 하나를 남깁니다.

- 새 실험 노트 파일 추가
- 기존 실험 노트에 결과 및 해석 업데이트

실험노트 폴더는 여기입니다.

- [실험노트 모음](docs/experiments/README.md)

---

## 11. 다음 비교 실험 우선순위

현재 기준으로 다음 비교 실험 우선순위는 아래와 같습니다.

1. `U-Ignore` vs `U-Ones`
2. threshold criterion 비교 (`f1`, `balanced_accuracy`, `recall`)
3. `pos_weight_clip_max` 비교

중요 원칙:

- baseline 하나를 먼저 고정합니다.
- 그다음 변경점 하나씩만 비교합니다.
- 많은 ablation을 한 번에 벌리지 않습니다.

---

## 12. 서비스 이식 원칙

이 리포는 나중에 `capstone-cxr`의 `ai-service`로 추론 기능을 이식하기 위한 source repo이기도 합니다.

다만 이 리포 전체를 옮기지 않습니다. 우선순위는 아래와 같습니다.

- `chexpert_poc/inference/service.py`
- `chexpert_poc/inference/gradcam_service.py`
- `chexpert_poc/inference/predictor.py`
- `chexpert_poc/inference/postprocess.py`
- `chexpert_poc/inference/checkpoint.py`
- 참조용으로 `models/densenet.py`, `datasets/label_policy.py`, `explain/gradcam.py`

즉,

- `chexpert_poc` = 실험 리포
- `capstone-cxr` = 제품 리포

이 분리를 계속 유지합니다.

---

## 13. 현재 시점 한 줄 요약

현재 `chexpert_poc`는 **리팩토링 1차 완료 후 baseline 재확립 단계**에 있으며,  
이제부터의 핵심은 **구조를 더 바꾸는 것**이 아니라 **기준 baseline을 고정하고, 비교 실험을 통제된 방식으로 기록하는 것**입니다.
