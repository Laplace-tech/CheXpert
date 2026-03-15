# chexpert_poc

CheXpert-small 기반 흉부 X-ray 멀티라벨 분류 PoC 저장소입니다.  
이 저장소는 **학습(train) / 평가(eval) / threshold tuning / 추론(infer) / Grad-CAM 시각화**까지 포함하는 **end-to-end baseline 실험 코드베이스**입니다.

현재 목적은 제품 서비스가 아니라, **CheXpert 공식 5개 병변 태스크에 대한 안정적인 baseline 구축과 분석**입니다.

---

## 1. 현재 범위

이 프로젝트는 아래 범위를 기준으로 동작합니다.

- 데이터셋: **CheXpert-small**
- 타겟 라벨: **공식 5개 competition task**
  - Atelectasis
  - Cardiomegaly
  - Consolidation
  - Edema
  - Pleural Effusion
- 모델: **DenseNet121 baseline**
- 입력 뷰: **frontal only**
- 기본 uncertain 정책: **U-Ignore**
- 비교 후보 uncertain 정책: **U-Ones**

주요 기능:

- 학습
- 검증셋 평가
- threshold tuning
- 단건/배치 추론
- Grad-CAM 생성

---

## 2. 저장소 구조

```text
chexpert_poc/
├── README.md
├── configs/
│   └── base.yaml
├── chexpert_poc/
│   ├── datasets/
│   │   ├── chexpert_dataset.py
│   │   └── labels.py
│   ├── explain/
│   │   └── gradcam.py
│   ├── metrics/
│   │   └── ...
│   ├── models/
│   │   └── densenet.py
│   └── utils/
│       ├── class_weights.py
│       ├── losses.py
│       └── train_utils.py
├── train.py
├── eval.py
├── threshold_tune.py
├── infer.py
├── error_analysis.py
├── gradcam_demo.py
├── scripts/
├── outputs/
└── checkpoints/
```

### 파일 역할 요약

- `configs/base.yaml`  
  실험 설정 파일입니다. 데이터 경로, 라벨, 이미지 크기, 학습 옵션 등을 정의합니다.

- `chexpert_poc/datasets/labels.py`  
  라벨 정책 정의 파일입니다. target labels, uncertainty 처리, frontal view 판별 로직이 들어 있습니다.

- `chexpert_poc/datasets/chexpert_dataset.py`  
  CheXpert CSV를 읽어 학습용 샘플(image, label, loss mask 등)로 변환합니다.

- `chexpert_poc/models/densenet.py`  
  DenseNet121 모델 생성 로직입니다.

- `chexpert_poc/utils/losses.py`  
  멀티라벨 BCE 기반 손실 계산 로직입니다.

- `chexpert_poc/utils/class_weights.py`  
  클래스 불균형 보정을 위한 `pos_weight` 계산 로직입니다.

- `train.py`  
  학습 진입점입니다.

- `eval.py`  
  검증셋 평가 및 메트릭 저장 스크립트입니다.

- `threshold_tune.py`  
  클래스별 threshold 탐색 및 저장 스크립트입니다.

- `infer.py`  
  단건/배치 추론 스크립트입니다.

- `gradcam_demo.py`, `chexpert_poc/explain/gradcam.py`  
  Grad-CAM 시각화 생성 코드입니다.

- `error_analysis.py`  
  예측 결과 기반 오분류/에러 분석용 스크립트입니다.

---

## 3. 환경 준비

권장 환경:

- Python 3.11+
- PyTorch with CUDA
- Linux/WSL 기반 실행

예시:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 환경 변수 예시

```bash
export PROJECT_ROOT=/home/anna/projects/chexpert_poc
export CHEXPERT_ROOT=/home/anna/datasets/cxr/chexpert_small/raw
export PYTHONPATH=/home/anna/projects/chexpert_poc
export OUTPUT_ROOT=/home/anna/projects/chexpert_poc/outputs
```

---

## 4. 데이터셋 준비

이 프로젝트는 **CheXpert-small raw dataset**을 기준으로 합니다.

예상 경로 예시:

```text
/home/anna/datasets/cxr/chexpert_small/raw/
├── train.csv
├── valid.csv
├── train/
└── valid/
```

데이터셋 루트 경로는 `configs/base.yaml`의 설정값으로 지정합니다.

---

## 5. 설정 파일

기본 설정은 `configs/base.yaml`에 있습니다.

핵심 설정 예시:

- `data.target_labels`: 사용할 병변 라벨 목록
- `data.image_size`: 입력 이미지 크기
- `data.uncertainty_strategy`: uncertain(-1) 처리 방식
- `data.view_mode`: frontal / all view 사용 정책
- `model.backbone`: 현재는 DenseNet121 baseline
- `train.use_pos_weight`: 클래스 불균형 보정 사용 여부
- `train.amp`: mixed precision 사용 여부

---

## 6. 학습

기본 학습 실행 예시:

```bash
python train.py --config configs/base.yaml
```

학습 결과는 보통 `outputs/train_runs/...` 아래에 저장됩니다.

대표 산출물:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `history.csv`
- `history.json`
- `model_info.json`
- `train_dataset_stats.json`
- `valid_dataset_stats.json`

---

## 7. 평가

검증셋 평가 예시:

```bash
python eval.py \
  --config configs/base.yaml \
  --checkpoint /path/to/best.pt
```

평가 결과는 보통 `outputs/train_runs/.../eval/` 아래에 저장됩니다.

대표 산출물:

- `summary_metrics.json`
- `per_class_metrics.json`
- `study_predictions.csv`

---

## 8. Threshold Tuning

클래스별 최적 threshold 탐색 예시:

```bash
python threshold_tune.py \
  --config configs/base.yaml \
  --checkpoint /path/to/best.pt
```

대표 산출물:

- `best_thresholds_by_class.json`
- `infer_thresholds.json`

`infer_thresholds.json`은 추론 시 사용되는 threshold 기준 파일입니다.

---

## 9. 추론

단일 이미지 추론 예시:

```bash
python infer.py \
  --config configs/base.yaml \
  --checkpoint /path/to/best.pt \
  --input /path/to/image.jpg \
  --thresholds 0.47,0.22,0.36,0.39,0.60
```

추론 결과는 보통 `outputs/infer_runs/...` 아래에 저장됩니다.

대표 산출물:

- `predictions.json`
- `predictions.csv`
- `infer_metadata.json`

---

## 10. Grad-CAM

Grad-CAM 실행 예시:

```bash
python gradcam_demo.py \
  --config configs/base.yaml \
  --checkpoint /path/to/best.pt \
  --input /path/to/image.jpg
```

결과는 `outputs/gradcam_runs/...` 아래에 저장됩니다.

대표 산출물:

- `original.png`
- `heatmap_<label>.png`
- `overlay_<label>.png`
- `panel_<label>.png`
- `gradcam_result.json`

---

## 11. 라벨 정책 / Uncertainty 정책

### CheXpert raw label 의미

- `1` : positive
- `0` : negative
- `-1`: uncertain
- `NaN`: missing

### 현재 프로젝트 기본 처리

- `NaN -> 0.0`
- 기본 uncertainty 정책: `U-Ignore`

### U-Ignore

- `-1 -> label=0.0, loss_mask=0.0`
- uncertain 샘플은 해당 클래스의 loss 계산에서 제외됩니다.

의미:

- 애매한 라벨을 정답으로 강하게 믿지 않음
- baseline을 더 안정적으로 가져가기 쉬움
- 대신 uncertain 샘플 일부를 학습에서 버리게 됨

### U-Ones

- `-1 -> label=1.0, loss_mask=1.0`
- uncertain 샘플을 양성으로 간주하여 학습에 포함합니다.

의미:

- uncertain 샘플도 학습 신호로 활용
- recall(민감도)에 유리할 수 있음
- 대신 라벨 노이즈 증가 위험이 있어 precision/특이도 저하 가능성 존재

### 왜 기본값이 U-Ignore인가

초기 baseline에서는 noisy supervision을 줄이는 것이 더 안전하기 때문입니다.  
이후 비교 실험으로 `U-Ones`를 적용해 pathology별 차이를 볼 수 있습니다.

---

## 12. View 정책

현재 기본 설정은 **frontal only**입니다.

즉:

- frontal 이미지만 사용
- lateral 이미지는 제외

이 선택은 입력 분포를 단순화하고 baseline 안정성을 높이기 위한 것입니다.

---

## 13. 주요 평가 지표

### Accuracy
전체 예측 중 맞춘 비율입니다.

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

주의할 점은, 의료 데이터처럼 클래스 불균형이 심한 경우 accuracy만 높아도 좋은 모델이라고 볼 수 없다는 점입니다.

### Recall = Sensitivity = 민감도
실제 양성 중에서 양성으로 맞춘 비율입니다.

```text
recall = TP / (TP + FN)
```

의미:

- 실제 환자를 얼마나 놓치지 않았는가
- 높을수록 FN(False Negative)이 적음

### Specificity = 특이도
실제 음성 중에서 음성으로 맞춘 비율입니다.

```text
specificity = TN / (TN + FP)
```

의미:

- 정상인을 괜히 병이라고 오진하지 않는가
- 높을수록 FP(False Positive)가 적음

### Precision = 정밀도
양성이라고 예측한 것 중 실제 양성 비율입니다.

```text
precision = TP / (TP + FP)
```

의미:

- 양성 판정이 얼마나 믿을 만한가

### F1-score
Precision과 Recall의 조화평균입니다.

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

의미:

- precision과 recall을 균형 있게 보고 싶을 때 사용

### 해석 요약

- recall/민감도 높음 → 실제 병변을 덜 놓침
- specificity/특이도 높음 → 정상인을 덜 오진함
- precision 높음 → 양성 판정의 신뢰도가 높음
- accuracy는 참고 지표일 뿐, 단독 해석은 위험함

---

## 14. Threshold와 Trade-off

모델의 sigmoid 출력에 threshold를 적용하면 최종 양성/음성 판단이 결정됩니다.

일반적으로:

- threshold를 낮추면  
  양성 판정이 많아져 recall은 올라갈 수 있지만 FP도 늘어 precision/특이도는 떨어질 수 있습니다.

- threshold를 높이면  
  양성 판정이 줄어 precision/특이도는 좋아질 수 있지만 실제 양성을 놓쳐 recall이 떨어질 수 있습니다.

따라서 클래스별 threshold tuning은 단순 후처리가 아니라 중요한 성능 조정 단계입니다.

---

## 15. 출력 폴더 구조

대표적인 출력 구조는 아래와 같습니다.

```text
outputs/
├── train_runs/
│   └── run_xxx/
│       ├── checkpoints/
│       ├── history.csv
│       ├── history.json
│       ├── model_info.json
│       └── eval/
├── infer_runs/
│   └── infer_xxx/
└── gradcam_runs/
    └── gradcam_xxx/
```

---

## 16. 현재 한계

- backbone은 현재 DenseNet121 baseline 중심입니다.
- uncertainty 정책은 `U-Ignore`, `U-Ones`만 구현되어 있습니다.
- 이 저장소는 서비스용 코드가 아니라 실험용 PoC 코드베이스입니다.
- 학습/평가/분석 로직과 추론 로직이 아직 완전히 분리된 서비스 구조는 아닙니다.

---

## 17. 앞으로의 사용 방향

이 저장소는 계속 **학습/평가/분석 실험실**로 유지합니다.  
서비스 이식 시에는 이 저장소를 통째로 복사하지 않고, 아래 최소 추론 코어만 분리하는 방향을 권장합니다.

- model loader
- single image preprocess
- inference runner
- threshold apply
- Grad-CAM generator

즉, `chexpert_poc`는 실험용 baseline의 기준점으로 유지하고, 서비스 코드는 별도 레이어에서 가져가는 것이 맞습니다.
