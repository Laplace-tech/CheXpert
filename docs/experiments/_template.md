# 실험 기록 템플릿

---

## 1. 실험 이름
- 예: `Baseline 02 — DenseNet121 / U-Ones / Frontal-Only`

## 2. 실험 목적
- 이 실험에서 검증하려는 가설을 1~3문장으로 적는다.

## 3. 고정 조건
- Dataset:
- Labels:
- View mode:
- Uncertainty policy:
- Backbone:
- Input size:
- Batch size:
- Optimizer:
- Learning rate:
- Pos weight policy:
- Seed:

## 4. 변경 변수
- 이번 실험에서 baseline 대비 바뀐 점만 적는다.

## 5. 실행 명령어
```bash
python scripts/check_dataset.py --config <config>
python scripts/sanity_dataloader.py --config <config>
python scripts/train.py --config <config>
python scripts/eval.py --config <config>
python scripts/threshold_tune.py --config <config> --criterion f1
python scripts/error_analysis.py --config <config>
```

## 6. Run 정보
- Run ID:
- Best epoch:
- Best valid loss:
- Baseline / comparison checkpoint:

## 7. Eval 결과
- Mean AUROC:
- Mean AUPRC:

### 클래스별 성능
| Label | AUROC | AUPRC |
|---|---:|---:|
| Atelectasis |  |  |
| Cardiomegaly |  |  |
| Consolidation |  |  |
| Edema |  |  |
| Pleural Effusion |  |  |

## 8. Threshold tuning 결과
- Criterion:
- Search range:
- Recommended thresholds:

| Label | Threshold | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Atelectasis |  |  |  |  |
| Cardiomegaly |  |  |  |  |
| Consolidation |  |  |  |  |
| Edema |  |  |  |  |
| Pleural Effusion |  |  |  |  |

## 9. Error analysis 요약
| Label | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Atelectasis |  |  |  |  |  |
| Cardiomegaly |  |  |  |  |  |
| Consolidation |  |  |  |  |  |
| Edema |  |  |  |  |  |
| Pleural Effusion |  |  |  |  |  |

## 10. 해석
- 좋아진 점:
- 나빠진 점:
- 주의할 점:

## 11. 다음 실험 판단
- 다음에 무엇을 왜 해볼지 2~4줄로 적는다.
