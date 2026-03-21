# 실험노트 모음

이 폴더는 `chexpert_poc`의 **실험 기록 아카이브**입니다.  
논문/보고서/발표/인수인계까지 고려해서, 각 실험을 **재현 가능한 단위**로 남기는 것을 목표로 합니다.

---

## 운영 원칙

- 모든 실험은 가능하면 **1 run = 1 note** 형태로 기록합니다.
- baseline, 비교실험, 실패실험 모두 남깁니다.
- 마지막 epoch가 아니라 **best checkpoint 기준**으로 기록합니다.
- 아래 항목이 빠지지 않도록 합니다.
  - 실험 목적
  - 고정 조건
  - 변경 변수
  - run id
  - best epoch / best valid loss
  - eval 결과
  - threshold tuning 결과
  - error analysis 요약
  - 다음 실험 판단

---

## 문서 목록

### 기준 baseline
- [Baseline 01 — DenseNet121 / U-Ignore / Frontal-Only / Official 5 Tasks](baseline_01_run_20260321_125758.md)

### 템플릿
- [실험 기록 템플릿](./_template.md)

---

## 추천 파일명 규칙

```text
baseline_01_run_YYYYMMDD_HHMMSS.md
ablation_uones_run_YYYYMMDD_HHMMSS.md
ablation_threshold_balacc_run_YYYYMMDD_HHMMSS.md
ablation_posweight4_run_YYYYMMDD_HHMMSS.md
failure_note_run_YYYYMMDD_HHMMSS.md
```

---

## 추천 실험 순서

1. Baseline 확정
2. `U-Ignore` vs `U-Ones`
3. threshold criterion 비교
4. `pos_weight_clip_max` 비교
5. 필요 시 추가 실험

---

## 현재 기준선 요약

현재 1차 기준 baseline은 아래입니다.

- Run ID: `run_20260321_125758`
- Best epoch: `3`
- Mean AUROC: `0.8811`
- Mean AUPRC: `0.7387`
- Best checkpoint 기준 baseline으로 사용

자세한 내용은 아래 문서를 봅니다.

- [Baseline 01 실험 기록](baseline_01_run_20260321_125758.md)
