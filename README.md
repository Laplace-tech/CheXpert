# CheXpert PoC Best-Run Snapshot

이 폴더는 대화 기준 **최고 성능 run (`run_20260307_181718`)을 만든 시점의 프로젝트 코드**를 재구성한 스냅샷이다.

주의:
- 사용자 로컬 디스크의 원본 파일을 byte-level로 복원한 것은 아니다.
- 대화 중 확정된 정책, 코드, 설정을 기준으로 **재구성한 실행 가능한 스냅샷**이다.
- 이후 진행된 실험(종횡비 유지 padding 전처리, sampler 실험, XRV 분기 등)은 포함하지 않았다.

## 재구성 기준
- 데이터 정책
  - NaN -> 0.0
  - -1 -> strategy
    - U-Ignore: label=0.0, loss_mask=0.0
    - U-Ones: label=1.0, loss_mask=1.0
- view_mode = frontal_only
- 첫 baseline = U-Ignore
- 최고 성능 run 요약
  - mean AUROC: 0.8785
  - mean AUPRC: 0.7387
  - run id: `run_20260307_181718`

## 주요 실행 순서
1. `python scripts/check_dataset.py --config configs/base.yaml --sample-size 256`
2. `python scripts/sanity_dataloader.py`
3. `python train.py`
4. `python eval.py`
5. `python infer.py --input <image_path>`
6. `python threshold_tune.py`
7. `python error_analysis.py`
# CheXpert
