from __future__ import annotations

from pathlib import Path   # 파일 경로 다루기
from typing import Callable, Final, Optional  

import pandas as pd                  # CSV 읽기
import torch                         # 텐서 만들기
from PIL import Image                # 이미지 열기
from torch.utils.data import Dataset # Pytorch 데이터셋 클래스 상속용
from torchvision import transforms   # 이미지 전처리용

# [import: labels.py]
# - 여기서 라벨 정책을 불러온다
from chexpert_poc.datasets.label_policy import (
    CHEXPERT_5_LABELS,              # 이 5개를 기본값으로 사용
    encode_chexpert_label,          # raw CSV 라벨(NaN/0/1/-1) => (label_value, loss_mask_value)로 변환
    is_frontal_view,                # frontal_only일 때 row를 남길지 버릴지 판정
    validate_uncertainty_strategy,  # U-Ignore / U-Ones 설정값 검증
)

# CSV의 Path 컬럼 값 앞에 붙을 수 있는 prefix 목록
# : resolve_image_path()에서 실제 로컬 파일 경로를 찾을 때 사용
# ex)
# - CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
# - CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg
KNOWN_CSV_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "CheXpert-v1.0-small/",
    "CheXpert-v1.0/",
    "./",
)

# 현재 프로젝트에서 허용하는 view 필터 정책
VALID_VIEW_MODES: Final[set[str]] = {"frontal_only", "all"}


# [바깥에서 제일 먼저 호출되는 wrapper]
# - train.py / eval.py / threshold_tune.py 등에서 dataset 만들 때의 진입점
def build_chexpert_dataset(
    config: dict,
    split: str,
    transform: Optional[Callable] = None,
) -> CheXpertDataset:
    
    # raw_root: CheXpert-small 원본 데이터셋 루트 경로
    # csv_path: CSV 파일이 있는 경로:
    #   ex:
    #   - /home/anna/datasets/cxr/chexpert_small/raw/train.csv 
    #   - /home/anna/datasets/cxr/chexpert_small/raw/valid.csv
    raw_root = Path(config["paths"]["chexpert_root"])
    csv_path = raw_root / f"{split}.csv"

    # [base.yaml]
    # - paths.chexpert_root: /home/anna/datasets/cxr/chexpert_small/raw
    # - data.image_size: 320
    # - data.uncertainty_strategy: U-Ignore
    # - data.view_mode: frontal_only
    # - data.target_labels: [Atelectasis, Cardiomegaly ....]
    # - data.path_column: Path
    return CheXpertDataset(
        csv_path=csv_path, 
        raw_root=raw_root,
        split=split,
        
        image_size=int(config["data"]["image_size"]),
        uncertainty_strategy=config["data"]["uncertainty_strategy"], 
        view_mode=config["data"].get("view_mode", "frontal_only"),
        transform=transform,
        target_labels=config["data"].get("target_labels", list(CHEXPERT_5_LABELS)),
        path_column=config["data"].get("path_column", "Path"),
    )

class CheXpertDataset(Dataset):
    """
    CheXpert CSV를 학습 가능한 PyTorch Dataset으로 변환.
    """

    def __init__(
        self,
        csv_path: str | Path,
        raw_root: str | Path,
        split: str,
        image_size: int = 320,
        uncertainty_strategy: str = "U-Ignore",
        view_mode: str = "frontal_only",
        transform: Optional[Callable] = None,
        target_labels: Optional[list[str]] = None,
        path_column: str = "Path",
    ) -> None:
        super().__init__()

        # 생성자 인자를 내부 상태로 저장
        self.csv_path = Path(csv_path) # raw_root: "train.csv" 혹은 "valid.csv"
        self.raw_root = Path(raw_root) # base.yml의 paths.chexpert_root
        self.split = str(split)        # "train" 또는 "valid" 문자열
        self.image_size = int(image_size)
        
        
        self.uncertainty_strategy = validate_uncertainty_strategy(uncertainty_strategy)
        self.view_mode = self._validate_view_mode(view_mode)
        self.path_column = str(path_column)
        self.target_labels = (
            list(target_labels) if target_labels is not None else list(CHEXPERT_5_LABELS)
        )
        self.transform = transform or build_image_transform(image_size=self.image_size)

        # 기본 입력 검증
        if not self.csv_path.exists():
            raise FileNotFoundError(f"csv_path not found: {self.csv_path}")
        if not self.raw_root.exists():
            raise FileNotFoundError(f"raw_root not found: {self.raw_root}")
        if not self.target_labels:
            raise ValueError("target_labels must not be empty")

        df = pd.read_csv(self.csv_path)

        required_columns = [self.path_column, *self.target_labels]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # CSV를 실제 사용 가능한 row만 남긴 dataframe으로 정리
        self.df, self.dataset_stats = self._prepare_dataframe(df)

        if len(self.df) == 0:
            raise RuntimeError(
                f"No usable rows after filtering. "
                f"csv_path={self.csv_path}, split={self.split}, view_mode={self.view_mode}"
            )

        # label / loss_mask는 매번 계산하지 않고 초기화 시점에 미리 계산
        self.label_matrix, self.loss_mask_matrix = self._precompute_targets(self.df)

    @staticmethod
    def _validate_view_mode(view_mode: str) -> str:
        # 허용된 view 정책인지 검증
        if view_mode not in VALID_VIEW_MODES:
            raise ValueError(
                f"Invalid view_mode: {view_mode}. "
                f"Expected one of {sorted(VALID_VIEW_MODES)}"
            )
        return view_mode

    def _prepare_dataframe(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, int | str]]:
        """
        CSV를 현재 프로젝트 정책에 맞는 usable dataframe으로 정리.
        """
        work_df = df.copy()
        raw_rows = len(work_df)

        # 1) view 필터링
        if self.view_mode == "frontal_only":
            work_df = work_df[work_df[self.path_column].apply(is_frontal_view)].copy()
        elif self.view_mode == "all":
            pass
        else:
            raise ValueError(
                f"Invalid view_mode: {self.view_mode}. "
                f"Expected one of {sorted(VALID_VIEW_MODES)}"
            )

        after_view_filter_rows = len(work_df)

        # 2) 실제 파일 경로 resolve
        work_df["resolved_path"] = work_df[self.path_column].apply(
            lambda x: resolve_image_path(self.raw_root, x)
        )
        work_df = work_df[work_df["resolved_path"].notna()].copy()
        after_path_resolve_rows = len(work_df)

        # 3) study 단위 식별자 생성
        work_df["study_id"] = work_df[self.path_column].apply(self._extract_study_id)
        work_df.reset_index(drop=True, inplace=True)

        # 디버깅 / sanity check용 통계
        stats = {
            "csv_path": str(self.csv_path),
            "split": self.split,
            "path_column": self.path_column,
            "view_mode": self.view_mode,
            "raw_rows": int(raw_rows),
            "after_view_filter_rows": int(after_view_filter_rows),
            "after_path_resolve_rows": int(after_path_resolve_rows),
            "dropped_by_view_filter": int(raw_rows - after_view_filter_rows),
            "dropped_by_unresolved_path": int(
                after_view_filter_rows - after_path_resolve_rows
            ),
            "usable_rows": int(len(work_df)),
        }

        return work_df, stats

    def _precompute_targets(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """
        label / loss_mask를 초기화 시점에 미리 계산.
        __getitem__에서 반복 계산하지 않게 하기 위한 처리.
        """
        all_labels: list[list[float]] = []
        all_loss_masks: list[list[float]] = []

        for _, row in df.iterrows():
            labels: list[float] = []
            loss_masks: list[float] = []

            for label_name in self.target_labels:
                raw_value = row[label_name]
                label_value, loss_mask_value = encode_chexpert_label(
                    raw_value,
                    strategy=self.uncertainty_strategy,
                )
                labels.append(label_value)
                loss_masks.append(loss_mask_value)

            all_labels.append(labels)
            all_loss_masks.append(loss_masks)

        label_tensor = torch.tensor(all_labels, dtype=torch.float32)
        loss_mask_tensor = torch.tensor(all_loss_masks, dtype=torch.float32)
        return label_tensor, loss_mask_tensor

    @staticmethod
    def _extract_study_id(path_value: object) -> str:
        # path 문자열에서 patient/study 단위 식별자 추출
        if not isinstance(path_value, str):
            return "unknown"

        parts = path_value.replace("\\", "/").split("/")

        patient_idx = None
        study_idx = None

        for i, part in enumerate(parts):
            if part.startswith("patient"):
                patient_idx = i
                break

        if patient_idx is not None and patient_idx + 1 < len(parts):
            if parts[patient_idx + 1].startswith("study"):
                study_idx = patient_idx + 1

        if patient_idx is not None and study_idx is not None:
            return f"{parts[patient_idx]}/{parts[study_idx]}"

        return Path(path_value).stem

    def __len__(self) -> int:
        # 사용 가능한 샘플 수 반환
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, object]:
        # index번째 샘플을 실제 학습 입력 형태로 반환
        row = self.df.iloc[index]

        img_path = Path(row["resolved_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Resolved image path not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label_tensor = self.label_matrix[index]
        loss_mask_tensor = self.loss_mask_matrix[index]

        return {
            "image": image,
            "label": label_tensor,
            "loss_mask": loss_mask_tensor,
            "path": row[self.path_column],
            "resolved_path": str(img_path),
            "study_id": row["study_id"],
        }


def resolve_image_path(raw_root: Path, csv_path_value: object) -> Optional[Path]:
    """
    CSV의 Path 값을 실제 로컬 이미지 경로로 해석.
    경로 표현이 제각각일 수 있어서 여러 후보를 순서대로 시도한다.
    """
    if pd.isna(csv_path_value):
        return None

    path_str = str(csv_path_value).strip().replace("\\", "/")
    if not path_str:
        return None

    original = Path(path_str)
    candidates: list[Path] = []

    if original.is_absolute():
        candidates.append(original)  # 절대경로면 그대로 시도

    candidates.append(raw_root / path_str)  # raw_root 기준

    stripped = path_str
    for prefix in KNOWN_CSV_PATH_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    stripped = stripped.lstrip("/")

    candidates.append(raw_root / stripped)         # prefix 제거 후 raw_root 기준
    candidates.append(raw_root.parent / path_str)  # raw_root.parent 기준 fallback
    candidates.append(raw_root.parent / stripped)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def build_image_transform(image_size: int) -> Callable:
    """
    기본 입력 전처리.
    현재는 train/valid augmentation 분리 없이 공통 transform 사용.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)), # 320x320 같은 고정 입력 크기 맞춤
            transforms.ToTensor(),                       # PIL Image -> torch.Tensor (C,H,W)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),  # ImageNet pretrained DenseNet 기준 normalize
        ]
    )
