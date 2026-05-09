# scripts/check_dataset.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chexpert_poc.common.config import get_split_csv_path, load_config
from chexpert_poc.datasets.path_utils import resolve_image_path


SUPPORTED_SPLITS = ("train", "valid", "test")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    raw_root = Path(config["paths"]["chexpert_root"])
    path_column = str(config["data"]["path_column"])
    target_labels = list(config["data"]["target_labels"])
    requested_splits = list(args.splits)

    print("=" * 80)
    print("CheXpert dataset check")
    print("=" * 80)
    print(f"configured dataset root: {raw_root}")
    print(f"exists: {raw_root.exists()}")

    if not raw_root.exists():
        raise FileNotFoundError(f"configured dataset root not found: {raw_root}")

    _print_root_structure(config, raw_root, requested_splits)

    for split_name in requested_splits:
        csv_path = get_split_csv_path(config, split_name)
        if not csv_path.exists():
            raise FileNotFoundError(f"{split_name} csv not found: {csv_path}")

        split_dir = raw_root / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_name} dir not found: {split_dir}")

        _inspect_split(
            split_name=split_name,
            csv_path=csv_path,
            raw_root=raw_root,
            target_labels=target_labels,
            path_column=path_column,
            sample_size=args.sample_size,
            full_path_check=args.full_path_check,
        )

    print("\n" + "=" * 80)
    print("dataset check finished")
    print("=" * 80)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--full-path-check", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=SUPPORTED_SPLITS,
        help="Which dataset splits to inspect. Default: train valid",
    )
    return parser


def _summarize_label(df: pd.DataFrame, label: str) -> dict[str, int]:
    """
    하나의 라벨 컬럼에 대해 값 분포를 요약한다.

    반환값:
    - pos_1: 양성(1)
    - neg_0: 음성(0)
    - unc_-1: 불확실(-1)
    - nan: 결측값
    """
    s = pd.to_numeric(df[label], errors="coerce")
    return {
        "pos_1": int((s == 1).sum()),
        "neg_0": int((s == 0).sum()),
        "unc_-1": int((s == -1).sum()),
        "nan": int(s.isna().sum()),
    }


def _inspect_split(
    split_name: str,
    csv_path: Path,
    raw_root: Path,
    target_labels: list[str],
    path_column: str,
    sample_size: int,
    full_path_check: bool,
) -> None:
    """
    단일 split(train / valid / test)에 대해 다음을 점검한다.

    1) CSV 로드 가능 여부
    2) 필수 컬럼 존재 여부
    3) 라벨 분포
    4) Path -> 실제 이미지 파일 resolve 가능 여부
    5) 미리보기 출력
    """
    print("\n" + "=" * 80)
    print(f"[{split_name}] CSV: {csv_path}")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    print(f"rows: {len(df):,}")
    print(f"columns: {len(df.columns)}")

    required_columns = [path_column, *target_labels]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] missing columns: {missing_cols}")
        return

    print(f"[OK] required columns present: {required_columns}")

    print("\n[label distribution]")
    for label in target_labels:
        stats = _summarize_label(df, label)
        print(
            f"- {label:17s} | "
            f"1: {stats['pos_1']:7d} | "
            f"0: {stats['neg_0']:7d} | "
            f"-1: {stats['unc_-1']:7d} | "
            f"NaN: {stats['nan']:7d}"
        )

    if full_path_check or len(df) <= sample_size:
        check_df = df.copy()
        check_mode = "FULL"
    else:
        check_df = df.sample(n=sample_size, random_state=42).copy()
        check_mode = f"SAMPLE({sample_size})"

    print(f"\n[path check: {check_mode}]")

    resolved = 0
    unresolved_examples: list[str] = []
    resolved_examples: list[tuple[str, str]] = []

    for _, row in check_df.iterrows():
        raw_path_value = row[path_column]
        resolved_path = resolve_image_path(raw_root, raw_path_value)

        if resolved_path is not None and resolved_path.exists():
            resolved += 1
            if len(resolved_examples) < 5:
                resolved_examples.append((str(raw_path_value), str(resolved_path)))
        else:
            if len(unresolved_examples) < 5:
                unresolved_examples.append(str(raw_path_value))

    total_checked = len(check_df)
    ratio = resolved / total_checked if total_checked > 0 else 0.0
    print(f"resolved paths: {resolved}/{total_checked} ({ratio:.2%})")

    if resolved_examples:
        print("\n[resolved path examples]")
        for raw_p, res_p in resolved_examples:
            print(f"- CSV Path: {raw_p}")
            print(f"  Real Path: {res_p}")

    if unresolved_examples:
        print("\n[unresolved examples]")
        for p in unresolved_examples:
            print(f"- {p}")
    else:
        print("\n[OK] no unresolved paths in checked set")

    print("\n[first 3 rows preview]")
    preview_cols = [path_column, *target_labels]
    print(df[preview_cols].head(3).to_string(index=False))


def _print_root_structure(
    config: dict,
    raw_root: Path,
    requested_splits: list[str],
) -> None:
    """
    현재 config 기준으로 요청된 split들의 CSV/디렉토리 존재 여부를 출력한다.
    """
    print("\n[root structure]")
    for split_name in requested_splits:
        csv_path = get_split_csv_path(config, split_name)
        split_dir = raw_root / split_name
        print(f"{split_name}.csv exists: {csv_path.exists()} ({csv_path.name})")
        print(f"{split_name} dir exists : {split_dir.exists()} ({split_dir})")


if __name__ == "__main__":
    main()