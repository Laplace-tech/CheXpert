from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chexpert_poc.common.config import load_config


def resolve_image_path(raw_root: Path, csv_path_value: str) -> Optional[Path]:
    """
    CSV에 들어있는 Path 문자열을 로컬 환경에서 실제 존재하는 이미지 파일 경로로 변환한다.
    """
    if pd.isna(csv_path_value):
        return None

    p = str(csv_path_value).strip().replace("\\", "/")
    if not p:
        return None

    original = Path(p)
    candidates: list[Path] = []

    if original.is_absolute():
        candidates.append(original)

    candidates.append(raw_root / p)

    stripped = p
    for prefix in ("CheXpert-v1.0-small/", "CheXpert-v1.0/", "./"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
    stripped = stripped.lstrip("/")

    candidates.append(raw_root / stripped)
    candidates.append(raw_root.parent / p)
    candidates.append(raw_root.parent / stripped)
    candidates.append(raw_root / Path(p).name)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    return None


def summarize_label(df: pd.DataFrame, label: str) -> dict[str, int]:
    """
    하나의 라벨에 대해 값 분포를 카운팅한다. (1 / 0 / -1 / NaN)
    """
    s = pd.to_numeric(df[label], errors="coerce")
    return {
        "pos_1": int((s == 1).sum()),
        "neg_0": int((s == 0).sum()),
        "unc_-1": int((s == -1).sum()),
        "nan": int(s.isna().sum()),
    }


def inspect_split(
    split_name: str,
    csv_path: Path,
    raw_root: Path,
    target_labels: list[str],
    path_column: str,
    sample_size: int,
    full_path_check: bool,
) -> None:
    print("\n" + "=" * 80)
    print(f"[{split_name}] CSV: {csv_path}")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    print(f"rows: {len(df):,}")
    print(f"columns: {len(df.columns)}")

    missing_cols = [c for c in [path_column, *target_labels] if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] missing columns: {missing_cols}")
        return

    print(f"[OK] required columns present: {[path_column, *target_labels]}")

    print("\n[label distribution]")
    for label in target_labels:
        stats = summarize_label(df, label)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--full-path-check", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_root = Path(config["paths"]["chexpert_root"])
    path_column = str(config["data"]["path_column"])
    target_labels = list(config["data"]["target_labels"])

    print("=" * 80)
    print("CheXpert dataset check")
    print("=" * 80)
    print(f"CHEXPERT_ROOT: {raw_root}")
    print(f"exists: {raw_root.exists()}")

    train_csv = raw_root / "train.csv"
    valid_csv = raw_root / "valid.csv"
    train_dir = raw_root / "train"
    valid_dir = raw_root / "valid"

    print("\n[root structure]")
    print(f"train.csv exists: {train_csv.exists()}")
    print(f"valid.csv exists: {valid_csv.exists()}")
    print(f"train dir exists : {train_dir.exists()}")
    print(f"valid dir exists : {valid_dir.exists()}")

    if not raw_root.exists():
        raise FileNotFoundError(f"CHEXPERT_ROOT not found: {raw_root}")
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not valid_csv.exists():
        raise FileNotFoundError(f"valid.csv not found: {valid_csv}")

    inspect_split(
        split_name="train",
        csv_path=train_csv,
        raw_root=raw_root,
        target_labels=target_labels,
        path_column=path_column,
        sample_size=args.sample_size,
        full_path_check=args.full_path_check,
    )

    inspect_split(
        split_name="valid",
        csv_path=valid_csv,
        raw_root=raw_root,
        target_labels=target_labels,
        path_column=path_column,
        sample_size=args.sample_size,
        full_path_check=args.full_path_check,
    )

    print("\n" + "=" * 80)
    print("dataset check finished")
    print("=" * 80)


if __name__ == "__main__":
    main()