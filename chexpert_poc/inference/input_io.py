from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def validate_input_image(path: str | Path) -> Path:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input must be a file: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")

    return path.resolve()


def collect_input_paths(input_path: str | Path, recursive: bool = False) -> list[Path]:
    # 입력이 파일이면 단일 이미지로 처리
    # 입력이 폴더면 이미지 파일들 전부 수집
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path.resolve()]

    if input_path.is_dir():
        if recursive:
            files = [p for p in input_path.rglob("*") if p.is_file()]
        else:
            files = [p for p in input_path.glob("*") if p.is_file()]

        image_files = [p.resolve() for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
        image_files = sorted(image_files)

        if not image_files:
            raise RuntimeError(f"No image files found under: {input_path}")

        return image_files

    raise ValueError(f"Unsupported input path: {input_path}")