from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def save_rows_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        actual_fieldnames = fieldnames or list(rows[0].keys())
    else:
        actual_fieldnames = fieldnames or []

    with open(path, "w", encoding="utf-8", newline="") as f:
        if not actual_fieldnames:
            f.write("")
            return

        writer = csv.DictWriter(f, fieldnames=actual_fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def save_threshold_grid_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    save_rows_csv(rows=rows, path=path)