from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def scan_dataset(root: str | Path, config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or {}
    scan_cfg = cfg.get("scan", cfg)
    root_path = Path(root).expanduser().resolve()
    include_ext = {str(x).lower() for x in scan_cfg.get("include_extensions", [])}
    exclude_dirs = set(str(x) for x in scan_cfg.get("exclude_dir_names", []))
    max_files = scan_cfg.get("max_files")

    rows: list[dict[str, Any]] = []
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in exclude_dirs for part in path.relative_to(root_path).parts[:-1]):
            continue
        suffix = path.suffix.lower()
        if include_ext and suffix not in include_ext:
            continue
        rel = path.relative_to(root_path)
        rows.append({
            "source_file": str(path),
            "relative_path": str(rel),
            "filename": path.name,
            "extension": suffix,
            "size_bytes": int(path.stat().st_size),
        })
        if max_files is not None and len(rows) >= int(max_files):
            break
    return pd.DataFrame(rows)
