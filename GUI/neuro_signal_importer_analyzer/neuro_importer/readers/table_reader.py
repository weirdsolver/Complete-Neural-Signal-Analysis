from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class TableReader:
    """Read CSV/TSV/Excel files into a wrapper dict containing a DataFrame."""

    def read(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(p)
        elif suffix == ".tsv":
            df = pd.read_csv(p, sep="\t")
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        else:
            raise ValueError(f"Unsupported table file suffix: {suffix}")
        return {"dataframe": df, "source_path": str(p), "file_type": "table"}
