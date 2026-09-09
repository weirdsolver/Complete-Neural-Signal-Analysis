from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def stable_json_hash(value: Any) -> str:
    def default(o: Any) -> Any:
        try:
            import numpy as np
            import pandas as pd
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return {"shape": list(o.shape), "dtype": str(o.dtype)}
            if isinstance(o, pd.DataFrame):
                return {"shape": list(o.shape), "columns": list(o.columns)}
        except Exception:
            pass
        if isinstance(o, Path):
            return str(o)
        return str(o)
    payload = json.dumps(value, sort_keys=True, default=default)
    return sha256_text(payload)
