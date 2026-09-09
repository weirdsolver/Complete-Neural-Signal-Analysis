from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class HDF5NodeSummary:
    path: str
    kind: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    attrs: dict[str, Any] | None = None


def _json_safe(value: Any) -> Any:
    try:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.size <= 32:
                return value.tolist()
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def summarize_hdf5_dict(raw: dict[str, Any]) -> list[HDF5NodeSummary]:
    """Summarize a nested dictionary produced by HDF5Reader."""
    out: list[HDF5NodeSummary] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            attrs = obj.get("__attrs__") if isinstance(obj.get("__attrs__"), dict) else None
            out.append(HDF5NodeSummary(path=path or "/", kind="group", attrs=attrs))
            for key, value in obj.items():
                if str(key).startswith("__"):
                    continue
                child = f"{path}/{key}" if path else f"/{key}"
                walk(value, child)
            return
        try:
            arr = np.asarray(obj)
            out.append(HDF5NodeSummary(path=path or "/", kind="dataset", shape=tuple(int(x) for x in arr.shape), dtype=str(arr.dtype)))
        except Exception:
            out.append(HDF5NodeSummary(path=path or "/", kind="dataset", dtype=type(obj).__name__))

    walk(raw, "")
    return out


def write_file_tree_report(raw: dict[str, Any], output_dir: str | Path) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = out / "file_tree_report.json"
    data = [asdict(node) for node in summarize_hdf5_dict(raw)]
    report.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")
    return str(report)
