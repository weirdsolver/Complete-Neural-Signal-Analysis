from __future__ import annotations

import re
from typing import Any

import numpy as np


def walk_dict(raw: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if str(key).startswith("__"):
                continue
            path = f"{prefix}/{key}" if prefix else f"/{key}"
            out.append((path, value))
            out.extend(walk_dict(value, path))
    return out


def collect_attrs(raw: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if isinstance(raw, dict):
        if isinstance(raw.get("__attrs__"), dict):
            attrs.update({str(k): v for k, v in raw["__attrs__"].items()})
        for key, value in raw.items():
            if str(key).startswith("__"):
                continue
            child = collect_attrs(value)
            for ck, cv in child.items():
                attrs.setdefault(str(ck), cv)
    return attrs


def get_by_slash_path(raw: Any, path: str) -> Any:
    parts = [p for p in path.strip("/").split("/") if p]
    obj = raw
    for part in parts:
        if not isinstance(obj, dict) or part not in obj:
            raise KeyError(path)
        obj = obj[part]
    return obj


def looks_like_finalspark_live_mea(raw: dict[str, Any]) -> bool:
    if not isinstance(raw, dict):
        return False
    timestamp_like = 0
    electrode_like = 0
    for key, group in raw.items():
        if str(key).startswith("__") or not isinstance(group, dict):
            continue
        if re.search(r"(timestamp|time|chunk|frame)[_\-]?\d+", str(key), flags=re.I) or str(key).replace('.', '', 1).isdigit():
            ekeys = [k for k in group.keys() if re.search(r"(electrode|channel|ch)[_\-]?\d+", str(k), flags=re.I)]
            if len(ekeys) >= 4:
                timestamp_like += 1
                electrode_like = max(electrode_like, len(ekeys))
    return timestamp_like >= 1 and electrode_like >= 4


def looks_like_cortical_labs_cl1(raw: dict[str, Any]) -> bool:
    if not isinstance(raw, dict):
        return False
    paths = [path.lower() for path, _ in walk_dict(raw)]
    attrs = {k.lower(): v for k, v in collect_attrs(raw).items()}
    has_samples = any(path.endswith("/samples") or path == "/samples" for path in paths)
    has_cl_attrs = any(k in attrs for k in ("system_id", "channel_count", "sampling_frequency", "file_format.version"))
    return has_samples and has_cl_attrs


def looks_like_nwb(raw: dict[str, Any]) -> bool:
    if not isinstance(raw, dict):
        return False
    attrs = {k.lower(): v for k, v in collect_attrs(raw).items()}
    root_keys = {str(k).lower() for k in raw.keys()}
    return "nwb_version" in attrs or "nwb_version" in root_keys or "acquisition" in root_keys


def numeric_2d_datasets(raw: dict[str, Any], *, min_size: int = 20) -> list[tuple[str, Any, tuple[int, int]]]:
    out: list[tuple[str, Any, tuple[int, int]]] = []
    for path, value in walk_dict(raw):
        try:
            arr = np.asarray(value)
            arr = np.squeeze(arr)
        except Exception:
            continue
        if arr.ndim != 2 or arr.size < min_size:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        rows, cols = arr.shape
        if min(rows, cols) < 2:
            continue
        out.append((path, value, (int(rows), int(cols))))
    return out
