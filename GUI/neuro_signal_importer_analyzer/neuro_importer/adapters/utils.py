from __future__ import annotations

from typing import Any

import numpy as np


def unwrap_singleton(x: Any) -> Any:
    """Repeatedly unwrap list/tuple/array objects with exactly one element."""
    while isinstance(x, (list, tuple, np.ndarray)) and np.asarray(x, dtype=object).size == 1:
        if isinstance(x, np.ndarray):
            x = x.reshape(-1)[0]
        else:
            x = x[0]
    return x


def flatten(x: Any) -> list[Any]:
    """Flatten nested lists/tuples/arrays while preserving scalar leaves."""
    out: list[Any] = []

    def _walk(v: Any) -> None:
        if isinstance(v, dict):
            # Dicts are semantic objects, not scalar leaves. Flatten values only
            # when explicitly requested by callers using this helper.
            out.append(v)
            return
        if isinstance(v, np.ndarray):
            for item in v.reshape(-1):
                _walk(item)
            return
        if isinstance(v, (list, tuple)):
            for item in v:
                _walk(item)
            return
        out.append(v)

    _walk(x)
    return out


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Access dict/object/structured-array fields with a single helper."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        try:
            return obj[name]
        except Exception:
            return default
    try:
        return obj[name]
    except Exception:
        return default


def as_scalar(x: Any, default: Any = None) -> Any:
    if x is None:
        return default
    try:
        arr = np.asarray(x, dtype=object)
        if arr.size == 0:
            return default
        value = unwrap_singleton(arr.reshape(-1)[0])
        if value is None:
            return default
        return value
    except Exception:
        return x


def safe_float(x: Any, default: float | None = None) -> float | None:
    value = as_scalar(x, default=None)
    if value is None:
        return default
    try:
        val = float(value)
        if np.isnan(val):
            return default
        return val
    except Exception:
        return default


def as_vector(x: Any) -> np.ndarray:
    if x is None:
        return np.asarray([])
    vals = flatten(x)
    if len(vals) == 1 and isinstance(vals[0], dict):
        return np.asarray([])
    return np.asarray(vals, dtype=object).reshape(-1)


def stringify_label(x: Any) -> str:
    value = unwrap_singleton(x)
    if isinstance(value, bytes):
        try:
            value = value.decode()
        except Exception:
            value = str(value)
    return str(value).strip()
