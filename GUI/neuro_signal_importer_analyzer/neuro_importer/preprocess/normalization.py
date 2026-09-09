from __future__ import annotations

import numpy as np


def zscore(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sigma = np.nanstd(x, axis=0, keepdims=True)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return (x - mu) / sigma


def robust_scale(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    med = np.nanmedian(x, axis=0, keepdims=True)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True)
    iqr = q75 - q25
    iqr = np.where(iqr < eps, 1.0, iqr)
    return (x - med) / iqr


def minmax_scale(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    mn = np.nanmin(x, axis=0, keepdims=True)
    mx = np.nanmax(x, axis=0, keepdims=True)
    rng = np.where((mx - mn) < eps, 1.0, mx - mn)
    return (x - mn) / rng


def normalize_signal(signal: np.ndarray, method: str | None) -> np.ndarray:
    if method is None or str(method).lower() in {"", "none", "false"}:
        return np.asarray(signal)
    method_l = str(method).lower().strip()
    if method_l in {"z", "zscore", "z-score", "standard"}:
        return zscore(signal)
    if method_l in {"robust", "iqr"}:
        return robust_scale(signal)
    if method_l in {"minmax", "min-max"}:
        return minmax_scale(signal)
    raise ValueError(f"Unknown normalization method: {method}")
