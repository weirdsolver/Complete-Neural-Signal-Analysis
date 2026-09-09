from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from neuro_importer.core.recording import Recording


@dataclass
class WindowedSignal:
    X: np.ndarray
    index: pd.DataFrame


def resolve_window_params(recording: Recording, config: dict[str, Any]) -> tuple[int, int]:
    fs = recording.sampling_rate
    window_samples = config.get("window_samples")
    step_samples = config.get("step_samples")

    if window_samples is None:
        seconds = config.get("window_seconds")
        if seconds is None:
            raise ValueError("Windowing requires window_samples or window_seconds")
        if fs is None:
            raise ValueError("window_seconds requires recording.sampling_rate; use window_samples instead")
        window_samples = int(round(float(seconds) * float(fs)))

    if step_samples is None:
        step_seconds = config.get("step_seconds")
        if step_seconds is not None:
            if fs is None:
                raise ValueError("step_seconds requires recording.sampling_rate; use step_samples instead")
            step_samples = int(round(float(step_seconds) * float(fs)))
        else:
            step_samples = int(window_samples)

    window_samples = int(window_samples)
    step_samples = int(step_samples)
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if step_samples <= 0:
        raise ValueError("step_samples must be positive")
    return window_samples, step_samples


def build_windows(recording: Recording, config: dict[str, Any] | None = None) -> WindowedSignal:
    """Build fixed-length windows from a continuous Recording.

    Output shape is windows × samples × channels.
    """
    cfg = config or {}
    window_samples, step_samples = resolve_window_params(recording, cfg)
    drop_last = bool(cfg.get("drop_last", True))
    max_windows = cfg.get("max_windows")
    astype = cfg.get("astype") or None

    starts: list[int] = []
    n = recording.n_samples
    start = 0
    while start < n:
        end = start + window_samples
        if end > n and drop_last:
            break
        starts.append(start)
        if max_windows is not None and len(starts) >= int(max_windows):
            break
        start += step_samples

    if not starts:
        raise ValueError(
            f"No windows created: n_samples={n}, window_samples={window_samples}, step_samples={step_samples}."
        )

    X = np.zeros((len(starts), window_samples, recording.n_channels), dtype=recording.signal.dtype)
    rows: list[dict[str, Any]] = []
    t = recording.effective_time()
    for i, s in enumerate(starts):
        e = min(s + window_samples, n)
        segment = recording.signal[s:e]
        X[i, : segment.shape[0], :] = segment
        rows.append({
            "window_id": i,
            "start_sample": int(s),
            "end_sample": int(e),
            "n_valid_samples": int(segment.shape[0]),
            "start_time": float(t[s]) if s < len(t) else None,
            "end_time": float(t[e - 1]) if e > 0 and e - 1 < len(t) else None,
            "padded": bool(segment.shape[0] < window_samples),
            "n_channels": int(recording.n_channels),
        })
    if astype:
        X = X.astype(str(astype))
    return WindowedSignal(X=X, index=pd.DataFrame(rows))
