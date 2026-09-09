from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from neuro_importer.core.recording import Recording


def channel_qc(recording: Recording, config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or {}
    flat_std_threshold = float(cfg.get("flat_std_threshold", 1e-12))
    x = np.asarray(recording.signal, dtype=float)
    names = recording.channel_names()
    finite = np.isfinite(x)
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        col = x[:, i]
        fin = finite[:, i]
        valid = col[fin]
        if valid.size == 0:
            rows.append({
                "channel_index": i,
                "name": name,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "peak_to_peak": np.nan,
                "nan_fraction": 1.0,
                "flat": True,
            })
            continue
        std = float(np.nanstd(valid))
        mn = float(np.nanmin(valid))
        mx = float(np.nanmax(valid))
        rows.append({
            "channel_index": i,
            "name": name,
            "mean": float(np.nanmean(valid)),
            "std": std,
            "min": mn,
            "max": mx,
            "peak_to_peak": float(mx - mn),
            "nan_fraction": float(1.0 - fin.mean()),
            "flat": bool(std <= flat_std_threshold),
        })
    return pd.DataFrame(rows)
