from __future__ import annotations

from typing import Any

import numpy as np

from neuro_importer.core.recording import Recording
from neuro_importer.qc.channel_qc import channel_qc


def signal_qc(recording: Recording, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or {}
    x = np.asarray(recording.signal, dtype=float)
    finite = np.isfinite(x)
    finite_vals = x[finite]
    duration = None
    if recording.sampling_rate is not None and recording.sampling_rate > 0:
        duration = recording.n_samples / float(recording.sampling_rate)
    elif recording.time is not None and len(recording.time) > 1:
        duration = float(recording.time[-1] - recording.time[0])

    if finite_vals.size:
        median = float(np.nanmedian(finite_vals))
        mad = float(np.nanmedian(np.abs(finite_vals - median)))
        std = float(np.nanstd(finite_vals))
        outlier_z_warn = float(cfg.get("outlier_z_warn", 12.0))
        if std > 0:
            outlier_fraction = float((np.abs((finite_vals - np.nanmean(finite_vals)) / std) > outlier_z_warn).mean())
        else:
            outlier_fraction = 0.0
        distribution = {
            "mean": float(np.nanmean(finite_vals)),
            "std": std,
            "median": median,
            "mad": mad,
            "min": float(np.nanmin(finite_vals)),
            "max": float(np.nanmax(finite_vals)),
            "p01": float(np.nanpercentile(finite_vals, 1)),
            "p99": float(np.nanpercentile(finite_vals, 99)),
            "outlier_fraction_by_z": outlier_fraction,
        }
    else:
        distribution = {}

    cq = channel_qc(recording, cfg)
    nan_fraction = float(1.0 - finite.mean())
    warnings: list[str] = []
    if nan_fraction > float(cfg.get("nan_fraction_warn", 0.01)):
        warnings.append(f"High non-finite fraction: {nan_fraction:.4f}")
    flat_channels = cq[cq["flat"]]
    if len(flat_channels):
        warnings.append(f"Flat channels detected: {flat_channels['name'].astype(str).tolist()}")
    if distribution.get("outlier_fraction_by_z", 0.0) > 0.001:
        warnings.append("Large-amplitude outliers detected by z-score threshold.")

    return {
        "n_samples": int(recording.n_samples),
        "n_channels": int(recording.n_channels),
        "sampling_rate": recording.sampling_rate,
        "duration_seconds": duration,
        "dtype": str(recording.signal.dtype),
        "nan_or_inf_fraction": nan_fraction,
        "distribution": distribution,
        "flat_channel_count": int(len(flat_channels)),
        "warnings": warnings,
    }
