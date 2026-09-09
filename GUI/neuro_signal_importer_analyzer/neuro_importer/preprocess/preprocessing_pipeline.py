from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from neuro_importer.core.recording import Recording
from neuro_importer.preprocess.filters import bandpass_filter, demean, detrend, downsample, notch_filter
from neuro_importer.preprocess.normalization import normalize_signal


def _parse_bandpass(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (None if value[0] is None else float(value[0]), None if value[1] is None else float(value[1]))
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(",", "-").split("-") if p.strip()]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    raise ValueError("bandpass_hz must be null, [low, high], or a string like '1-100'")


def apply_preprocessing(recording: Recording, config: dict[str, Any] | None = None) -> Recording:
    """Return a new Recording with optional continuous-signal preprocessing applied.

    This never mutates the input Recording. Raw canonical exports should generally be
    saved before this step.
    """
    cfg = config or {}
    if not cfg.get("enabled", False):
        return recording

    q = recording.quality
    x = np.asarray(recording.signal)
    fs = recording.sampling_rate
    time = recording.time.copy() if recording.time is not None else None
    operations: list[str] = []

    if cfg.get("demean", False):
        x = demean(x)
        operations.append("demean")

    if cfg.get("detrend", False):
        x = detrend(x)
        operations.append("detrend")

    notch = cfg.get("notch_hz")
    if notch is not None:
        if fs is None:
            q.add_warning("Skipped notch filter because sampling_rate is missing.")
        else:
            x = notch_filter(x, float(fs), float(notch))
            operations.append(f"notch_{notch}_Hz")

    bp = cfg.get("bandpass_hz")
    if bp is not None:
        if fs is None:
            q.add_warning("Skipped bandpass filter because sampling_rate is missing.")
        else:
            low, high = _parse_bandpass(bp)
            x = bandpass_filter(x, float(fs), low, high)
            operations.append(f"bandpass_{low}_{high}_Hz")

    target = cfg.get("downsample_to_hz")
    if target is not None:
        if fs is None:
            q.add_warning("Skipped downsampling because sampling_rate is missing.")
        else:
            x, new_fs, factor = downsample(x, float(fs), float(target))
            if time is not None:
                time = time[::factor]
            fs = new_fs
            operations.append(f"downsample_factor_{factor}_to_{new_fs:g}_Hz")

    norm = cfg.get("normalization")
    if norm:
        x = normalize_signal(x, norm)
        operations.append(f"normalize_{norm}")

    astype = cfg.get("astype")
    if astype:
        x = x.astype(str(astype))
        operations.append(f"astype_{astype}")

    metadata = dict(recording.metadata)
    metadata["preprocessing"] = {
        "enabled": True,
        "operations": operations,
        "config": cfg,
        "raw_n_samples": recording.n_samples,
        "processed_n_samples": int(x.shape[0]),
        "raw_sampling_rate": recording.sampling_rate,
        "processed_sampling_rate": fs,
    }
    q.add_info(f"Applied preprocessing: {operations if operations else 'enabled but no operations selected'}")

    return replace(recording, signal=x, sampling_rate=fs, time=time, metadata=metadata)
