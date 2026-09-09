from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from neuro_importer.core.quality import QualityReport
from neuro_importer.adapters.utils import flatten, stringify_label

DEFAULT_AUX_PREFIXES = ("RESP", "AUX", "BIP", "EOG", "EMG", "ECG", "EKG", "TRIG", "STATUS", "STI")
DEFAULT_AUX_NAMES = {"BIP1", "BIP2", "RESP1", "GND", "GROUND", "REF", "REFERENCE"}


def labels_from_any(value: Any) -> list[str]:
    if value is None:
        return []
    labels = [stringify_label(v) for v in flatten(value)]
    return [x for x in labels if x and x not in {"None", "nan", "NaN"}]


def numeric_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(flatten(value), dtype=object).reshape(-1)
        if arr.size == 0:
            return None
        numeric = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
        if np.isnan(numeric).all():
            return None
        return numeric
    except Exception:
        return None


def as_numeric_2d(value: Any, *, name: str = "signal") -> np.ndarray:
    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must resolve to a 2D numeric array. Got shape {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(float)
    return arr


def normalize_signal_orientation(
    signal_raw: np.ndarray,
    *,
    labels: list[str] | None = None,
    time: np.ndarray | None = None,
    n_channels: int | None = None,
    n_samples: int | None = None,
    quality: QualityReport | None = None,
) -> tuple[np.ndarray, str]:
    """Normalize a 2D array into samples × channels.

    Uses label count, time length, declared channel/sample counts, then a final
    rows<columns heuristic common in neurophysiology files stored as channels × samples.
    """
    signal_raw = as_numeric_2d(signal_raw)
    rows, cols = signal_raw.shape

    label_count = len(labels) if labels else None
    time_count = len(time) if time is not None else None
    channel_counts = {x for x in [label_count, n_channels] if isinstance(x, int) and x > 0}
    sample_counts = {x for x in [time_count, n_samples] if isinstance(x, int) and x > 0}

    if rows in channel_counts and cols in sample_counts:
        return signal_raw.T, "Original signal appeared to be channels × samples; transposed to samples × channels."
    if cols in channel_counts and rows in sample_counts:
        return signal_raw, "Original signal appeared to already be samples × channels."

    if label_count is not None:
        if rows == label_count and cols != label_count:
            return signal_raw.T, "Original signal first dimension matched channel labels; transposed to samples × channels."
        if cols == label_count and rows != label_count:
            return signal_raw, "Original signal second dimension matched channel labels; kept as samples × channels."

    if time_count is not None:
        if cols == time_count and rows != time_count:
            return signal_raw.T, "Original signal second dimension matched time; transposed to samples × channels."
        if rows == time_count and cols != time_count:
            return signal_raw, "Original signal first dimension matched time; kept as samples × channels."

    if rows < cols:
        if quality is not None:
            quality.add_warning("Could not prove signal orientation from metadata; assumed rows are channels because rows < columns.")
        return signal_raw.T, "Heuristic orientation assumption: rows < columns, so transposed to samples × channels."

    if quality is not None:
        quality.add_warning("Could not prove signal orientation from metadata; assumed rows are samples.")
    return signal_raw, "Heuristic orientation assumption: rows are samples and columns are channels."


def make_channel_table(
    labels: Iterable[str] | None,
    n_channels: int,
    *,
    quality: QualityReport | None = None,
    exclude_names: Iterable[str] | None = None,
    aux_prefixes: tuple[str, ...] = DEFAULT_AUX_PREFIXES,
) -> pd.DataFrame:
    names = list(labels or [])
    if not names:
        names = [f"ch_{i:03d}" for i in range(n_channels)]
        if quality is not None:
            quality.add_warning("No channel labels found; generated generic channel names.")
    if len(names) != n_channels:
        if quality is not None:
            quality.add_warning(f"Label count ({len(names)}) did not match signal channels ({n_channels}); adjusting labels.")
        if len(names) < n_channels:
            names = names + [f"ch_{i:03d}" for i in range(len(names), n_channels)]
        else:
            names = names[:n_channels]

    excluded = {x.upper() for x in (exclude_names or [])} | DEFAULT_AUX_NAMES
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        upper = str(name).upper()
        is_aux = upper in excluded or upper.startswith(aux_prefixes)
        if upper.startswith("RESP"):
            ch_type = "respiratory"
        elif upper.startswith(("EOG", "EMG", "ECG", "EKG")):
            ch_type = "biopotential_aux"
        elif upper.startswith("BIP"):
            ch_type = "bipolar_aux"
        elif upper.startswith(("TRIG", "STATUS", "STI")):
            ch_type = "trigger_aux"
        elif upper in {"GND", "GROUND", "REF", "REFERENCE"}:
            ch_type = "reference_aux"
        elif upper.startswith("AUX"):
            ch_type = "auxiliary"
        else:
            ch_type = "neural"
        rows.append({"index": idx, "name": str(name), "type": ch_type, "include_by_default": not is_aux})
    return pd.DataFrame(rows)


def apply_aux_filter(signal: np.ndarray, channels: pd.DataFrame, *, include_aux: bool, quality: QualityReport) -> tuple[np.ndarray, pd.DataFrame]:
    if include_aux:
        quality.add_info("Kept auxiliary channels because include_aux=True.")
        return signal, channels.copy().reset_index(drop=True)
    if "include_by_default" not in channels.columns:
        return signal, channels.copy().reset_index(drop=True)
    mask = channels["include_by_default"].to_numpy(dtype=bool)
    if len(mask) != signal.shape[1]:
        quality.add_warning("Could not apply auxiliary-channel filter because mask length did not match signal channel count.")
        return signal, channels.copy().reset_index(drop=True)
    if mask.all():
        return signal, channels.copy().reset_index(drop=True)
    excluded = channels.loc[~mask, "name"].astype(str).tolist()
    kept_channels = channels.loc[mask].copy().reset_index(drop=True)
    kept_channels["index"] = range(len(kept_channels))
    quality.add_info(f"Excluded auxiliary/non-neural channels from signal export: {excluded}")
    return signal[:, mask], kept_channels


def synthesize_or_validate_time(time: np.ndarray | None, n_samples: int, fs: float | None, quality: QualityReport) -> np.ndarray:
    if time is not None and len(time) == n_samples:
        return np.asarray(time, dtype=float).reshape(-1)
    if fs is not None and fs > 0:
        quality.add_warning("Time vector missing/mismatched; synthesized seconds from sampling rate.")
        return np.arange(n_samples, dtype=float) / float(fs)
    quality.add_warning("Time vector and sampling rate missing/mismatched; synthesized sample index as Time.")
    return np.arange(n_samples, dtype=float)
