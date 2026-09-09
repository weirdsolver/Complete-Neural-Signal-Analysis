from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_channels(recording_dir: Path, n_channels: int) -> list[str]:
    channels_csv = recording_dir / "channels.csv"
    if channels_csv.exists():
        try:
            df = pd.read_csv(channels_csv)
            for col in ("name", "channel_name", "label"):
                if col in df.columns:
                    names = [str(x) for x in df[col].tolist()]
                    if len(names) == n_channels:
                        return names
        except Exception:
            pass
    return [f"ch_{i:03d}" for i in range(n_channels)]


def _load_fs(recording_dir: Path, default_fs: float | None = None) -> float | None:
    metadata_json = recording_dir / "metadata.json"
    if metadata_json.exists():
        try:
            metadata = json.loads(metadata_json.read_text())
            for key in ("sampling_rate", "sampling_rate_hz", "fs", "sfreq"):
                if key in metadata and metadata[key] is not None:
                    return float(metadata[key])
        except Exception:
            pass
    return default_fs


def _safe_float(x: Any) -> float | None:
    try:
        value = float(x)
        if np.isfinite(value):
            return value
    except Exception:
        return None
    return None


def extract_features_from_signal(
    signal: np.ndarray,
    *,
    sampling_rate: float | None = None,
    channel_names: list[str] | None = None,
    dataset_id: str = "recording",
    max_fft_samples: int = 200_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract continuous-signal features without spike/event assumptions.

    Features are intentionally channel-count agnostic. They support EEG, ECoG,
    MEA, organoid, FinalSpark-like, Cortical Labs-like, and generic continuous
    neural matrices shaped samples x channels.
    """
    x = np.asarray(signal)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D signal matrix samples x channels, got shape {x.shape}")
    if x.shape[0] < x.shape[1]:
        # Do not transpose automatically here. The importer should already have
        # canonicalized orientation. Warn through metadata instead.
        pass

    n_samples, n_channels = x.shape
    names = channel_names or [f"ch_{i:03d}" for i in range(n_channels)]
    if len(names) != n_channels:
        names = [f"ch_{i:03d}" for i in range(n_channels)]

    # Use nan-safe metrics. Convert to float64 just for summaries.
    xf = x.astype(np.float64, copy=False)
    means = np.nanmean(xf, axis=0)
    stds = np.nanstd(xf, axis=0)
    variances = np.nanvar(xf, axis=0)
    rms = np.sqrt(np.nanmean(np.square(xf), axis=0))
    mins = np.nanmin(xf, axis=0)
    maxs = np.nanmax(xf, axis=0)
    ptp = maxs - mins
    nan_fraction = np.mean(~np.isfinite(xf), axis=0)

    centroid = np.full(n_channels, np.nan, dtype=np.float64)
    band_delta = np.full(n_channels, np.nan, dtype=np.float64)
    band_theta = np.full(n_channels, np.nan, dtype=np.float64)
    band_alpha = np.full(n_channels, np.nan, dtype=np.float64)
    band_beta = np.full(n_channels, np.nan, dtype=np.float64)
    band_gamma = np.full(n_channels, np.nan, dtype=np.float64)

    if sampling_rate and sampling_rate > 0 and n_samples >= 8:
        # Limit FFT cost for large recordings by taking an evenly spaced slice.
        if n_samples > max_fft_samples:
            step = int(np.ceil(n_samples / max_fft_samples))
            y = xf[::step]
            fs_eff = sampling_rate / step
        else:
            y = xf
            fs_eff = sampling_rate
        y = y - np.nanmean(y, axis=0, keepdims=True)
        y = np.nan_to_num(y, copy=False)
        freqs = np.fft.rfftfreq(y.shape[0], d=1.0 / fs_eff)
        psd = np.abs(np.fft.rfft(y, axis=0)) ** 2
        valid = freqs > 0
        denom = np.sum(psd[valid], axis=0)
        good = denom > 0
        centroid[good] = np.sum(freqs[valid, None] * psd[valid, :], axis=0)[good] / denom[good]

        def band_power(lo: float, hi: float) -> np.ndarray:
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                return np.full(n_channels, np.nan)
            return np.mean(psd[mask, :], axis=0)

        band_delta[:] = band_power(1, 4)
        band_theta[:] = band_power(4, 8)
        band_alpha[:] = band_power(8, 13)
        band_beta[:] = band_power(13, 30)
        band_gamma[:] = band_power(30, min(80, sampling_rate / 2))

    rows = []
    for i, name in enumerate(names):
        rows.append({
            "dataset_id": dataset_id,
            "channel_index": i,
            "channel_name": name,
            "mean": _safe_float(means[i]),
            "std": _safe_float(stds[i]),
            "variance": _safe_float(variances[i]),
            "rms": _safe_float(rms[i]),
            "min": _safe_float(mins[i]),
            "max": _safe_float(maxs[i]),
            "peak_to_peak": _safe_float(ptp[i]),
            "nan_fraction": _safe_float(nan_fraction[i]),
            "spectral_centroid_hz": _safe_float(centroid[i]),
            "delta_power": _safe_float(band_delta[i]),
            "theta_power": _safe_float(band_theta[i]),
            "alpha_power": _safe_float(band_alpha[i]),
            "beta_power": _safe_float(band_beta[i]),
            "gamma_power": _safe_float(band_gamma[i]),
        })
    df = pd.DataFrame(rows)
    summary = {
        "dataset_id": dataset_id,
        "n_samples": int(n_samples),
        "n_channels": int(n_channels),
        "sampling_rate_hz": float(sampling_rate) if sampling_rate else None,
        "duration_seconds": float(n_samples / sampling_rate) if sampling_rate else None,
        "global_rms_mean": _safe_float(df["rms"].mean()),
        "global_variance_mean": _safe_float(df["variance"].mean()),
        "global_centroid_mean_hz": _safe_float(df["spectral_centroid_hz"].mean()),
        "global_alpha_power_mean": _safe_float(df["alpha_power"].mean()),
        "global_beta_power_mean": _safe_float(df["beta_power"].mean()),
    }
    return df, summary


def extract_features_from_recording_dir(
    recording_dir: str | Path,
    *,
    dataset_id: str | None = None,
    sampling_rate: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    recording_dir = Path(recording_dir)
    signal_path = recording_dir / "signal.npy"
    if not signal_path.exists():
        processed_signal = recording_dir / "processed" / "signal.npy"
        if processed_signal.exists():
            signal_path = processed_signal
        else:
            raise FileNotFoundError(f"No signal.npy found in {recording_dir}")
    x = np.load(signal_path, mmap_mode="r")
    fs = _load_fs(signal_path.parent, sampling_rate)
    if fs is None:
        fs = _load_fs(recording_dir, sampling_rate)
    names = _load_channels(signal_path.parent, x.shape[1])
    if names == [f"ch_{i:03d}" for i in range(x.shape[1])]:
        names = _load_channels(recording_dir, x.shape[1])
    return extract_features_from_signal(
        x,
        sampling_rate=fs,
        channel_names=names,
        dataset_id=dataset_id or recording_dir.name,
    )
