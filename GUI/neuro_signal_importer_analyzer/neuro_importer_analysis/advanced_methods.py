from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        return None
    return None


def _safe_int(value: Any, default: int) -> int:
    try:
        out = int(value)
        if out > 0:
            return out
    except Exception:
        pass
    return default


def _positive_float(value: Any, default: float) -> float:
    try:
        out = float(value)
        if math.isfinite(out) and out > 0:
            return out
    except Exception:
        pass
    return float(default)


def _load_signal_path(recording_dir: Path) -> Path:
    candidates = [
        recording_dir / "signal.npy",
        recording_dir / "processed" / "signal.npy",
        recording_dir / "raw" / "signal.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No signal.npy found in {recording_dir}")


def _load_metadata(recording_dir: Path, signal_path: Path) -> dict[str, Any]:
    for candidate in (
        signal_path.parent / "metadata.json",
        recording_dir / "metadata.json",
        recording_dir / "processed" / "metadata.json",
        recording_dir / "raw" / "metadata.json",
    ):
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _sampling_rate(metadata: dict[str, Any], override: Any = None) -> float:
    if override not in (None, ""):
        return _positive_float(override, 1.0)
    for key in ("sampling_rate", "sampling_rate_hz", "sample_rate_hz", "fs", "sfreq"):
        if metadata.get(key) not in (None, ""):
            return _positive_float(metadata[key], 1.0)
    return 1.0


def _load_channels(recording_dir: Path, signal_path: Path, n_channels: int) -> list[str]:
    for candidate in (
        signal_path.parent / "channels.csv",
        recording_dir / "channels.csv",
        recording_dir / "processed" / "channels.csv",
        recording_dir / "raw" / "channels.csv",
    ):
        if not candidate.exists():
            continue
        try:
            df = pd.read_csv(candidate)
            name_col = next((col for col in ("name", "channel_name", "label", "channel") if col in df.columns), None)
            if name_col:
                names = [str(x) for x in df[name_col].tolist()]
                if len(names) == n_channels:
                    return names
        except Exception:
            pass
    return [f"ch_{i:03d}" for i in range(n_channels)]


def _load_recording(recording_dir: str | Path, *, sampling_rate: Any = None, max_samples: Any = None) -> dict[str, Any]:
    root = Path(recording_dir).expanduser().resolve()
    signal_path = _load_signal_path(root)
    signal = np.load(signal_path, mmap_mode="r")
    if signal.ndim != 2:
        raise ValueError(f"Expected signal.npy samples x channels, got shape {signal.shape}")
    n_samples, n_channels = signal.shape
    metadata = _load_metadata(root, signal_path)
    fs = _sampling_rate(metadata, sampling_rate)
    channels = _load_channels(root, signal_path, n_channels)
    max_count = _safe_int(max_samples, 240_000) if max_samples not in (None, "") else 240_000
    if n_samples > max_count:
        step = int(math.ceil(n_samples / max_count))
        data = np.asarray(signal[::step], dtype=np.float64)
        fs_eff = fs / step if fs > 0 else fs
    else:
        step = 1
        data = np.asarray(signal, dtype=np.float64)
        fs_eff = fs
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "recording_dir": str(root),
        "signal_path": str(signal_path),
        "metadata": metadata,
        "signal": data,
        "sampling_rate_hz": fs_eff,
        "original_sampling_rate_hz": fs,
        "analysis_downsample_step": step,
        "n_samples": int(n_samples),
        "analysis_samples": int(data.shape[0]),
        "n_channels": int(n_channels),
        "channels": channels,
        "duration_sec": float(n_samples / fs) if fs > 0 else None,
    }


def _nperseg(sample_count: int, fs: float) -> int:
    if sample_count < 8:
        return max(1, sample_count)
    return int(min(sample_count, max(8, min(4096, round(fs * 4.0)))))


def _band_power_summary(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    band_name = str(params.get("band") or "custom").lower()
    if band_name in BANDS and params.get("min_hz") in (None, "") and params.get("max_hz") in (None, ""):
        min_hz, max_hz = BANDS[band_name]
    else:
        min_hz = _positive_float(params.get("min_hz"), 8.0)
        max_hz = _positive_float(params.get("max_hz"), 13.0)
        band_name = next((name for name, pair in BANDS.items() if abs(pair[0] - min_hz) < 1e-9 and abs(pair[1] - max_hz) < 1e-9), "custom")
    if max_hz <= min_hz:
        raise ValueError("max_hz must be greater than min_hz")
    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=params.get("max_analysis_samples"))
    x = rec["signal"]
    fs = float(rec["sampling_rate_hz"] or 1.0)
    nps = _nperseg(x.shape[0], fs)
    freqs, psd = welch(x - np.mean(x, axis=0, keepdims=True), fs=fs, nperseg=nps, axis=0)
    total_mask = (freqs > 0) & (freqs < min(fs / 2.0, 120.0))
    band_mask = (freqs >= min_hz) & (freqs < min(max_hz, fs / 2.0))
    if not np.any(band_mask):
        raise ValueError(f"No frequency bins available in requested band {min_hz:g}-{max_hz:g} Hz at analysis sampling rate {fs:g} Hz")
    band_power = np.trapezoid(psd[band_mask, :], freqs[band_mask], axis=0)
    total_power = np.trapezoid(psd[total_mask, :], freqs[total_mask], axis=0) if np.any(total_mask) else np.sum(psd, axis=0)
    rel_power = band_power / (total_power + 1e-20)
    peak_idx = np.argmax(psd[band_mask, :], axis=0)
    band_freqs = freqs[band_mask]
    rows: list[dict[str, Any]] = []
    for i, channel in enumerate(rec["channels"]):
        rows.append({
            "channel": channel,
            "channel_index": i,
            "band_power": _safe_float(band_power[i]),
            "relative_power": _safe_float(rel_power[i]),
            "peak_frequency_hz": _safe_float(band_freqs[int(peak_idx[i])]),
        })
    top_i = int(np.nanargmax(band_power)) if band_power.size else 0
    return {
        "band_power_summary": {
            "band": band_name,
            "min_hz": min_hz,
            "max_hz": max_hz,
            "rows": rows,
            "summary": {
                "mean_band_power": _safe_float(np.nanmean(band_power)),
                "median_relative_power": _safe_float(np.nanmedian(rel_power)),
                "top_channel": rows[top_i]["channel"] if rows else None,
                "top_channel_power": _safe_float(band_power[top_i]) if rows else None,
                "n_channels": rec["n_channels"],
                "sampling_rate_analysis_hz": fs,
            },
        }
    }


def _standardize(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.median(x, axis=0, keepdims=True)
    scale = np.std(x, axis=0, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return x / scale


def _maybe_filter_for_spikes(x: np.ndarray, fs: float, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    low = params.get("bandpass_low_hz")
    high = params.get("bandpass_high_hz")
    if low in (None, "") and high in (None, ""):
        return _standardize(x), "standardized_broadband"
    low_f = _safe_float(low)
    high_f = _safe_float(high)
    nyq = fs / 2.0
    if nyq <= 0:
        return _standardize(x), "standardized_broadband_no_valid_fs"
    try:
        if low_f and high_f and 0 < low_f < high_f < nyq:
            sos = butter(3, [low_f / nyq, high_f / nyq], btype="bandpass", output="sos")
            return _standardize(sosfiltfilt(sos, x, axis=0)), f"bandpass_{low_f:g}_{high_f:g}_hz"
        if low_f and 0 < low_f < nyq:
            sos = butter(3, low_f / nyq, btype="highpass", output="sos")
            return _standardize(sosfiltfilt(sos, x, axis=0)), f"highpass_{low_f:g}_hz"
        if high_f and 0 < high_f < nyq:
            sos = butter(3, high_f / nyq, btype="lowpass", output="sos")
            return _standardize(sosfiltfilt(sos, x, axis=0)), f"lowpass_{high_f:g}_hz"
    except Exception:
        pass
    return _standardize(x), "standardized_broadband_filter_unavailable"


def _detect_spikes_from_recording(recording_dir: str | Path, params: dict[str, Any]) -> tuple[dict[str, list[float]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=params.get("max_analysis_samples"))
    x, filter_note = _maybe_filter_for_spikes(rec["signal"], float(rec["sampling_rate_hz"] or 1.0), params)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    threshold_multiplier = _positive_float(params.get("threshold_multiplier"), 5.0)
    refractory_ms = _positive_float(params.get("refractory_ms"), 2.0)
    max_spikes_per_channel = _safe_int(params.get("max_spikes_per_channel"), 5000)
    polarity = str(params.get("polarity") or "negative").lower()
    distance = max(1, int(round((refractory_ms / 1000.0) * fs)))
    spikes: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    duration = float(rec["analysis_samples"] / fs) if fs > 0 else 0.0
    for ch_i, channel in enumerate(rec["channels"]):
        trace = x[:, ch_i]
        mad = np.median(np.abs(trace - np.median(trace)))
        threshold = threshold_multiplier * (1.4826 * mad if mad > 1e-12 else np.std(trace) + 1e-12)
        peak_indices: np.ndarray
        if polarity == "positive":
            peak_indices, _ = find_peaks(trace, height=threshold, distance=distance)
        elif polarity == "both":
            pos, _ = find_peaks(trace, height=threshold, distance=distance)
            neg, _ = find_peaks(-trace, height=threshold, distance=distance)
            peak_indices = np.unique(np.concatenate([pos, neg]))
        else:
            peak_indices, _ = find_peaks(-trace, height=threshold, distance=distance)
        times = (peak_indices.astype(np.float64) / fs).tolist()
        kept_times = times[:max_spikes_per_channel]
        spikes[channel] = [round(float(t), 6) for t in kept_times]
        rows.append({
            "channel": channel,
            "channel_index": ch_i,
            "spike_count": int(len(times)),
            "stored_spike_count": int(len(kept_times)),
            "firing_rate_hz": _safe_float(len(times) / duration) if duration > 0 else 0.0,
            "threshold": _safe_float(threshold),
            "first_spike_sec": _safe_float(times[0]) if times else None,
        })
    summary = {
        "total_spikes": int(sum(row["spike_count"] for row in rows)),
        "mean_firing_rate_hz": _safe_float(np.mean([row["firing_rate_hz"] for row in rows])) if rows else 0.0,
        "active_channels": int(sum(1 for row in rows if row["spike_count"] > 0)),
        "duration_sec": duration,
        "sampling_rate_analysis_hz": fs,
        "analysis_downsample_step": rec["analysis_downsample_step"],
        "filter_note": filter_note,
        "polarity": polarity,
        "threshold_multiplier": threshold_multiplier,
        "refractory_ms": refractory_ms,
        "spikes_truncated_per_channel_at": max_spikes_per_channel,
    }
    return spikes, rows, summary, rec


def _spike_detect(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    spikes, rows, summary, rec = _detect_spikes_from_recording(recording_dir, params)
    return {
        "spike_detect": {
            "spikes": spikes,
            "rows": rows,
            "summary": summary,
            "channels": rec["channels"],
        },
        "mea": {
            "spikes": spikes,
            "duration_sec": summary["duration_sec"],
        },
    }


def _flatten_spikes(spikes: dict[str, list[float]]) -> list[tuple[float, str]]:
    events: list[tuple[float, str]] = []
    for channel, times in spikes.items():
        for t in times:
            try:
                events.append((float(t), channel))
            except Exception:
                pass
    events.sort(key=lambda item: item[0])
    return events


def _network_burst(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    spikes, rows, spike_summary, rec = _detect_spikes_from_recording(recording_dir, params)
    duration = float(spike_summary.get("duration_sec") or 0.0)
    bin_size_ms = _positive_float(params.get("bin_size_ms"), 10.0)
    threshold_count = _safe_int(params.get("threshold_count"), max(3, int(math.ceil(0.1 * max(1, rec["n_channels"])))) )
    merge_gap_ms = _positive_float(params.get("merge_gap_ms"), 50.0)
    min_spikes = _safe_int(params.get("min_spikes"), 5)
    bin_size = bin_size_ms / 1000.0
    merge_gap = merge_gap_ms / 1000.0
    if duration <= 0:
        bins = np.asarray([0.0, bin_size])
    else:
        bins = np.arange(0.0, duration + bin_size, bin_size)
        if bins.size < 2:
            bins = np.asarray([0.0, max(bin_size, duration)])
    all_events = _flatten_spikes(spikes)
    all_times = np.asarray([t for t, _ in all_events], dtype=float)
    counts, edges = np.histogram(all_times, bins=bins) if all_times.size else (np.zeros(len(bins) - 1, dtype=int), bins)
    active = counts >= threshold_count
    bursts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for idx, is_active in enumerate(active):
        if not is_active:
            continue
        start = float(edges[idx])
        end = float(edges[idx + 1])
        if current and start - float(current["end_sec"]) <= merge_gap:
            current["end_sec"] = end
            current["peak_bin_count"] = max(int(current["peak_bin_count"]), int(counts[idx]))
        else:
            if current:
                bursts.append(current)
            current = {"start_sec": start, "end_sec": end, "peak_bin_count": int(counts[idx])}
    if current:
        bursts.append(current)
    filtered: list[dict[str, Any]] = []
    for burst in bursts:
        start = float(burst["start_sec"])
        end = float(burst["end_sec"])
        burst_events = [(t, ch) for t, ch in all_events if start <= t <= end]
        if len(burst_events) < min_spikes:
            continue
        channels = sorted({ch for _, ch in burst_events})
        filtered.append({
            "start_sec": round(start, 6),
            "end_sec": round(end, 6),
            "duration_ms": round((end - start) * 1000.0, 3),
            "spike_count": int(len(burst_events)),
            "active_channels": int(len(channels)),
            "peak_bin_count": int(burst["peak_bin_count"]),
            "channels": channels[:30],
        })
    return {
        "network_burst": {
            "bursts": filtered,
            "timeline": filtered,
            "summary": {
                "burst_count": len(filtered),
                "bin_size_ms": bin_size_ms,
                "threshold_count": threshold_count,
                "merge_gap_ms": merge_gap_ms,
                "min_spikes": min_spikes,
                "total_detected_spikes": spike_summary["total_spikes"],
                "duration_sec": duration,
            },
            "spike_detection_summary": spike_summary,
        }
    }


def _bin_spikes(spikes: dict[str, list[float]], channels: list[str], duration_sec: float, bin_size_sec: float) -> np.ndarray:
    bin_count = max(1, int(math.ceil(duration_sec / bin_size_sec)))
    binned = np.zeros((len(channels), bin_count), dtype=np.float64)
    last_bin = bin_count - 1
    for row, channel in enumerate(channels):
        times = np.asarray(spikes.get(channel, []), dtype=float)
        if not times.size:
            continue
        valid = times[(times >= 0.0) & (times <= duration_sec)]
        if valid.size == 0:
            continue
        indices = np.minimum((valid / bin_size_sec).astype(int), last_bin)
        np.add.at(binned[row], indices, 1)
    return binned


def _electrode_connectivity(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    spikes, rows, spike_summary, rec = _detect_spikes_from_recording(recording_dir, params)
    channels = rec["channels"]
    duration = float(spike_summary.get("duration_sec") or 0.0)
    bin_size_ms = _positive_float(params.get("bin_size_ms"), 5.0)
    max_lag_ms = float(params.get("max_lag_ms") if params.get("max_lag_ms") not in (None, "") else 20.0)
    if not math.isfinite(max_lag_ms) or max_lag_ms < 0:
        raise ValueError("max_lag_ms must be a non-negative finite number")
    bin_size_sec = bin_size_ms / 1000.0
    lag_bins = int(round((max_lag_ms / 1000.0) / bin_size_sec))
    binned = _bin_spikes(spikes, channels, duration, bin_size_sec)
    c = len(channels)
    matrix = np.eye(c, dtype=float)
    links: list[dict[str, Any]] = []
    strongest: dict[str, Any] | None = None
    for i in range(c):
        for j in range(i + 1, c):
            best_score = 0.0
            best_lag = 0
            best_support = 0.0
            for lag in range(-lag_bins, lag_bins + 1):
                if lag >= 0:
                    left = binned[i, lag:]
                    right = binned[j, : binned.shape[1] - lag]
                else:
                    offset = -lag
                    left = binned[i, : binned.shape[1] - offset]
                    right = binned[j, offset:]
                if left.size == 0 or right.size == 0:
                    score = 0.0
                    support = 0.0
                else:
                    denom = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
                    support = float(np.dot(left, right))
                    score = support / denom if denom > 0 else 0.0
                if (score, support, -abs(lag)) > (best_score, best_support, -abs(best_lag)):
                    best_score, best_lag, best_support = score, lag, support
            score_r = round(float(max(0.0, best_score)), 6)
            lag_ms = round(float(best_lag * bin_size_ms), 6)
            matrix[i, j] = matrix[j, i] = score_r
            link = {
                "source": channels[i],
                "target": channels[j],
                "score": score_r,
                "lag_ms": lag_ms,
                "support": int(round(best_support)),
            }
            links.append(link)
            if strongest is None or (link["score"], link["support"]) > (strongest["score"], strongest["support"]):
                strongest = link
    links.sort(key=lambda row: (-row["score"], -row["support"], row["source"], row["target"]))
    max_links = _safe_int(params.get("max_links"), 200)
    return {
        "electrode_connectivity": {
            "channels": channels,
            "matrix": np.round(matrix, 6).tolist(),
            "links": links[:max_links],
            "summary": {
                "method_note": "reference binned spike-train cross-correlation over spikes detected from the continuous signal",
                "strongest_pair": strongest or {"source": "", "target": "", "score": 0.0, "lag_ms": 0.0, "support": 0},
                "bin_size_ms": bin_size_ms,
                "max_lag_ms": max_lag_ms,
                "stored_link_count": min(len(links), max_links),
                "total_pair_count": len(links),
                "spike_detection_summary": spike_summary,
            },
        }
    }


HFD_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    # v0.11.22: expose the same speed labels used by the newer analyses.
    # Keep the legacy names as aliases for existing saved requests.
    "ultra": {
        "k_max": 6,
        "decimate": 5,
        "max_samples": 20_000,
        "rolling_window_sec": 1.0,
        "rolling_step_sec": 3.0,
        "rolling_max_windows": 18,
    },
    "fast": {
        "k_max": 8,
        "decimate": 2,
        "max_samples": 60_000,
        "rolling_window_sec": 2.0,
        "rolling_step_sec": 2.0,
        "rolling_max_windows": 40,
    },
    "balanced": {
        "k_max": 10,
        "decimate": 1,
        "max_samples": 120_000,
        "rolling_window_sec": 2.0,
        "rolling_step_sec": 1.0,
        "rolling_max_windows": 80,
    },
    "full": {
        "k_max": 12,
        "decimate": 1,
        "max_samples": None,
        "rolling_window_sec": 2.0,
        "rolling_step_sec": 1.0,
        "rolling_max_windows": 120,
    },
    "ultrafast": {
        "k_max": 6,
        "decimate": 5,
        "max_samples": 20_000,
        "rolling_window_sec": 1.0,
        "rolling_step_sec": 3.0,
        "rolling_max_windows": 18,
    },
    "normal": {
        "k_max": 10,
        "decimate": 1,
        "max_samples": None,
        "rolling_window_sec": 2.0,
        "rolling_step_sec": 1.0,
        "rolling_max_windows": 80,
    },
}

HFD_SCALP_XY: dict[str, tuple[float, float]] = {
    "Fp1": (-0.45, 1.00), "Fpz": (0.00, 1.08), "Fp2": (0.45, 1.00),
    "F7": (-0.95, 0.60), "F3": (-0.45, 0.55), "Fz": (0.00, 0.60),
    "F4": (0.45, 0.55), "F8": (0.95, 0.60),
    "FC5": (-0.75, 0.25), "FC1": (-0.25, 0.25), "FC2": (0.25, 0.25), "FC6": (0.75, 0.25),
    "M1": (-1.15, 0.05), "T7": (-1.00, 0.00), "C3": (-0.50, 0.00),
    "Cz": (0.00, 0.00), "C4": (0.50, 0.00), "T8": (1.00, 0.00), "M2": (1.15, 0.05),
    "CP5": (-0.75, -0.30), "CP1": (-0.25, -0.30), "CP2": (0.25, -0.30), "CP6": (0.75, -0.30),
    "P7": (-0.90, -0.60), "P3": (-0.45, -0.60), "Pz": (0.00, -0.65),
    "P4": (0.45, -0.60), "P8": (0.90, -0.60),
    "POz": (0.00, -0.88), "O1": (-0.35, -1.00), "Oz": (0.00, -1.08), "O2": (0.35, -1.00),
}

HFD_REGIONS: dict[str, list[str]] = {
    "Frontal": ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "Frontocentral": ["FC5", "FC1", "FC2", "FC6"],
    "Central": ["T7", "C3", "Cz", "C4", "T8"],
    "Centroparietal": ["CP5", "CP1", "CP2", "CP6"],
    "Parietal": ["P7", "P3", "Pz", "P4", "P8"],
    "Occipital": ["POz", "O1", "Oz", "O2"],
    "Mastoid": ["M1", "M2"],
}

HFD_LR_PAIRS: list[tuple[str, str]] = [
    ("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("FC5", "FC6"), ("FC1", "FC2"),
    ("T7", "T8"), ("C3", "C4"), ("CP5", "CP6"), ("CP1", "CP2"),
    ("P7", "P8"), ("P3", "P4"), ("O1", "O2"),
]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return _safe_float(value)


def _hfd_bool(value: Any, default: bool = True) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _higuchi_fd_curve(x: np.ndarray, k_max: int = 10) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n < k_max + 2:
        return np.nan, None, None
    k_vals = np.arange(1, k_max + 1, dtype=int)
    lk_values: list[float] = []
    for k in k_vals:
        lm_vals: list[float] = []
        for m in range(k):
            n_max = (n - m - 1) // k
            if n_max < 1:
                continue
            idx1 = m + np.arange(1, n_max + 1) * k
            idx0 = m + np.arange(0, n_max) * k
            diffs = np.abs(x[idx1] - x[idx0]).sum()
            lm_vals.append(float(diffs * (n - 1) / (n_max * k * k)))
        lk_values.append(float(np.mean(lm_vals)) if lm_vals else np.nan)
    lk = np.asarray(lk_values, dtype=float)
    mask = np.isfinite(lk) & (lk > 0)
    if np.sum(mask) < 2:
        return np.nan, k_vals, lk
    slope = np.polyfit(np.log(k_vals[mask]), np.log(lk[mask]), 1)[0]
    return float(-slope), k_vals, lk


def _higuchi_fit_diagnostics(k_vals: np.ndarray | None, lk: np.ndarray | None) -> dict[str, float]:
    empty = {"slope": np.nan, "intercept": np.nan, "hfd": np.nan, "r2": np.nan, "rmse": np.nan}
    if k_vals is None or lk is None:
        return empty
    mask = np.isfinite(lk) & (lk > 0)
    if np.sum(mask) < 2:
        return empty
    logk = np.log(k_vals[mask])
    logl = np.log(lk[mask])
    slope, intercept = np.polyfit(logk, logl, 1)
    pred = slope * logk + intercept
    resid = logl - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((logl - np.mean(logl)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "hfd": float(-slope),
        "r2": float(1.0 - ss_res / (ss_tot + 1e-12)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
    }


def _load_hfd_segment(recording_dir: str | Path, params: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(recording_dir).expanduser().resolve()
    signal_path = _load_signal_path(root)
    signal = np.load(signal_path, mmap_mode="r")
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal.npy, got shape {signal.shape}")
    metadata = _load_metadata(root, signal_path)
    fs = _sampling_rate(metadata, params.get("sampling_rate"))
    # Canonical exports are samples x channels. If a channels.csv strongly suggests
    # channels x samples, transpose after slicing.
    n0, n1 = int(signal.shape[0]), int(signal.shape[1])
    ch_sample_major = _load_channels(root, signal_path, n1)
    ch_channel_major = _load_channels(root, signal_path, n0)
    assume_channel_major = len(ch_channel_major) == n0 and len(ch_sample_major) != n1 and n0 < n1
    total_samples = n1 if assume_channel_major else n0
    channels = ch_channel_major if assume_channel_major else ch_sample_major
    start_sec = _optional_float(params.get("segment_start_sec"))
    end_sec = _optional_float(params.get("segment_end_sec"))
    start_idx = 0 if start_sec is None else max(0, int(start_sec * fs))
    end_idx = total_samples if end_sec is None else min(total_samples, int(end_sec * fs))
    if end_idx <= start_idx:
        raise ValueError("segment_end_sec must be greater than segment_start_sec")
    decimate = _safe_int(params.get("decimate"), int(cfg.get("decimate") or 1))
    if assume_channel_major:
        data = np.asarray(signal[:, start_idx:end_idx:decimate], dtype=np.float64).T
    else:
        data = np.asarray(signal[start_idx:end_idx:decimate, :], dtype=np.float64)
    max_samples_param = params.get("max_samples")
    if max_samples_param in (None, ""):
        max_samples = cfg.get("max_samples")
    else:
        max_samples = _safe_int(max_samples_param, int(cfg.get("max_samples") or data.shape[0]))
    if max_samples is not None and data.shape[0] > int(max_samples):
        data = data[: int(max_samples), :]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "root": root,
        "signal_path": signal_path,
        "metadata": metadata,
        "signal": data,
        "channels": channels,
        "sampling_rate_hz": fs / decimate if fs > 0 else fs,
        "original_sampling_rate_hz": fs,
        "decimate": decimate,
        "source_total_samples": total_samples,
        "segment_start_sec": float(start_idx / fs) if fs > 0 else 0.0,
        "segment_end_sec": float(end_idx / fs) if fs > 0 else None,
        "segment_original_samples": int(end_idx - start_idx),
    }


def _fallback_scalp_points(channels: list[str], hfd_map: dict[str, float]) -> list[dict[str, Any]]:
    n = max(1, len(channels))
    points: list[dict[str, Any]] = []
    for i, ch in enumerate(channels):
        val = hfd_map.get(ch)
        if val is None or not np.isfinite(val):
            continue
        if ch in HFD_SCALP_XY:
            x, y = HFD_SCALP_XY[ch]
            layout = "10-20"
        else:
            angle = -math.pi / 2 + 2 * math.pi * i / n
            x, y = 0.95 * math.cos(angle), 0.95 * math.sin(angle)
            layout = "circular"
        points.append({"channel": ch, "x": float(x), "y": float(y), "higuchi_fd": float(val), "layout": layout})
    return points


def _hfd_regional_summary(channels: list[str], hfd_map: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assigned = {ch for group in HFD_REGIONS.values() for ch in group}
    region_rows: list[dict[str, Any]] = []
    region_points: list[dict[str, Any]] = []
    region_map = {**HFD_REGIONS}
    other = [ch for ch in channels if ch not in assigned]
    if other:
        region_map["Other"] = other
    for region_name, chans in region_map.items():
        vals: list[float] = []
        for ch in chans:
            val = hfd_map.get(ch)
            if val is not None and np.isfinite(val):
                vals.append(float(val))
                region_points.append({"region": region_name, "channel": ch, "higuchi_fd": float(val)})
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        region_rows.append({
            "region": region_name,
            "mean_hfd": float(np.mean(arr)),
            "std_hfd": sd,
            "sem_hfd": float(sd / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            "n_channels": int(len(arr)),
        })
    return region_rows, region_points


def _hfd_asymmetry(hfd_map: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in HFD_LR_PAIRS:
        lv = hfd_map.get(left)
        rv = hfd_map.get(right)
        if lv is None or rv is None or not np.isfinite(lv) or not np.isfinite(rv):
            continue
        rows.append({
            "pair": f"{left}-{right}",
            "left_channel": left,
            "right_channel": right,
            "left_hfd": float(lv),
            "right_hfd": float(rv),
            "asymmetry_left_minus_right": float(lv - rv),
        })
    return rows


def _higuchi_fractal_dimension(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "fast").lower()
    if mode not in HFD_MODE_CONFIGS:
        raise ValueError(f"Unknown Higuchi mode {mode!r}; use normal, fast, or ultrafast")
    cfg = dict(HFD_MODE_CONFIGS[mode])
    k_max = _safe_int(params.get("k_max"), int(cfg["k_max"]))
    rec = _load_hfd_segment(recording_dir, params, cfg)
    signal = np.asarray(rec["signal"], dtype=np.float64)
    channels = list(rec["channels"])
    if signal.ndim != 2 or signal.shape[1] != len(channels):
        raise ValueError(f"Loaded segment shape/channel mismatch: {signal.shape}, {len(channels)} channel labels")

    rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for ch_idx, ch_name in enumerate(channels):
        x = signal[:, ch_idx].astype(float)
        x = (x - np.mean(x)) / (np.std(x) + 1e-12)
        hfd, k_vals, lk = _higuchi_fd_curve(x, k_max=k_max)
        diag = _higuchi_fit_diagnostics(k_vals, lk)
        row = {"channel": ch_name, "channel_index": ch_idx, "higuchi_fd": _safe_float(hfd)}
        rows.append(row)
        fit_row = {
            "channel": ch_name,
            "channel_index": ch_idx,
            "higuchi_fd": _safe_float(diag["hfd"]),
            "fit_slope": _safe_float(diag["slope"]),
            "fit_intercept": _safe_float(diag["intercept"]),
            "fit_r2": _safe_float(diag["r2"]),
            "fit_rmse": _safe_float(diag["rmse"]),
        }
        fit_rows.append(fit_row)
        if k_vals is not None and lk is not None:
            mask = np.isfinite(lk) & (lk > 0)
            log_k = np.log(k_vals[mask]).tolist()
            log_l = np.log(lk[mask]).tolist()
            fit_log_l = []
            if np.isfinite(diag["slope"]) and np.isfinite(diag["intercept"]):
                fit_log_l = (diag["slope"] * np.asarray(log_k) + diag["intercept"]).tolist()
            curves.append({
                "channel": ch_name,
                "channel_index": ch_idx,
                "k_vals": k_vals.tolist(),
                "Lk": lk.tolist(),
                "log_k": log_k,
                "log_Lk": log_l,
                "fit_log_Lk": fit_log_l,
                "higuchi_fd": _safe_float(hfd),
                "fit_r2": _safe_float(diag["r2"]),
            })

    rows = sorted(rows, key=lambda r: (r["higuchi_fd"] is None, -(r["higuchi_fd"] or -1e9)))
    hfd_map = {row["channel"]: float(row["higuchi_fd"]) for row in rows if row["higuchi_fd"] is not None}
    values = np.asarray(list(hfd_map.values()), dtype=float)
    region_rows, region_points = _hfd_regional_summary(channels, hfd_map)
    asym_rows = _hfd_asymmetry(hfd_map)

    rolling_payload: dict[str, Any] = {"enabled": False, "times_sec": [], "channels": channels, "matrix": []}
    temporal_rows: list[dict[str, Any]] = []
    if _hfd_bool(params.get("rolling"), True):
        fs_eff = float(rec["sampling_rate_hz"] or 1.0)
        window_sec = _positive_float(params.get("rolling_window_sec"), float(cfg["rolling_window_sec"]))
        step_sec = _positive_float(params.get("rolling_step_sec"), float(cfg["rolling_step_sec"]))
        max_windows = _safe_int(params.get("rolling_max_windows"), int(cfg["rolling_max_windows"]))
        window_samples = max(int(window_sec * fs_eff), k_max + 5)
        step_samples = max(int(step_sec * fs_eff), 1)
        possible = np.arange(0, max(1, signal.shape[0] - window_samples + 1), step_samples, dtype=int)
        if possible.size > max_windows:
            possible = possible[np.linspace(0, possible.size - 1, max_windows).astype(int)]
        rolling = np.full((len(channels), len(possible)), np.nan, dtype=float)
        for w_idx, s0 in enumerate(possible):
            s1 = int(s0 + window_samples)
            for ch_idx in range(len(channels)):
                xw = signal[s0:s1, ch_idx].astype(float)
                xw = (xw - np.mean(xw)) / (np.std(xw) + 1e-12)
                hfd_w, _, _ = _higuchi_fd_curve(xw, k_max=k_max)
                rolling[ch_idx, w_idx] = hfd_w
        times = (possible / fs_eff).astype(float).tolist() if fs_eff > 0 else [float(x) for x in possible]
        rolling_payload = {
            "enabled": True,
            "window_sec": window_sec,
            "step_sec": step_sec,
            "times_sec": times,
            "channels": channels,
            "matrix": np.round(rolling, 6).tolist(),
        }
        temporal_std = np.nanstd(rolling, axis=1) if rolling.size else np.full(len(channels), np.nan)
        temporal_rows = sorted([
            {"channel": ch, "channel_index": i, "rolling_hfd_std": _safe_float(temporal_std[i]), "mean_hfd": _safe_float(hfd_map.get(ch))}
            for i, ch in enumerate(channels)
        ], key=lambda r: (r["rolling_hfd_std"] is None, -(r["rolling_hfd_std"] or -1e9)))

    fit_map = {row["channel"]: row for row in fit_rows}
    complexity_rows = []
    for row in temporal_rows:
        ch = row["channel"]
        fit = fit_map.get(ch, {})
        complexity_rows.append({
            "channel": ch,
            "mean_hfd": row.get("mean_hfd"),
            "rolling_hfd_std": row.get("rolling_hfd_std"),
            "fit_r2": fit.get("fit_r2"),
            "fit_rmse": fit.get("fit_rmse"),
        })

    out_dir = Path(rec["root"]) / "advanced_methods" / "higuchi_fractal_dimension"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    try:
        summary_df = pd.DataFrame(rows)
        fit_df = pd.DataFrame(fit_rows)
        region_df = pd.DataFrame(region_rows)
        asym_df = pd.DataFrame(asym_rows)
        temporal_df = pd.DataFrame(temporal_rows)
        summary_path = out_dir / "higuchi_fd_summary.csv"
        fit_path = out_dir / "higuchi_fd_fit_diagnostics.csv"
        summary_df.to_csv(summary_path, index=False)
        fit_df.to_csv(fit_path, index=False)
        output_paths["summary_csv"] = str(summary_path)
        output_paths["fit_diagnostics_csv"] = str(fit_path)
        if region_rows:
            p = out_dir / "regional_higuchi_fd_summary.csv"
            region_df.to_csv(p, index=False)
            output_paths["regional_summary_csv"] = str(p)
        if asym_rows:
            p = out_dir / "left_right_higuchi_fd_asymmetry.csv"
            asym_df.to_csv(p, index=False)
            output_paths["asymmetry_csv"] = str(p)
        if temporal_rows:
            p = out_dir / "rolling_higuchi_fd_temporal_variability.csv"
            temporal_df.to_csv(p, index=False)
            output_paths["temporal_variability_csv"] = str(p)
        if curves:
            npz = out_dir / "higuchi_fd_curves.npz"
            save_dict: dict[str, np.ndarray] = {}
            for c in curves:
                name = str(c["channel"]).replace("/", "_")
                save_dict[f"{name}_k_vals"] = np.asarray(c["k_vals"], dtype=float)
                save_dict[f"{name}_Lk"] = np.asarray(c["Lk"], dtype=float)
            if save_dict:
                np.savez(npz, **save_dict)
                output_paths["curves_npz"] = str(npz)
    except Exception as exc:
        output_paths["save_warning"] = repr(exc)

    summary = {
        "mode": mode,
        "k_max": k_max,
        "n_channels": len(channels),
        "analysis_samples": int(signal.shape[0]),
        "sampling_rate_analysis_hz": _safe_float(rec["sampling_rate_hz"]),
        "duration_sec": _safe_float(signal.shape[0] / float(rec["sampling_rate_hz"] or 1.0)),
        "segment_start_sec": _safe_float(rec["segment_start_sec"]),
        "segment_end_sec": _safe_float(rec["segment_end_sec"]),
        "mean_hfd": _safe_float(np.nanmean(values)) if values.size else None,
        "median_hfd": _safe_float(np.nanmedian(values)) if values.size else None,
        "min_hfd": _safe_float(np.nanmin(values)) if values.size else None,
        "max_hfd": _safe_float(np.nanmax(values)) if values.size else None,
        "top_channel": rows[0]["channel"] if rows else None,
        "top_channel_hfd": rows[0]["higuchi_fd"] if rows else None,
        "decimate": rec["decimate"],
    }

    return {
        "higuchi_fractal_dimension": {
            "summary": summary,
            "rows": rows,
            "fit_diagnostics": fit_rows,
            "curves": curves,
            "scalp_layout": {"points": _fallback_scalp_points(channels, hfd_map)},
            "regional_summary": region_rows,
            "regional_points": region_points,
            "asymmetry": asym_rows,
            "rolling": rolling_payload,
            "temporal_variability": temporal_rows,
            "complexity_instability": complexity_rows,
            "outputs": output_paths,
        }
    }


def _neuromouse_advanced_plots(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    """Build the full NeuroMouse advanced data objects for guaranteed plot rendering.

    This method is intentionally separate from the legacy browser workbench. It
    runs the backend NeuroMouse dataset builder and writes a concrete data.json
    containing polar_chronomap, kuramoto, channel_network, and tda fields, then
    points the user to the server-rendered /advanced-analysis page.
    """
    from neuro_importer_neuromouse.advanced_plot_renderer import availability
    from neuro_importer_neuromouse.static_dataset_builder import write_speedmouse_dataset

    root = Path(recording_dir).expanduser().resolve()
    output_dir = root / "neuromouse" / "advanced_plots"
    paths = write_speedmouse_dataset(
        root,
        output_dir,
        dataset_id=str(params.get("dataset_id") or root.name or "advanced_plots"),
        sampling_rate=params.get("sampling_rate") if params.get("sampling_rate") not in (None, "") else None,
        max_analysis_samples=_safe_int(params.get("max_analysis_samples"), 240000),
        max_windows=_safe_int(params.get("max_windows"), 600),
    )
    data_path = Path(paths["data_json"])
    data = json.loads(data_path.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    return {
        "neuromouse_advanced_plots": {
            "summary": {
                "data_json": str(data_path),
                "plot_page_url": "/advanced-analysis",
                "dataset_id": meta.get("dataset_id"),
                "n_channels": meta.get("n_channels"),
                "sampling_rate_hz": meta.get("sampling_rate_hz"),
                "advanced_plots_generated": True,
            },
            "availability": availability(data),
            "data_json": str(data_path),
            "plot_page_url": "/advanced-analysis",
            "open_in_neuromouse_url": "/neuromouse-latest/",
        }
    }



# ---- v0.11.14 Embedded attractor fractal-dimension analysis ----
# Adapted from the user's embedded_fractal_dimension_research_fast script.
# This native method can either use precomputed 2D..10D embedding .npy files
# when they exist, or generate delay embeddings directly from canonical
# signal.npy so the analysis works from a normal converted recording.

EMBEDDED_FD_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {
        "embedding_dims": [2, 3, 4, 6, 8, 10],
        "max_points_per_embedding": 4000,
        "corr_subsample_n": 800,
        "corr_n_pairs": 25000,
        "corr_n_r": 16,
        "corr_cmin": 5e-4,
        "corr_cmax": 2e-1,
        "box_subsample_n": 2500,
        "box_eps_count": 8,
        "diagnostic_top_n": 6,
        "max_analysis_samples": 90000,
    },
    "fast": {
        "embedding_dims": list(range(2, 11)),
        "max_points_per_embedding": 7000,
        "corr_subsample_n": 1200,
        "corr_n_pairs": 60000,
        "corr_n_r": 20,
        "corr_cmin": 3e-4,
        "corr_cmax": 2e-1,
        "box_subsample_n": 4000,
        "box_eps_count": 10,
        "diagnostic_top_n": 9,
        "max_analysis_samples": 120000,
    },
    "balanced": {
        "embedding_dims": list(range(2, 11)),
        "max_points_per_embedding": 15000,
        "corr_subsample_n": 2500,
        "corr_n_pairs": 200000,
        "corr_n_r": 28,
        "corr_cmin": 1e-4,
        "corr_cmax": 1e-1,
        "box_subsample_n": 8000,
        "box_eps_count": 12,
        "diagnostic_top_n": 12,
        "max_analysis_samples": 180000,
    },
    "full": {
        "embedding_dims": list(range(2, 11)),
        "max_points_per_embedding": 25000,
        "corr_subsample_n": 5000,
        "corr_n_pairs": 500000,
        "corr_n_r": 36,
        "corr_cmin": 5e-5,
        "corr_cmax": 2e-1,
        "box_subsample_n": 15000,
        "box_eps_count": 16,
        "diagnostic_top_n": 16,
        "max_analysis_samples": 240000,
    },
}


def _efd_standardize_cols(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    col_mean = np.nanmean(x, axis=0, keepdims=True)
    bad = np.where(~np.isfinite(x))
    if bad[0].size:
        x[bad] = np.take(col_mean, bad[1])
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True) + 1e-12
    return (x - mu) / sd


def _efd_rescale_to_unit_cube(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = np.min(x, axis=0, keepdims=True)
    xmax = np.max(x, axis=0, keepdims=True)
    rng = xmax - xmin
    rng[rng < 1e-12] = 1.0
    return (x - xmin) / rng


def _efd_subsample_rows(x: np.ndarray, max_points: int | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    n = len(x)
    if max_points is None or n <= max_points:
        return x, np.arange(n)
    idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
    return x[idx], idx


def _efd_linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray | None]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan, np.array([np.nan, np.nan]), None
    xv = x[mask]
    yv = y[mask]
    coef = np.polyfit(xv, yv, 1)
    yhat = np.polyval(coef, xv)
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(r2), coef, yhat


def _efd_box_counting_dimension(x: np.ndarray, eps_vals: np.ndarray, max_points: int | None, rng: np.random.Generator) -> dict[str, Any]:
    x, _ = _efd_subsample_rows(np.asarray(x, dtype=float), max_points, rng)
    if x.ndim != 2 or len(x) < 10:
        return {"dimension": np.nan, "inv_eps": [], "counts": [], "logx_fit": [], "logy_fit": [], "fit_line": [], "r2": np.nan, "n_fit": 0}
    x = _efd_rescale_to_unit_cube(_efd_standardize_cols(x))
    counts: list[float] = []
    inv_eps: list[float] = []
    for eps in eps_vals:
        if not np.isfinite(eps) or eps <= 0:
            continue
        bins = np.floor(x / eps).astype(np.int64)
        bins = np.clip(bins, 0, int(np.ceil(1 / eps)) + 2)
        n_boxes = int(len(np.unique(bins, axis=0)))
        if n_boxes > 0:
            counts.append(float(n_boxes))
            inv_eps.append(float(1.0 / eps))
    counts_arr = np.asarray(counts, dtype=float)
    inv_arr = np.asarray(inv_eps, dtype=float)
    mask = np.isfinite(counts_arr) & (counts_arr > 0) & np.isfinite(inv_arr) & (inv_arr > 0)
    if np.sum(mask) < 3:
        return {"dimension": np.nan, "inv_eps": inv_arr, "counts": counts_arr, "logx_fit": [], "logy_fit": [], "fit_line": [], "r2": np.nan, "n_fit": int(np.sum(mask))}
    logx = np.log(inv_arr[mask])
    logy = np.log(counts_arr[mask])
    r2, coef, fit_line = _efd_linear_fit_r2(logx, logy)
    return {
        "dimension": float(coef[0]),
        "inv_eps": inv_arr[mask],
        "counts": counts_arr[mask],
        "logx_fit": logx,
        "logy_fit": logy,
        "fit_line": fit_line if fit_line is not None else [],
        "r2": r2,
        "n_fit": int(np.sum(mask)),
    }


def _efd_correlation_dimension_sampled(
    x: np.ndarray,
    *,
    subsample_n: int,
    n_pairs: int,
    n_r: int,
    cmin: float,
    cmax: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    x = _efd_standardize_cols(np.asarray(x, dtype=float))
    n = len(x)
    empty = {"dimension": np.nan, "rs": [], "Cs": [], "logr_fit": [], "logC_fit": [], "fit_line": [], "n_fit": 0, "r2": np.nan, "n_used": int(n), "n_pairs_used": 0}
    if n < 50:
        return empty
    if n > subsample_n:
        idx = np.sort(rng.choice(n, size=int(subsample_n), replace=False))
        x = x[idx]
        n = len(x)
    n_pairs_use = max(1000, int(n_pairs))
    i = rng.integers(0, n, size=n_pairs_use)
    j = rng.integers(0, n, size=n_pairs_use)
    keep = i != j
    i = i[keep]
    j = j[keep]
    if len(i) < 1000:
        out = dict(empty)
        out.update({"n_used": int(n), "n_pairs_used": int(len(i))})
        return out
    d = np.linalg.norm(x[i] - x[j], axis=1)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 1000:
        out = dict(empty)
        out.update({"n_used": int(n), "n_pairs_used": int(len(d))})
        return out
    qlo = float(np.quantile(d, 0.01))
    qhi = float(np.quantile(d, 0.50))
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        out = dict(empty)
        out.update({"n_used": int(n), "n_pairs_used": int(len(d))})
        return out
    rs = np.logspace(np.log10(qlo), np.log10(qhi), int(n_r))
    d_sorted = np.sort(d)
    cs = np.searchsorted(d_sorted, rs, side="right") / len(d_sorted)
    mask = np.isfinite(cs) & (cs > cmin) & (cs < cmax) & np.isfinite(rs) & (rs > 0)
    n_fit = int(np.sum(mask))
    if n_fit < 4:
        return {"dimension": np.nan, "rs": rs, "Cs": cs, "logr_fit": [], "logC_fit": [], "fit_line": [], "n_fit": n_fit, "r2": np.nan, "n_used": int(n), "n_pairs_used": int(len(d))}
    log_r = np.log(rs[mask])
    log_c = np.log(cs[mask])
    r2, coef, fit_line = _efd_linear_fit_r2(log_r, log_c)
    return {
        "dimension": float(coef[0]),
        "rs": rs,
        "Cs": cs,
        "logr_fit": log_r,
        "logC_fit": log_c,
        "fit_line": fit_line if fit_line is not None else [],
        "n_fit": n_fit,
        "r2": r2,
        "n_used": int(n),
        "n_pairs_used": int(len(d)),
    }


def _efd_embedding_file(root: Path, channel: str, emb_dim: int) -> Path | None:
    candidates: list[Path] = []
    base = root / "embedding_data"
    if emb_dim == 2:
        candidates.append(base / "2dembedding_data" / f"2dembedded_{channel}.npy")
    elif emb_dim == 3:
        candidates.append(base / "3dembedding_data" / f"3dembedded_{channel}.npy")
    elif 4 <= emb_dim <= 10:
        candidates.append(base / "embeddings_4to10" / f"{emb_dim}dembedded_{channel}.npy")
    candidates.append(root / f"{emb_dim}dembedded_{channel}.npy")
    for c in candidates:
        if c.exists():
            return c
    return None


def _efd_delay_embedding_1d(x: np.ndarray, emb_dim: int, tau: int, max_points: int | None, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    tau = max(1, int(tau))
    n_vec = len(x) - (int(emb_dim) - 1) * tau
    if n_vec <= max(50, int(emb_dim) + 5):
        return np.empty((0, int(emb_dim)), dtype=float)
    if max_points is not None and n_vec > int(max_points):
        starts = np.sort(rng.choice(n_vec, size=int(max_points), replace=False))
    else:
        starts = np.arange(n_vec)
    return np.column_stack([x[starts + lag * tau] for lag in range(int(emb_dim))])


def _efd_parse_dims(value: Any, default: list[int]) -> list[int]:
    if value in (None, ""):
        return default
    if isinstance(value, (list, tuple)):
        vals = value
    else:
        vals = str(value).replace(";", ",").split(",")
    out: list[int] = []
    for raw in vals:
        try:
            iv = int(str(raw).strip())
            if 2 <= iv <= 20 and iv not in out:
                out.append(iv)
        except Exception:
            pass
    return out or default


def _efd_add_derived_metrics(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()
    for col in ("corr_dim_d2", "boxcount_fd", "emb_dim"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["d2_minus_boxfd"] = df["corr_dim_d2"] - df["boxcount_fd"]
    df["d2_over_emb_dim"] = df["corr_dim_d2"] / df["emb_dim"]
    df["boxfd_over_emb_dim"] = df["boxcount_fd"] / df["emb_dim"]
    out_parts = []
    for _, sub in df.groupby("channel", sort=False):
        sub = sub.sort_values("emb_dim").copy()
        sub["corr_dim_d2_increment"] = sub["corr_dim_d2"].diff()
        sub["boxcount_fd_increment"] = sub["boxcount_fd"].diff()
        out_parts.append(sub)
    df = pd.concat(out_parts, ignore_index=True) if out_parts else df
    summary_rows: list[dict[str, Any]] = []
    for channel, sub in df.groupby("channel"):
        sub = sub.sort_values("emb_dim")
        valid_d2 = sub[np.isfinite(sub["corr_dim_d2"])]
        valid_box = sub[np.isfinite(sub["boxcount_fd"])]
        row: dict[str, Any] = {"channel": channel}
        if len(valid_d2):
            last_d2 = valid_d2.iloc[-1]
            row.update({
                "max_available_emb_dim": int(last_d2["emb_dim"]),
                "terminal_corr_dim_d2": float(last_d2["corr_dim_d2"]),
                "terminal_corr_r2": float(last_d2["corr_fit_r2"]),
                "terminal_corr_fit_points": int(last_d2["corr_fit_points"]),
                "terminal_d2_over_m": float(last_d2["d2_over_emb_dim"]),
            })
        else:
            row.update({"max_available_emb_dim": None, "terminal_corr_dim_d2": np.nan, "terminal_corr_r2": np.nan, "terminal_corr_fit_points": None, "terminal_d2_over_m": np.nan})
        if len(valid_box):
            last_box = valid_box.iloc[-1]
            row.update({"terminal_boxcount_fd": float(last_box["boxcount_fd"]), "terminal_box_r2": float(last_box["box_fit_r2"])})
        else:
            row.update({"terminal_boxcount_fd": np.nan, "terminal_box_r2": np.nan})
        if len(valid_d2) >= 3:
            tail = valid_d2.tail(3)
            row["d2_tail_slope"] = float(np.polyfit(tail["emb_dim"], tail["corr_dim_d2"], 1)[0])
            row["d2_tail_std"] = float(np.nanstd(tail["corr_dim_d2"]))
        else:
            row["d2_tail_slope"] = np.nan
            row["d2_tail_std"] = np.nan
        if len(valid_box) >= 3:
            tail = valid_box.tail(3)
            row["box_tail_slope"] = float(np.polyfit(tail["emb_dim"], tail["boxcount_fd"], 1)[0])
            row["box_tail_std"] = float(np.nanstd(tail["boxcount_fd"]))
        else:
            row["box_tail_slope"] = np.nan
            row["box_tail_std"] = np.nan
        terminal_d2 = row.get("terminal_corr_dim_d2", np.nan)
        terminal_box = row.get("terminal_boxcount_fd", np.nan)
        tail_slope = row.get("d2_tail_slope", 0.0)
        terminal_r2 = row.get("terminal_corr_r2", np.nan)
        row["dimension_complexity_score"] = (
            0.55 * (terminal_d2 if np.isfinite(terminal_d2) else 0.0)
            + 0.25 * (terminal_box if np.isfinite(terminal_box) else 0.0)
            - 0.10 * abs(tail_slope if np.isfinite(tail_slope) else 0.0)
            + 0.10 * (terminal_r2 if np.isfinite(terminal_r2) else 0.0)
        )
        summary_rows.append(row)
    return df, pd.DataFrame(summary_rows)


def _embedded_fractal_dimension(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "fast").lower()
    if mode not in EMBEDDED_FD_MODE_CONFIGS:
        mode = "fast"
    cfg = dict(EMBEDDED_FD_MODE_CONFIGS[mode])
    seed = _safe_int(params.get("random_seed"), 0)
    rng = np.random.default_rng(seed)
    dims = _efd_parse_dims(params.get("embedding_dims"), cfg["embedding_dims"])
    max_channels_raw = params.get("max_channels")
    max_channels = None if max_channels_raw in (None, "", "all") else _safe_int(max_channels_raw, 0)
    max_analysis_samples = _safe_int(params.get("max_analysis_samples"), int(cfg["max_analysis_samples"]))
    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=max_analysis_samples)
    root = Path(rec["recording_dir"])
    signal = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels and max_channels > 0:
        signal = signal[:, :max_channels]
        channels = channels[:max_channels]
    tau_samples = params.get("tau_samples")
    if tau_samples not in (None, ""):
        tau = _safe_int(tau_samples, 1)
    else:
        tau_ms = _positive_float(params.get("tau_ms"), 10.0)
        tau = max(1, int(round(fs * tau_ms / 1000.0)))
    eps_count = _safe_int(params.get("box_eps_count"), int(cfg["box_eps_count"]))
    box_eps = np.logspace(np.log10(1 / 5), np.log10(1 / 60), eps_count)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[tuple[str, int], dict[str, Any]] = {}
    use_precomputed_count = 0
    generated_count = 0
    for ch_idx, ch_name in enumerate(channels):
        x = signal[:, ch_idx]
        for emb_dim in dims:
            try:
                source = "generated_delay_embedding"
                fpath = _efd_embedding_file(root, ch_name, emb_dim)
                if fpath is not None:
                    X = np.asarray(np.load(fpath), dtype=float)
                    if X.ndim != 2:
                        raise ValueError(f"bad embedding shape {X.shape}")
                    source = "precomputed_embedding"
                    use_precomputed_count += 1
                    X_used, _ = _efd_subsample_rows(X, int(cfg["max_points_per_embedding"]), rng)
                else:
                    X_used = _efd_delay_embedding_1d(x, emb_dim, tau, int(cfg["max_points_per_embedding"]), rng)
                    generated_count += 1
                n_points_original = int(len(X_used))
                if X_used.ndim != 2 or len(X_used) < 50:
                    raise ValueError("insufficient embedding points")
                box = _efd_box_counting_dimension(X_used, box_eps, int(cfg["box_subsample_n"]), rng)
                corr = _efd_correlation_dimension_sampled(
                    X_used,
                    subsample_n=int(cfg["corr_subsample_n"]),
                    n_pairs=int(cfg["corr_n_pairs"]),
                    n_r=int(cfg["corr_n_r"]),
                    cmin=float(cfg["corr_cmin"]),
                    cmax=float(cfg["corr_cmax"]),
                    rng=rng,
                )
                diagnostics[(ch_name, int(emb_dim))] = {"box": box, "corr": corr}
                rows.append({
                    "channel": ch_name,
                    "channel_index": ch_idx,
                    "emb_dim": int(emb_dim),
                    "n_points_original": n_points_original,
                    "n_points_used": int(len(X_used)),
                    "ambient_dim": int(X_used.shape[1]),
                    "boxcount_fd": _safe_float(box["dimension"]),
                    "box_fit_r2": _safe_float(box["r2"]),
                    "box_fit_points": int(box["n_fit"]),
                    "corr_dim_d2": _safe_float(corr["dimension"]),
                    "corr_fit_r2": _safe_float(corr["r2"]),
                    "corr_fit_points": int(corr["n_fit"]),
                    "corr_n_used": int(corr["n_used"]),
                    "corr_pairs_used": int(corr["n_pairs_used"]),
                    "embedding_source": source,
                    "status": "ok",
                    "error": "",
                })
            except Exception as exc:
                rows.append({
                    "channel": ch_name,
                    "channel_index": ch_idx,
                    "emb_dim": int(emb_dim),
                    "n_points_original": None,
                    "n_points_used": None,
                    "ambient_dim": int(emb_dim),
                    "boxcount_fd": None,
                    "box_fit_r2": None,
                    "box_fit_points": 0,
                    "corr_dim_d2": None,
                    "corr_fit_r2": None,
                    "corr_fit_points": 0,
                    "corr_n_used": 0,
                    "corr_pairs_used": 0,
                    "embedding_source": "error",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    results_df, channel_summary_df = _efd_add_derived_metrics(ok_rows)
    mean_df = pd.DataFrame()
    if not results_df.empty:
        mean_df = results_df.groupby("emb_dim").agg(
            boxcount_fd=("boxcount_fd", "mean"),
            corr_dim_d2=("corr_dim_d2", "mean"),
            box_fit_r2=("box_fit_r2", "mean"),
            corr_fit_r2=("corr_fit_r2", "mean"),
            n_rows=("channel", "count"),
        ).reset_index()
    out_dir = root / "advanced_methods" / "embedded_fractal_dimension"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    try:
        results_csv = out_dir / f"embedded_fractal_dimension_summary_{mode}.csv"
        pd.DataFrame(rows).to_csv(results_csv, index=False)
        outputs["summary_csv"] = str(results_csv)
        if not channel_summary_df.empty:
            p = out_dir / f"embedded_fractal_dimension_channel_summary_{mode}.csv"
            channel_summary_df.to_csv(p, index=False)
            outputs["channel_summary_csv"] = str(p)
        if not mean_df.empty:
            p = out_dir / f"embedded_fractal_dimension_mean_by_dim_{mode}.csv"
            mean_df.to_csv(p, index=False)
            outputs["mean_by_dimension_csv"] = str(p)
        txt = out_dir / f"embedded_fractal_dimension_summary_{mode}.txt"
        top = channel_summary_df.sort_values("dimension_complexity_score", ascending=False).head(20) if not channel_summary_df.empty else pd.DataFrame()
        txt.write_text(
            "Embedded attractor fractal-dimension analysis\n"
            "=============================================\n\n"
            f"mode: {mode}\nembedding_dims: {dims}\ntau_samples: {tau}\n"
            f"channels: {len(channels)}\nrows: {len(rows)}\nok_rows: {len(ok_rows)}\n\n"
            + (top.to_string(index=False) if not top.empty else "No successful rows."),
            encoding="utf-8",
        )
        outputs["summary_txt"] = str(txt)
    except Exception:
        pass
    diag_rows = []
    if not results_df.empty:
        top_diag = results_df[np.isfinite(results_df["corr_dim_d2"]) & np.isfinite(results_df["corr_fit_r2"])].sort_values(["corr_fit_r2", "corr_dim_d2"], ascending=False).head(int(cfg["diagnostic_top_n"]))
        for _, row in top_diag.iterrows():
            key = (str(row["channel"]), int(row["emb_dim"]))
            diag = diagnostics.get(key) or {}
            corr = diag.get("corr") or {}
            box = diag.get("box") or {}
            diag_rows.append({
                "channel": str(row["channel"]),
                "emb_dim": int(row["emb_dim"]),
                "corr_dim_d2": _safe_float(row["corr_dim_d2"]),
                "corr_fit_r2": _safe_float(row["corr_fit_r2"]),
                "boxcount_fd": _safe_float(row["boxcount_fd"]),
                "box_fit_r2": _safe_float(row["box_fit_r2"]),
                "corr_log_r": _json_safe(corr.get("logr_fit", [])),
                "corr_log_C": _json_safe(corr.get("logC_fit", [])),
                "corr_fit_line": _json_safe(corr.get("fit_line", [])),
                "box_log_inv_eps": _json_safe(box.get("logx_fit", [])),
                "box_log_counts": _json_safe(box.get("logy_fit", [])),
                "box_fit_line": _json_safe(box.get("fit_line", [])),
            })
    channel_summary = channel_summary_df.replace({np.nan: None}).to_dict(orient="records") if not channel_summary_df.empty else []
    mean_by_dimension = mean_df.replace({np.nan: None}).to_dict(orient="records") if not mean_df.empty else []
    summary = {
        "mode": mode,
        "embedding_dims": dims,
        "tau_samples": tau,
        "tau_ms_effective": _safe_float(1000.0 * tau / fs if fs > 0 else None),
        "n_channels": len(channels),
        "n_rows": len(rows),
        "ok_rows": len(ok_rows),
        "generated_embeddings": generated_count,
        "precomputed_embeddings": use_precomputed_count,
        "sampling_rate_analysis_hz": _safe_float(fs),
        "recording_dir": str(root),
        "top_channel": None,
        "top_complexity_score": None,
    }
    if channel_summary:
        top_channel = sorted(channel_summary, key=lambda r: (r.get("dimension_complexity_score") is None, -(r.get("dimension_complexity_score") or -1e99)))[0]
        summary["top_channel"] = top_channel.get("channel")
        summary["top_complexity_score"] = top_channel.get("dimension_complexity_score")
    return {
        "embedded_fractal_dimension": {
            "summary": summary,
            "rows": pd.DataFrame(rows).replace({np.nan: None}).to_dict(orient="records"),
            "channel_summary": channel_summary,
            "mean_by_dimension": mean_by_dimension,
            "diagnostics": diag_rows,
            "outputs": outputs,
        }
    }

# v0.11.17 Dimension saturation profiling.
# Adapted from the user's dimension_saturation_profiling notebook script.
def _dsp_first_consecutive_true(mask: Any, n_consecutive: int = 2) -> int | None:
    arr = np.asarray(mask, dtype=bool)
    if len(arr) < n_consecutive:
        return None
    run = 0
    for i, val in enumerate(arr):
        if val:
            run += 1
            if run >= n_consecutive:
                return i - n_consecutive + 1
        else:
            run = 0
    return None


def _dsp_tail_slope(x: Any, y: Any, tail_points: int = 3) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < max(2, int(tail_points)):
        return float("nan")
    x_tail = x_arr[-int(tail_points):]
    y_tail = y_arr[-int(tail_points):]
    return float(np.polyfit(x_tail, y_tail, 1)[0])


def _dsp_classify_saturation(m_sat: Any, max_m: int) -> str:
    try:
        if m_sat is None or pd.isna(m_sat):
            return "no_saturation"
        m = float(m_sat)
    except Exception:
        return "no_saturation"
    if m <= 5:
        return "early"
    if m <= 7:
        return "mid"
    if m <= max_m - 1:
        return "late"
    return "no_saturation"


def _find_latest_embedded_fd_csv(root: Path, mode: str | None = None) -> Path | None:
    base = root / "advanced_methods" / "embedded_fractal_dimension"
    if not base.exists():
        return None
    candidates: list[Path] = []
    if mode:
        candidates.extend(base.glob(f"embedded_fractal_dimension_summary_{mode}.csv"))
    candidates.extend(base.glob("embedded_fractal_dimension_summary_*.csv"))
    candidates = [p for p in candidates if p.exists() and p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _dimension_saturation_profiling(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    root = Path(recording_dir).expanduser().resolve()
    mode = str(params.get("mode") or params.get("embedded_mode") or "fast").lower()
    use_metric = str(params.get("use_metric") or "corr_dim_d2")
    abs_delta_thresh = _positive_float(params.get("abs_delta_thresh"), 0.12)
    rel_delta_thresh = _positive_float(params.get("rel_delta_thresh"), 0.04)
    n_consecutive = _safe_int(params.get("n_consecutive"), 2)
    tail_points = _safe_int(params.get("tail_points"), 3)
    run_embedded_first = str(params.get("run_embedded_if_missing", "true")).lower() not in ("false", "0", "no")

    source_csv_raw = str(params.get("source_csv") or "").strip()
    source_csv = Path(source_csv_raw).expanduser() if source_csv_raw else _find_latest_embedded_fd_csv(root, mode)
    embedded_result_payload: dict[str, Any] | None = None

    if (source_csv is None or not source_csv.exists()) and run_embedded_first:
        embedded_params = {
            "mode": mode,
            "embedding_dims": params.get("embedding_dims") or "2,3,4,5,6,7,8,9,10",
            "tau_ms": params.get("tau_ms", 10.0),
            "tau_samples": params.get("tau_samples", ""),
            "max_channels": params.get("max_channels", ""),
            "max_analysis_samples": params.get("max_analysis_samples", 120000),
            "sampling_rate": params.get("sampling_rate", ""),
            "random_seed": params.get("random_seed", 0),
        }
        embedded_result_payload = _embedded_fractal_dimension(root, embedded_params).get("embedded_fractal_dimension", {})
        output_csv = embedded_result_payload.get("outputs", {}).get("summary_csv") if isinstance(embedded_result_payload, dict) else None
        source_csv = Path(output_csv) if output_csv else _find_latest_embedded_fd_csv(root, mode)

    if source_csv is None or not source_csv.exists():
        raise FileNotFoundError("No embedded fractal-dimension summary CSV found. Run Embedded FD first or leave run_embedded_if_missing enabled.")

    df = pd.read_csv(source_csv)
    if use_metric not in df.columns:
        raise ValueError(f"Column '{use_metric}' not found in {source_csv}")
    if "channel" not in df.columns or "emb_dim" not in df.columns:
        raise ValueError(f"Expected channel and emb_dim columns in {source_csv}")

    df = df[["channel", "emb_dim", use_metric]].copy().rename(columns={use_metric: "D"})
    df["emb_dim"] = pd.to_numeric(df["emb_dim"], errors="coerce")
    df["D"] = pd.to_numeric(df["D"], errors="coerce")
    df = df[np.isfinite(df["emb_dim"]) & np.isfinite(df["D"])].sort_values(["channel", "emb_dim"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No finite embedded-dimension rows were available for saturation profiling.")

    channels = list(dict.fromkeys(df["channel"].astype(str).tolist()))
    max_m_global = int(np.nanmax(df["emb_dim"].to_numpy(dtype=float)))
    summary_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for ch in channels:
        sub = df[df["channel"].astype(str) == str(ch)].sort_values("emb_dim").copy()
        if len(sub) < 3:
            continue
        m = sub["emb_dim"].to_numpy(dtype=float)
        d2 = sub["D"].to_numpy(dtype=float)
        delta = np.diff(d2)
        delta_m = m[1:]
        rel_delta = delta / np.maximum(np.abs(d2[:-1]), 1e-12)
        abs_mask = np.abs(delta) <= abs_delta_thresh
        abs_idx = _dsp_first_consecutive_true(abs_mask, n_consecutive=n_consecutive)
        m_sat_abs = float(delta_m[abs_idx]) if abs_idx is not None else np.nan
        rel_mask = np.abs(rel_delta) <= rel_delta_thresh
        rel_idx = _dsp_first_consecutive_true(rel_mask, n_consecutive=n_consecutive)
        m_sat_rel = float(delta_m[rel_idx]) if rel_idx is not None else np.nan
        slope_tail = _dsp_tail_slope(m, d2, tail_points=tail_points)

        def _plateau(m_sat: float) -> tuple[float, float]:
            if pd.isna(m_sat):
                return float("nan"), float("nan")
            idx = np.where(m >= m_sat)[0]
            if len(idx) == 0:
                return float("nan"), float("nan")
            vals = d2[idx[0]:]
            return float(np.mean(vals)), float(np.std(vals))

        plateau_mean_abs, plateau_std_abs = _plateau(m_sat_abs)
        plateau_mean_rel, plateau_std_rel = _plateau(m_sat_rel)
        summary_rows.append({
            "channel": str(ch),
            "m_min": _safe_float(np.min(m)),
            "m_max": _safe_float(np.max(m)),
            "D2_m2": _safe_float(d2[0]) if len(d2) else None,
            "D2_mmax": _safe_float(d2[-1]) if len(d2) else None,
            "D2_total_gain": _safe_float(d2[-1] - d2[0]) if len(d2) > 1 else None,
            "mean_delta_D2": _safe_float(np.mean(delta)) if len(delta) else None,
            "std_delta_D2": _safe_float(np.std(delta)) if len(delta) else None,
            "tail_slope_last_points": _safe_float(slope_tail),
            "m_sat_abs": _safe_float(m_sat_abs),
            "m_sat_rel": _safe_float(m_sat_rel),
            "plateau_mean_abs": _safe_float(plateau_mean_abs),
            "plateau_std_abs": _safe_float(plateau_std_abs),
            "plateau_mean_rel": _safe_float(plateau_mean_rel),
            "plateau_std_rel": _safe_float(plateau_std_rel),
            "class_abs": _dsp_classify_saturation(m_sat_abs, max_m_global),
            "class_rel": _dsp_classify_saturation(m_sat_rel, max_m_global),
        })
        for mm, dd, rr in zip(delta_m, delta, rel_delta):
            delta_rows.append({
                "channel": str(ch),
                "emb_dim_to": _safe_float(mm),
                "delta_D2": _safe_float(dd),
                "rel_delta_D2": _safe_float(rr),
                "small_abs": bool(abs(dd) <= abs_delta_thresh),
                "small_rel": bool(abs(rr) <= rel_delta_thresh),
            })

    summary_df = pd.DataFrame(summary_rows)
    delta_df = pd.DataFrame(delta_rows)
    if summary_df.empty:
        raise RuntimeError("No channels had at least three embedding-dimension points for saturation profiling.")
    summary_df = summary_df.sort_values(["class_abs", "m_sat_abs", "tail_slope_last_points"], na_position="last")
    delta_df = delta_df.sort_values(["channel", "emb_dim_to"]) if not delta_df.empty else delta_df

    out_dir = root / "advanced_methods" / "dimension_saturation_profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "dimension_saturation_summary.csv"
    delta_csv = out_dir / "dimension_saturation_deltas.csv"
    txt_path = out_dir / "dimension_saturation_summary.txt"
    summary_df.to_csv(summary_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    class_counts_series = summary_df["class_abs"].value_counts()
    class_counts = [{"class_abs": str(k), "count": int(v)} for k, v in class_counts_series.items()]
    strongest = summary_df.sort_values("tail_slope_last_points", ascending=False).head(10)
    earliest = summary_df.dropna(subset=["m_sat_abs"]).sort_values("m_sat_abs", ascending=True).head(10)
    txt_path.write_text(
        "Dimension-saturation profiling per channel\n"
        "=========================================\n\n"
        f"Source CSV: {source_csv}\nMetric used: {use_metric}\n"
        f"Absolute delta threshold: {abs_delta_thresh}\nRelative delta threshold: {rel_delta_thresh}\n"
        f"Consecutive small increments required: {n_consecutive}\nTail points for slope: {tail_points}\n\n"
        "Class counts (absolute threshold):\n"
        + "".join(f"  {r['class_abs']}: {r['count']}\n" for r in class_counts)
        + "\nTop channels with strongest continuing growth:\n"
        + "".join(f"  {row.channel}: tail_slope={row.tail_slope_last_points:.6f}, m_sat_abs={row.m_sat_abs}, D2_gain={row.D2_total_gain:.6f}\n" for row in strongest.itertuples())
        + "\nTop channels with earliest apparent saturation:\n"
        + "".join(f"  {row.channel}: m_sat_abs={row.m_sat_abs}, tail_slope={row.tail_slope_last_points:.6f}, plateau_mean_abs={row.plateau_mean_abs:.6f}\n" for row in earliest.itertuples()),
        encoding="utf-8",
    )
    outputs = {"summary_csv": str(summary_csv), "delta_csv": str(delta_csv), "summary_txt": str(txt_path)}

    profiles = df.rename(columns={"D": use_metric}).replace({np.nan: None}).to_dict(orient="records")
    heatmap = []
    if not delta_df.empty:
        pivot = delta_df.pivot_table(index="channel", columns="emb_dim_to", values="delta_D2", aggfunc="mean")
        for channel, row in pivot.iterrows():
            heatmap.append({"channel": str(channel), "values": [{"emb_dim_to": _safe_float(k), "delta_D2": _safe_float(v)} for k, v in row.items()]})
    summary = {
        "mode": mode,
        "use_metric": use_metric,
        "source_csv": str(source_csv),
        "embedded_fd_was_run": embedded_result_payload is not None,
        "abs_delta_thresh": abs_delta_thresh,
        "rel_delta_thresh": rel_delta_thresh,
        "n_consecutive": n_consecutive,
        "tail_points": tail_points,
        "max_embedding_dim": max_m_global,
        "n_channels": int(summary_df["channel"].nunique()),
        "n_profile_rows": int(len(df)),
        "n_delta_rows": int(len(delta_df)),
        "class_counts": class_counts,
        "strongest_growth_channel": str(strongest.iloc[0]["channel"]) if len(strongest) else None,
        "earliest_saturation_channel": str(earliest.iloc[0]["channel"]) if len(earliest) else None,
        "recording_dir": str(root),
    }
    return {
        "dimension_saturation_profiling": {
            "summary": summary,
            "profiles": profiles,
            "summary_rows": summary_df.replace({np.nan: None}).to_dict(orient="records"),
            "delta_rows": delta_df.replace({np.nan: None}).to_dict(orient="records"),
            "delta_heatmap": heatmap,
            "class_counts": class_counts,
            "outputs": outputs,
            "embedded_fd_summary": embedded_result_payload.get("summary") if isinstance(embedded_result_payload, dict) else None,
        }
    }


# v0.11.18 Katz fractal-dimension analysis for embedded trajectories.
# Adapted from the user's Katz FD embedding notebook. It keeps the same
# black/cyan plotting semantics in the frontend and supports either
# precomputed embedding .npy files or generated delay embeddings.
def _katz_fd_trajectory(data: Any) -> float:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        n = int(len(x))
        if n < 2:
            return float("nan")
        step_dist = np.abs(np.diff(x))
        length = float(np.sum(step_dist))
        diameter = float(abs(x[-1] - x[0]))
    elif x.ndim == 2:
        n = int(x.shape[0])
        if n < 2:
            return float("nan")
        diffs = np.diff(x, axis=0)
        step_dist = np.sqrt(np.sum(diffs ** 2, axis=1))
        length = float(np.sum(step_dist))
        diameter = float(np.sqrt(np.sum((x[-1] - x[0]) ** 2)))
    else:
        raise ValueError(f"Unsupported array shape for Katz FD: {x.shape}")
    if length <= 0 or diameter <= 0:
        return float("nan")
    return float(np.log10(n) / (np.log10(diameter / (length + 1e-12)) + np.log10(n) + 1e-12))


def _katz_precomputed_embedding_paths(root: Path, channels: list[str], dims: list[int]) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    for channel in channels:
        for emb_dim in dims:
            p = _efd_embedding_file(root, channel, emb_dim)
            if p is not None:
                found[(channel, int(emb_dim))] = p
    return found


def _katz_fractal_dimension(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "fast").lower()
    if mode not in EMBEDDED_FD_MODE_CONFIGS:
        mode = "fast"
    cfg = dict(EMBEDDED_FD_MODE_CONFIGS[mode])
    seed = _safe_int(params.get("random_seed"), 0)
    rng = np.random.default_rng(seed)
    dims = _efd_parse_dims(params.get("embedding_dims"), cfg["embedding_dims"])
    max_channels_raw = params.get("max_channels")
    max_channels = None if max_channels_raw in (None, "", "all") else _safe_int(max_channels_raw, 0)
    max_analysis_samples = _safe_int(params.get("max_analysis_samples"), int(cfg["max_analysis_samples"]))
    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=max_analysis_samples)
    root = Path(rec["recording_dir"])
    signal = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels and max_channels > 0:
        signal = signal[:, :max_channels]
        channels = channels[:max_channels]

    tau_samples = params.get("tau_samples")
    if tau_samples not in (None, ""):
        tau = _safe_int(tau_samples, 1)
    else:
        tau_ms = _positive_float(params.get("tau_ms"), 10.0)
        tau = max(1, int(round(fs * tau_ms / 1000.0)))

    max_points = int(cfg["max_points_per_embedding"])
    precomputed = _katz_precomputed_embedding_paths(root, channels, dims)
    use_precomputed_count = 0
    generated_count = 0
    rows: list[dict[str, Any]] = []

    for ch_idx, ch_name in enumerate(channels):
        x = signal[:, ch_idx]
        for emb_dim in dims:
            rec_row = {
                "file": f"{int(emb_dim)}dembedded_{ch_name}.npy",
                "stem": f"{int(emb_dim)}dembedded_{ch_name}",
                "path": "",
                "source_dir": "generated_delay_embedding",
                "channel": ch_name,
                "channel_index": ch_idx,
                "embedding_dimension": int(emb_dim),
                "n_samples": None,
                "state_dim": int(emb_dim),
                "katz_fd": None,
                "status": "ok",
                "error": "",
                "embedding_source": "generated_delay_embedding",
            }
            try:
                p = precomputed.get((ch_name, int(emb_dim)))
                if p is not None:
                    X = np.asarray(np.load(p, allow_pickle=True), dtype=float)
                    if X.ndim != 2:
                        raise ValueError(f"bad embedding shape {X.shape}")
                    X_used, _ = _efd_subsample_rows(X, max_points, rng)
                    rec_row.update({
                        "path": str(p),
                        "source_dir": p.parent.name,
                        "embedding_source": "precomputed_embedding",
                    })
                    use_precomputed_count += 1
                else:
                    X_used = _efd_delay_embedding_1d(x, int(emb_dim), tau, max_points, rng)
                    generated_count += 1
                if X_used.ndim != 2 or len(X_used) < 2:
                    raise ValueError("insufficient embedding points")
                fd = _katz_fd_trajectory(X_used)
                rec_row.update({
                    "n_samples": int(X_used.shape[0]),
                    "state_dim": int(X_used.shape[1]),
                    "katz_fd": _safe_float(fd),
                })
            except Exception as exc:
                rec_row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "katz_fd": None})
            rows.append(rec_row)

    df = pd.DataFrame(rows)
    df_ok = df[(df["status"] == "ok") & np.isfinite(pd.to_numeric(df["katz_fd"], errors="coerce"))].copy()
    if df_ok.empty:
        raise ValueError("No valid Katz FD results were computed.")

    by_dim_df = (
        df_ok.groupby("embedding_dimension", dropna=True)
        .agg(
            n_records=("katz_fd", "count"),
            mean_katz_fd=("katz_fd", "mean"),
            std_katz_fd=("katz_fd", "std"),
            min_katz_fd=("katz_fd", "min"),
            max_katz_fd=("katz_fd", "max"),
        )
        .reset_index()
        .sort_values("embedding_dimension")
    )
    by_channel_df = (
        df_ok.groupby("channel", dropna=True)
        .agg(
            n_records=("katz_fd", "count"),
            mean_katz_fd=("katz_fd", "mean"),
            std_katz_fd=("katz_fd", "std"),
            min_katz_fd=("katz_fd", "min"),
            max_katz_fd=("katz_fd", "max"),
        )
        .reset_index()
        .sort_values("channel")
    )
    matrix_df = df_ok.pivot_table(index="channel", columns="embedding_dimension", values="katz_fd", aggfunc="mean").sort_index()

    out_dir = root / "advanced_methods" / "katz_fractal_dimension"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    results_csv = out_dir / "katz_fd_per_embedding.csv"
    by_dim_csv = out_dir / "katz_fd_by_dimension.csv"
    by_channel_csv = out_dir / "katz_fd_by_channel.csv"
    matrix_csv = out_dir / "katz_fd_channel_dimension_matrix.csv"
    summary_txt = out_dir / "katz_fd_summary.txt"
    df.to_csv(results_csv, index=False)
    by_dim_df.to_csv(by_dim_csv, index=False)
    by_channel_df.to_csv(by_channel_csv, index=False)
    matrix_df.to_csv(matrix_csv)
    outputs.update({
        "per_embedding_csv": str(results_csv),
        "by_dimension_csv": str(by_dim_csv),
        "by_channel_csv": str(by_channel_csv),
        "matrix_csv": str(matrix_csv),
        "summary_txt": str(summary_txt),
    })
    summary_txt.write_text(
        "Katz Fractal Dimension Summary\n"
        "=============================\n\n"
        f"mode: {mode}\n"
        f"recording_dir: {root}\n"
        f"embedding_dims: {dims}\n"
        f"tau_samples: {tau}\n"
        f"total_rows: {len(rows)}\n"
        f"successful_results: {len(df_ok)}\n"
        f"generated_embeddings: {generated_count}\n"
        f"precomputed_embeddings: {use_precomputed_count}\n\n"
        "Mean Katz FD by embedding dimension:\n"
        + "".join(
            f"  {int(row.embedding_dimension)}D: n={int(row.n_records)}, mean={row.mean_katz_fd:.6f}, std={(0.0 if pd.isna(row.std_katz_fd) else row.std_katz_fd):.6f}\n"
            for row in by_dim_df.itertuples()
        ),
        encoding="utf-8",
    )

    matrix_records = []
    for channel, row in matrix_df.iterrows():
        matrix_records.append({
            "channel": str(channel),
            "values": [{"embedding_dimension": _safe_float(dim), "katz_fd": _safe_float(val)} for dim, val in row.items()],
        })
    top = df_ok.sort_values("katz_fd", ascending=False).head(1)
    summary = {
        "mode": mode,
        "embedding_dims": dims,
        "tau_samples": tau,
        "tau_ms_effective": _safe_float(1000.0 * tau / fs if fs > 0 else None),
        "n_channels": len(channels),
        "n_rows": int(len(rows)),
        "ok_rows": int(len(df_ok)),
        "generated_embeddings": generated_count,
        "precomputed_embeddings": use_precomputed_count,
        "sampling_rate_analysis_hz": _safe_float(fs),
        "recording_dir": str(root),
        "top_embedding": str(top.iloc[0]["stem"]) if len(top) else None,
        "top_katz_fd": _safe_float(top.iloc[0]["katz_fd"]) if len(top) else None,
    }
    return {
        "katz_fractal_dimension": {
            "summary": summary,
            "rows": df.replace({np.nan: None}).to_dict(orient="records"),
            "by_dimension": by_dim_df.replace({np.nan: None}).to_dict(orient="records"),
            "by_channel": by_channel_df.replace({np.nan: None}).to_dict(orient="records"),
            "channel_dimension_matrix": matrix_records,
            "outputs": outputs,
        }
    }


# ---- v0.11.19 Manual wavelet Hurst exponent analysis ----
# Adapted from the user's manual wavelet Hurst notebook. This version reads
# canonical signal.npy from a converted recording and returns JSON-friendly
# arrays for the web GUI to render in the same black/cyan visual style.

WAVELET_HURST_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {"wavelet_name": "haar", "max_level": 5, "run_temporal_stability": True, "temporal_windows": 4, "scaling_gallery_top_n": 6, "raw_plot_max_points": 4000, "max_analysis_samples": 90000},
    "fast": {"wavelet_name": "db4", "max_level": 8, "run_temporal_stability": True, "temporal_windows": 8, "scaling_gallery_top_n": 9, "raw_plot_max_points": 12000, "max_analysis_samples": 120000},
    "balanced": {"wavelet_name": "db4", "max_level": 10, "run_temporal_stability": True, "temporal_windows": 14, "scaling_gallery_top_n": 12, "raw_plot_max_points": 24000, "max_analysis_samples": 180000},
    "full": {"wavelet_name": "db4", "max_level": None, "run_temporal_stability": True, "temporal_windows": 24, "scaling_gallery_top_n": 16, "raw_plot_max_points": 24000, "max_analysis_samples": 240000},
}

WH_LEFT_CHANNELS = {"Fp1", "F7", "F3", "FC5", "FC1", "M1", "T7", "C3", "CP5", "CP1", "P7", "P3", "O1"}
WH_RIGHT_CHANNELS = {"Fp2", "F8", "F4", "FC2", "FC6", "M2", "T8", "C4", "CP2", "CP6", "P8", "P4", "O2"}
WH_MIDLINE_CHANNELS = {"Fpz", "Fz", "Cz", "Pz", "POz", "Oz"}
WH_LR_PAIRS: list[tuple[str, str]] = [("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("FC5", "FC6"), ("FC1", "FC2"), ("M1", "M2"), ("T7", "T8"), ("C3", "C4"), ("CP5", "CP6"), ("CP1", "CP2"), ("P7", "P8"), ("P3", "P4"), ("O1", "O2")]
WH_ELECTRODE_XY: dict[str, tuple[float, float]] = {
    "Fp1": (-0.45, 1.00), "Fpz": (0.00, 1.05), "Fp2": (0.45, 1.00),
    "F7": (-0.95, 0.55), "F3": (-0.45, 0.55), "Fz": (0.00, 0.60), "F4": (0.45, 0.55), "F8": (0.95, 0.55),
    "FC5": (-0.70, 0.25), "FC1": (-0.25, 0.25), "FC2": (0.25, 0.25), "FC6": (0.70, 0.25),
    "M1": (-1.10, 0.00), "T7": (-0.95, 0.00), "C3": (-0.45, 0.00), "Cz": (0.00, 0.00), "C4": (0.45, 0.00), "T8": (0.95, 0.00), "M2": (1.10, 0.00),
    "CP5": (-0.70, -0.25), "CP1": (-0.25, -0.25), "CP2": (0.25, -0.25), "CP6": (0.70, -0.25),
    "P7": (-0.95, -0.55), "P3": (-0.45, -0.55), "Pz": (0.00, -0.60), "P4": (0.45, -0.55), "P8": (0.95, -0.55),
    "POz": (0.00, -0.82), "O1": (-0.35, -1.00), "Oz": (0.00, -1.05), "O2": (0.35, -1.00),
}


def _wh_safe_zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.where(np.isfinite(x), x, np.nan)
    if np.any(~np.isfinite(x)):
        med = np.nanmedian(x)
        if not np.isfinite(med):
            med = 0.0
        x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def _wh_get_wavelet_filters(name: str = "db4") -> tuple[np.ndarray, np.ndarray]:
    name = str(name).lower()
    if name == "haar":
        h = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=float)
    elif name == "db4":
        h = np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171888114, -0.027983769416859854, 0.6308807679298587, 0.7148465705529157, 0.2303778133088965], dtype=float)
    else:
        raise ValueError("Unsupported wavelet. Use 'haar' or 'db4'.")
    g = ((-1) ** np.arange(len(h))) * h[::-1]
    return h, g


def _wh_dwt_single_level_periodic(x: np.ndarray, h: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    L = len(h)
    if n < L:
        raise ValueError("Signal too short for one DWT level with chosen filter.")
    out_len = n // 2
    if out_len < 1:
        raise ValueError("Signal too short after decimation.")
    a = np.zeros(out_len, dtype=float)
    d = np.zeros(out_len, dtype=float)
    offsets = np.arange(L)
    for k in range(out_len):
        segment = x[(2 * k + offsets) % n]
        a[k] = np.sum(h * segment)
        d[k] = np.sum(g * segment)
    return a, d


def _wh_max_dwt_level(n: int, filter_len: int) -> int:
    level = 0
    current = int(n)
    while current >= filter_len and current >= 2:
        current //= 2
        level += 1
    return max(level, 0)


def _wh_wavedec_manual(x: np.ndarray, wavelet: str = "db4", level: int | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    h, g = _wh_get_wavelet_filters(wavelet)
    x = np.asarray(x, dtype=float).reshape(-1)
    if level is None:
        level = _wh_max_dwt_level(len(x), len(h))
    if int(level) < 2:
        raise ValueError("Need at least 2 levels for stable wavelet Hurst estimation.")
    a = x.copy()
    details: list[np.ndarray] = []
    for _ in range(int(level)):
        a, d = _wh_dwt_single_level_periodic(a, h, g)
        details.append(d)
        if len(a) < 2:
            break
    if len(details) < 2:
        raise ValueError("Too few valid detail levels after decomposition.")
    return a, details


def _wh_fit_line_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray | None]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan, None
    xv = x[mask]
    yv = y[mask]
    slope, intercept = np.polyfit(xv, yv, 1)
    yhat = slope * xv + intercept
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(slope), float(intercept), float(r2), yhat


def _wh_estimate_hurst_from_details(details: list[np.ndarray]) -> dict[str, Any]:
    n_levels = len(details)
    scales = np.array([2 ** j for j in range(1, n_levels + 1)], dtype=float)
    stdevs = np.array([np.std(np.asarray(d, dtype=float), ddof=1) for d in details], dtype=float)
    energies = np.array([np.mean(np.asarray(d, dtype=float) ** 2) for d in details], dtype=float)
    valid = np.isfinite(stdevs) & (stdevs > 0)
    if np.sum(valid) < 2:
        return {"hurst_exponent": np.nan, "fit_r2": np.nan, "scales": scales, "detail_stdevs": stdevs, "detail_energies": energies, "log2_scales_fit": np.array([]), "log2_stdevs_fit": np.array([]), "fit_line": np.array([]), "slope": np.nan, "intercept": np.nan}
    x = np.log2(scales[valid])
    y = np.log2(stdevs[valid])
    slope, intercept, r2, yhat = _wh_fit_line_r2(x, y)
    return {"hurst_exponent": float(slope), "fit_r2": float(r2), "scales": scales, "detail_stdevs": stdevs, "detail_energies": energies, "log2_scales_fit": x, "log2_stdevs_fit": y, "fit_line": yhat if yhat is not None else np.array([]), "slope": slope, "intercept": intercept}


def _wh_analyze_signal(signal: np.ndarray, *, wavelet: str, max_level: int | None) -> dict[str, Any]:
    x = _wh_safe_zscore_1d(signal)
    h, _ = _wh_get_wavelet_filters(wavelet)
    allowed = _wh_max_dwt_level(len(x), len(h))
    if allowed < 2:
        raise ValueError(f"Signal too short for stable wavelet Hurst estimate with wavelet={wavelet}.")
    level = allowed if max_level is None else min(int(max_level), allowed)
    if level < 2:
        raise ValueError("Need at least 2 valid levels.")
    _, details = _wh_wavedec_manual(x, wavelet=wavelet, level=level)
    scaling = _wh_estimate_hurst_from_details(details)
    total_detail_energy = float(np.sum(scaling["detail_energies"]))
    rel = np.asarray(scaling["detail_energies"], dtype=float) / (total_detail_energy + 1e-12)
    return {"hurst_exponent": scaling["hurst_exponent"], "fit_r2": scaling["fit_r2"], "n_levels": int(len(details)), "scales": scaling["scales"], "detail_stdevs": scaling["detail_stdevs"], "detail_energies": scaling["detail_energies"], "relative_detail_energies": rel, "log2_scales_fit": scaling["log2_scales_fit"], "log2_stdevs_fit": scaling["log2_stdevs_fit"], "fit_line": scaling["fit_line"], "slope": scaling["slope"], "intercept": scaling["intercept"], "level_allowed": int(allowed), "level_used": int(level)}


def _wh_region(channel: str) -> str:
    ch = str(channel)
    if ch.startswith("Fp"):
        return "frontal_polar"
    if ch.startswith("F") or ch.startswith("FC"):
        return "frontal"
    if ch.startswith("C") or ch.startswith("T") or ch.startswith("M"):
        return "central_temporal"
    if ch.startswith("CP") or ch.startswith("P"):
        return "parietal"
    if ch.startswith("PO") or ch.startswith("O"):
        return "occipital"
    return "unknown"


def _wh_hemisphere(channel: str) -> str:
    ch = str(channel)
    if ch in WH_LEFT_CHANNELS:
        return "left"
    if ch in WH_RIGHT_CHANNELS:
        return "right"
    if ch in WH_MIDLINE_CHANNELS:
        return "midline"
    return "unknown"


def _wh_downsample_for_plot(x: np.ndarray, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    if max_points is None or x.shape[0] <= int(max_points):
        idx = np.arange(x.shape[0])
        return x, idx
    step = int(math.ceil(x.shape[0] / int(max_points)))
    idx = np.arange(0, x.shape[0], step)
    return x[idx], idx


def _wh_temporal_hurst(x: np.ndarray, channels: list[str], *, fs: float, wavelet: str, max_level: int | None, n_windows: int) -> pd.DataFrame:
    n = int(x.shape[0])
    window_len = max(2048, n // max(int(n_windows), 1))
    rows: list[dict[str, Any]] = []
    for w in range(int(n_windows)):
        s = w * window_len
        e = min(n, s + window_len)
        if e - s < 2048:
            continue
        for i, channel in enumerate(channels):
            try:
                res = _wh_analyze_signal(x[s:e, i], wavelet=wavelet, max_level=max_level)
                rows.append({"window": int(w + 1), "channel": channel, "start_sample": int(s), "end_sample": int(e), "start_sec_relative": float(s / fs) if fs > 0 else None, "end_sec_relative": float(e / fs) if fs > 0 else None, "hurst_exponent": _safe_float(res["hurst_exponent"]), "fit_r2": _safe_float(res["fit_r2"]), "n_levels": int(res["n_levels"]), "status": "ok", "error": ""})
            except Exception as exc:
                rows.append({"window": int(w + 1), "channel": channel, "start_sample": int(s), "end_sample": int(e), "start_sec_relative": float(s / fs) if fs > 0 else None, "end_sec_relative": float(e / fs) if fs > 0 else None, "hurst_exponent": np.nan, "fit_r2": np.nan, "n_levels": np.nan, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def _wavelet_hurst_exponent(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    root = Path(recording_dir).expanduser().resolve()
    mode = str(params.get("mode") or params.get("FAST_OPTION") or "ultra").lower()
    if mode not in WAVELET_HURST_MODE_CONFIGS:
        mode = "ultra"
    cfg = dict(WAVELET_HURST_MODE_CONFIGS[mode])
    wavelet = str(params.get("wavelet") or cfg["wavelet_name"]).lower()
    max_level_param = params.get("max_level")
    max_level = cfg["max_level"] if max_level_param in (None, "") else _safe_int(max_level_param, cfg["max_level"] or 10)
    if max_level == 0:
        max_level = None
    temporal = _hfd_bool(params.get("temporal_stability"), bool(cfg["run_temporal_stability"]))
    temporal_windows = _safe_int(params.get("temporal_windows"), int(cfg["temporal_windows"]))
    max_channels = params.get("max_channels")
    max_channels_i = _safe_int(max_channels, 0) if max_channels not in (None, "") else 0
    raw_plot_max_points = _safe_int(params.get("raw_plot_max_points"), int(cfg["raw_plot_max_points"]))
    rec = _load_recording(root, sampling_rate=params.get("sampling_rate"), max_samples=params.get("max_analysis_samples") if params.get("max_analysis_samples") not in (None, "") else int(cfg["max_analysis_samples"]))
    x = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels_i and max_channels_i < len(channels):
        channels = channels[:max_channels_i]
        x = x[:, :max_channels_i]
    rows: list[dict[str, Any]] = []
    channel_results: dict[str, dict[str, Any]] = {}
    for i, channel in enumerate(channels):
        try:
            res = _wh_analyze_signal(x[:, i], wavelet=wavelet, max_level=max_level)
            channel_results[channel] = res
            h = _safe_float(res["hurst_exponent"])
            r2 = _safe_float(res["fit_r2"])
            rows.append({"channel": channel, "hurst_exponent": h, "fractal_dimension_proxy": _safe_float(2.0 - h) if h is not None else None, "fit_r2": r2, "n_levels": int(res["n_levels"]), "level_allowed": int(res["level_allowed"]), "level_used": int(res["level_used"]), "signal_length": int(x.shape[0]), "wavelet": wavelet, "mode": mode, "hemisphere": _wh_hemisphere(channel), "region": _wh_region(channel), "quality_weighted_hurst": _safe_float((h or 0.0) * (r2 or 0.0)) if h is not None and r2 is not None else None, "status": "ok", "error": ""})
        except Exception as exc:
            rows.append({"channel": channel, "hurst_exponent": None, "fractal_dimension_proxy": None, "fit_r2": None, "n_levels": None, "level_allowed": None, "level_used": None, "signal_length": int(x.shape[0]), "wavelet": wavelet, "mode": mode, "hemisphere": _wh_hemisphere(channel), "region": _wh_region(channel), "quality_weighted_hurst": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    results_df = pd.DataFrame(rows)
    temporal_df = _wh_temporal_hurst(x, channels, fs=fs, wavelet=wavelet, max_level=max_level, n_windows=temporal_windows) if temporal else pd.DataFrame()
    regional_summary = []
    if not results_df.empty:
        regional = results_df.copy()
        regional["hurst_exponent"] = pd.to_numeric(regional["hurst_exponent"], errors="coerce")
        regional["fit_r2"] = pd.to_numeric(regional["fit_r2"], errors="coerce")
        tmp = regional.groupby("region").agg(mean_hurst=("hurst_exponent", "mean"), std_hurst=("hurst_exponent", "std"), mean_r2=("fit_r2", "mean"), n=("channel", "count")).reset_index().sort_values("mean_hurst", ascending=True)
        regional_summary = tmp.to_dict(orient="records")
    hdict = {str(r["channel"]): r.get("hurst_exponent") for r in rows}
    asym_rows: list[dict[str, Any]] = []
    for left, right in WH_LR_PAIRS:
        lv = _safe_float(hdict.get(left)); rv = _safe_float(hdict.get(right))
        if lv is not None and rv is not None:
            asym_rows.append({"pair": f"{left}-{right}", "left": left, "right": right, "H_left": lv, "H_right": rv, "H_left_minus_right": float(lv - rv)})
    topographic_points = []
    for r in rows:
        ch = str(r["channel"]); hval = _safe_float(r.get("hurst_exponent"))
        if hval is not None and ch in WH_ELECTRODE_XY:
            xy = WH_ELECTRODE_XY[ch]
            topographic_points.append({"channel": ch, "x": xy[0], "y": xy[1], "hurst_exponent": hval})
    channel_plot_results: list[dict[str, Any]] = []
    for ch, res in channel_results.items():
        channel_plot_results.append({"channel": ch, "hurst_exponent": _safe_float(res.get("hurst_exponent")), "fit_r2": _safe_float(res.get("fit_r2")), "n_levels": int(res.get("n_levels") or 0), "log2_scales_fit": _json_safe(res.get("log2_scales_fit", [])), "log2_stdevs_fit": _json_safe(res.get("log2_stdevs_fit", [])), "fit_line": _json_safe(res.get("fit_line", [])), "detail_stdevs": _json_safe(res.get("detail_stdevs", [])), "detail_energies": _json_safe(res.get("detail_energies", [])), "relative_detail_energies": _json_safe(res.get("relative_detail_energies", []))})
    raw_x, raw_idx = _wh_downsample_for_plot(x, raw_plot_max_points)
    raw_z = [_wh_safe_zscore_1d(raw_x[:, j]) for j in range(raw_x.shape[1])]
    raw_matrix = np.column_stack(raw_z) if raw_z else np.empty((0, 0))
    raw_plot = {"time_sec": _json_safe(raw_idx / fs if fs > 0 else raw_idx), "channels": channels, "zscore_traces": _json_safe(raw_matrix.T), "offset": 5.0}
    temporal_rows = temporal_df.to_dict(orient="records") if not temporal_df.empty else []
    temporal_summary = []
    temporal_matrix_rows = []
    if not temporal_df.empty:
        tdf = temporal_df.copy(); tdf["hurst_exponent"] = pd.to_numeric(tdf["hurst_exponent"], errors="coerce"); tdf["fit_r2"] = pd.to_numeric(tdf["fit_r2"], errors="coerce")
        temporal_summary = tdf.groupby("window").agg(mean_hurst=("hurst_exponent", "mean"), std_hurst=("hurst_exponent", "std"), mean_r2=("fit_r2", "mean"), n=("channel", "count")).reset_index().to_dict(orient="records")
        piv = tdf.pivot_table(index="channel", columns="window", values="hurst_exponent", aggfunc="mean")
        if not piv.empty:
            order = piv.mean(axis=1).sort_values(ascending=False).index; piv = piv.loc[order]
            temporal_matrix_rows = [{"channel": idx, "values": [{"window": int(col), "hurst_exponent": _safe_float(val)} for col, val in row.items()]} for idx, row in piv.iterrows()]
    out_dir = root / "advanced_methods" / "wavelet_hurst_exponent"; out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / f"manual_wavelet_hurst_per_channel_{mode}.csv"; results_df.to_csv(results_csv, index=False)
    temporal_csv = out_dir / f"manual_wavelet_hurst_temporal_windows_{mode}.csv"
    if not temporal_df.empty: temporal_df.to_csv(temporal_csv, index=False)
    summary_txt = out_dir / f"manual_wavelet_hurst_summary_{mode}.txt"
    ok_h = pd.to_numeric(results_df["hurst_exponent"], errors="coerce") if "hurst_exponent" in results_df else pd.Series(dtype=float)
    ok_r2 = pd.to_numeric(results_df["fit_r2"], errors="coerce") if "fit_r2" in results_df else pd.Series(dtype=float)
    summary_txt.write_text("Manual Wavelet-Based Hurst Exponent Analysis\n===========================================\n\n" + f"Recording: {root}\nMode: {mode}\nWavelet: {wavelet}\nMax level request: {max_level}\nSampling rate analysis Hz: {fs}\nChannels analyzed: {len(rows)}\nMean Hurst: {float(np.nanmean(ok_h)) if len(ok_h) else np.nan}\nMedian Hurst: {float(np.nanmedian(ok_h)) if len(ok_h) else np.nan}\nMean fit R2: {float(np.nanmean(ok_r2)) if len(ok_r2) else np.nan}\n", encoding="utf-8")
    return {"wavelet_hurst_exponent": {"summary": {"mode": mode, "wavelet": wavelet, "max_level": max_level, "n_channels": int(len(channels)), "analysis_samples": int(x.shape[0]), "sampling_rate_hz": fs, "temporal_stability": bool(temporal), "temporal_windows": int(temporal_windows), "mean_hurst": _safe_float(np.nanmean(ok_h)) if len(ok_h) else None, "median_hurst": _safe_float(np.nanmedian(ok_h)) if len(ok_h) else None, "mean_fit_r2": _safe_float(np.nanmean(ok_r2)) if len(ok_r2) else None}, "rows": rows, "channel_results": channel_plot_results, "regional_summary": regional_summary, "asymmetry": asym_rows, "topographic_points": topographic_points, "raw_plot": raw_plot, "temporal_rows": temporal_rows, "temporal_summary": temporal_summary, "temporal_matrix": temporal_matrix_rows, "outputs": {"results_csv": str(results_csv), "temporal_csv": str(temporal_csv) if not temporal_df.empty else "", "summary_txt": str(summary_txt)}}}



# ---- v0.11.20 Manual Expert MFDFA analysis ----
# Adapted from the user's Manual Expert MFDFA notebook. This app-native version
# reads the converted recording's canonical signal.npy, saves the same black/cyan
# Matplotlib plots to disk, and returns plot paths that the frontend displays.

MFDFA_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {
        "scale_min_power": 0.8, "scale_max_power": 3.45, "n_scales": 14,
        "q_min": -4.0, "q_max": 4.0, "q_count": 17, "poly_order": 1,
        "raw_plot_max_points": 4000, "save_per_channel_plots": False,
        "per_channel_plot_top_n": 0, "spectrum_gallery_top_n": 6,
        "fq_gallery_top_n": 6, "run_temporal_stability": False,
        "temporal_windows": 4, "temporal_q_vals": [-4.0, 0.0, 4.0],
        "max_analysis_samples": 90_000,
    },
    "fast": {
        "scale_min_power": 0.7, "scale_max_power": 3.75, "n_scales": 20,
        "q_min": -5.0, "q_max": 5.0, "q_count": 25, "poly_order": 1,
        "raw_plot_max_points": 10000, "save_per_channel_plots": True,
        "per_channel_plot_top_n": None, "spectrum_gallery_top_n": 9,
        "fq_gallery_top_n": 9, "run_temporal_stability": True,
        "temporal_windows": 6, "temporal_q_vals": [-5.0, 0.0, 5.0],
        "max_analysis_samples": 120_000,
    },
    "balanced": {
        "scale_min_power": 0.7, "scale_max_power": 4.0, "n_scales": 30,
        "q_min": -5.0, "q_max": 5.0, "q_count": 41, "poly_order": 1,
        "raw_plot_max_points": 20000, "save_per_channel_plots": True,
        "per_channel_plot_top_n": None, "spectrum_gallery_top_n": 12,
        "fq_gallery_top_n": 12, "run_temporal_stability": True,
        "temporal_windows": 10, "temporal_q_vals": [-5.0, -2.0, 0.0, 2.0, 5.0],
        "max_analysis_samples": 180_000,
    },
    "full": {
        "scale_min_power": 0.7, "scale_max_power": 4.0, "n_scales": 36,
        "q_min": -6.0, "q_max": 6.0, "q_count": 49, "poly_order": 2,
        "raw_plot_max_points": None, "save_per_channel_plots": True,
        "per_channel_plot_top_n": None, "spectrum_gallery_top_n": 16,
        "fq_gallery_top_n": 16, "run_temporal_stability": True,
        "temporal_windows": 16, "temporal_q_vals": [-6.0, -3.0, 0.0, 3.0, 6.0],
        "max_analysis_samples": 240_000,
    },
}

MFDFA_ACCENT = "#00FFFF"
MFDFA_ACCENT_SOFT = "#66FFFF"
MFDFA_ACCENT_DIM = "#008B8B"
MFDFA_WHITE = "#E8FFFF"
MFDFA_BLACK = "#000000"
MFDFA_EPS = 1e-12
MFDFA_PROJECTION_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _mfdfa_import_plotting():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cyan_seq = LinearSegmentedColormap.from_list("cyan_sequential", [MFDFA_BLACK, MFDFA_ACCENT_DIM, MFDFA_ACCENT, MFDFA_ACCENT_SOFT], N=256)
    return plt, cyan_seq


def _mfdfa_apply_style(plt: Any) -> None:
    plt.rcParams.update({
        "figure.facecolor": MFDFA_BLACK,
        "axes.facecolor": MFDFA_BLACK,
        "savefig.facecolor": MFDFA_BLACK,
        "text.color": MFDFA_ACCENT,
        "axes.labelcolor": MFDFA_ACCENT,
        "axes.edgecolor": MFDFA_ACCENT,
        "axes.titlecolor": MFDFA_ACCENT,
        "xtick.color": MFDFA_ACCENT,
        "ytick.color": MFDFA_ACCENT,
        "grid.color": MFDFA_ACCENT_DIM,
        "grid.alpha": 0.25,
        "legend.facecolor": MFDFA_BLACK,
        "legend.edgecolor": MFDFA_ACCENT,
        "legend.labelcolor": MFDFA_ACCENT,
        "font.size": 11,
    })


def _mfdfa_style_ax(ax: Any, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, grid: bool = True) -> None:
    ax.set_facecolor(MFDFA_BLACK)
    for spine in ax.spines.values():
        spine.set_color(MFDFA_ACCENT)
        spine.set_linewidth(1.1)
    ax.tick_params(axis="x", colors=MFDFA_ACCENT, labelcolor=MFDFA_ACCENT)
    ax.tick_params(axis="y", colors=MFDFA_ACCENT, labelcolor=MFDFA_ACCENT)
    if title is not None:
        ax.set_title(title, color=MFDFA_ACCENT, pad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=MFDFA_ACCENT)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=MFDFA_ACCENT)
    if grid:
        ax.grid(True, alpha=0.22, color=MFDFA_ACCENT_DIM, linewidth=0.7)
    else:
        ax.grid(False)


def _mfdfa_save_fig(fig: Any, path: Path, *, dpi: int = 220) -> str:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=MFDFA_BLACK)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)


def _mfdfa_safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.where(np.isfinite(x), x, np.nan)
    if np.any(~np.isfinite(x)):
        med = np.nanmedian(x)
        if not np.isfinite(med):
            med = 0.0
        x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)
    return (x - np.mean(x)) / (np.std(x) + MFDFA_EPS)


def _mfdfa_downsample_for_plot(X: np.ndarray, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    if max_points is None or X.shape[0] <= max_points:
        return X, np.arange(X.shape[0])
    step = int(math.ceil(X.shape[0] / max_points))
    idx = np.arange(0, X.shape[0], step)
    return X[idx], idx


def _mfdfa_make_scales(n_timepoints: int, cfg: dict[str, Any], poly_order: int) -> np.ndarray:
    lag = np.logspace(float(cfg["scale_min_power"]), float(cfg["scale_max_power"]), int(cfg["n_scales"])).astype(int)
    lag = np.unique(lag)
    lag = lag[(lag > poly_order + 2) & (lag < n_timepoints // 4)]
    lag = lag[lag >= 4]
    if len(lag) < 5:
        raise ValueError("Too few valid MFDFA scales after filtering. Increase segment length or reduce scale_min/scale_max.")
    return lag.astype(int)


def _mfdfa_detrend_segments_matrix(segments: np.ndarray, order: int) -> np.ndarray:
    segments = np.asarray(segments, dtype=float)
    _, s = segments.shape
    key = (s, order)
    if key not in MFDFA_PROJECTION_CACHE:
        t = np.linspace(-1, 1, s)
        design = np.column_stack([t ** k for k in range(order + 1)])
        projection = design @ np.linalg.pinv(design)
        MFDFA_PROJECTION_CACHE[key] = projection
    trend = segments @ MFDFA_PROJECTION_CACHE[key].T
    return segments - trend


def _mfdfa_manual_fast(x: np.ndarray, scales: np.ndarray, q_vals: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _mfdfa_safe_zscore(np.asarray(x, dtype=float).reshape(-1))
    N = len(x)
    profile = np.cumsum(x - np.mean(x))
    Fq = np.full((len(scales), len(q_vals)), np.nan, dtype=float)
    segment_counts = np.zeros(len(scales), dtype=int)
    for si, s_raw in enumerate(scales):
        s = int(s_raw)
        Ns = N // s
        if Ns < 2 or s <= order + 2:
            continue
        forward = profile[: Ns * s].reshape(Ns, s)
        backward = profile[N - Ns * s :].reshape(Ns, s)
        segments = np.vstack([forward, backward])
        resid = _mfdfa_detrend_segments_matrix(segments, order)
        F2 = np.maximum(np.mean(resid ** 2, axis=1), MFDFA_EPS)
        segment_counts[si] = len(F2)
        log_F2_mean = np.mean(np.log(F2))
        for qi, q in enumerate(q_vals):
            if np.isclose(q, 0.0):
                Fq[si, qi] = np.exp(0.5 * log_F2_mean)
            else:
                Fq[si, qi] = (np.mean(F2 ** (q / 2.0))) ** (1.0 / q)
    return np.asarray(scales, dtype=float), Fq, segment_counts


def _mfdfa_estimate_hq(scales: np.ndarray, Fq: np.ndarray, min_points: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_s = np.log(np.asarray(scales, dtype=float))
    Fq = np.asarray(Fq, dtype=float)
    hq = np.full(Fq.shape[1], np.nan, dtype=float)
    intercepts = np.full(Fq.shape[1], np.nan, dtype=float)
    fit_r2 = np.full(Fq.shape[1], np.nan, dtype=float)
    fit_points = np.zeros(Fq.shape[1], dtype=int)
    for qi in range(Fq.shape[1]):
        y = Fq[:, qi]
        mask = np.isfinite(y) & (y > 0) & np.isfinite(log_s)
        if np.sum(mask) >= min_points:
            x_fit = log_s[mask]
            y_fit = np.log(y[mask])
            coef = np.polyfit(x_fit, y_fit, 1)
            y_hat = coef[0] * x_fit + coef[1]
            ss_res = np.sum((y_fit - y_hat) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            hq[qi] = coef[0]
            intercepts[qi] = coef[1]
            fit_r2[qi] = 1.0 - ss_res / (ss_tot + MFDFA_EPS)
            fit_points[qi] = int(np.sum(mask))
    return hq, intercepts, fit_r2, fit_points


def _mfdfa_spectrum(q_vals: np.ndarray, hq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_vals = np.asarray(q_vals, dtype=float)
    hq = np.asarray(hq, dtype=float)
    tau = np.full_like(q_vals, np.nan, dtype=float)
    alpha = np.full_like(q_vals, np.nan, dtype=float)
    f_alpha = np.full_like(q_vals, np.nan, dtype=float)
    mask = np.isfinite(hq) & np.isfinite(q_vals)
    if np.sum(mask) < 3:
        return tau, alpha, f_alpha
    qv = q_vals[mask]
    hv = hq[mask]
    tau_v = qv * hv - 1.0
    alpha_v = np.gradient(tau_v, qv)
    f_v = qv * alpha_v - tau_v
    tau[mask] = tau_v
    alpha[mask] = alpha_v
    f_alpha[mask] = f_v
    return tau, alpha, f_alpha


def _mfdfa_summary(alpha: np.ndarray, f_alpha: np.ndarray, hq: np.ndarray, hq_r2: np.ndarray, q_vals: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(alpha) & np.isfinite(f_alpha)
    def nearest_h(q_target: float) -> float | None:
        idx = int(np.argmin(np.abs(q_vals - q_target)))
        return _safe_float(hq[idx]) if np.isfinite(hq[idx]) else None
    q_min = float(np.nanmin(q_vals)); q_max = float(np.nanmax(q_vals))
    out: dict[str, Any] = {
        "hq_qmin": nearest_h(q_min), "hq_q0": nearest_h(0.0), "hq_qmax": nearest_h(q_max),
        "mean_hq_fit_r2": _safe_float(np.nanmean(hq_r2)) if np.any(np.isfinite(hq_r2)) else None,
        "min_hq_fit_r2": _safe_float(np.nanmin(hq_r2)) if np.any(np.isfinite(hq_r2)) else None,
        "hq_range": _safe_float(np.nanmax(hq) - np.nanmin(hq)) if np.sum(np.isfinite(hq)) >= 2 else None,
    }
    if np.sum(mask) < 3:
        out.update({"alpha_min": None, "alpha_max": None, "alpha_width": None, "alpha_at_peak": None, "f_peak": None, "asymmetry": None, "left_width": None, "right_width": None, "multifractal_complexity_score": None})
        return out
    a = alpha[mask]; f = f_alpha[mask]
    alpha_min = float(np.min(a)); alpha_max = float(np.max(a)); alpha_width = float(alpha_max - alpha_min)
    peak_idx = int(np.argmax(f)); alpha_at_peak = float(a[peak_idx]); f_peak = float(f[peak_idx])
    left_width = float(alpha_at_peak - alpha_min); right_width = float(alpha_max - alpha_at_peak)
    asymmetry = float(right_width - left_width)
    r2 = float(out["mean_hq_fit_r2"] or 0.0)
    score = alpha_width * max(r2, 0.0) * (1.0 + 0.15 * abs(asymmetry))
    out.update({"alpha_min": alpha_min, "alpha_max": alpha_max, "alpha_width": alpha_width, "alpha_at_peak": alpha_at_peak, "f_peak": f_peak, "asymmetry": asymmetry, "left_width": left_width, "right_width": right_width, "multifractal_complexity_score": float(score)})
    return out


def _mfdfa_channel(x: np.ndarray, scales: np.ndarray, q_vals: np.ndarray, order: int) -> dict[str, Any]:
    scales_out, Fq, segment_counts = _mfdfa_manual_fast(x, scales, q_vals, order)
    hq, hq_intercepts, hq_r2, hq_fit_points = _mfdfa_estimate_hq(scales_out, Fq)
    tau_q, alpha_q, f_alpha_q = _mfdfa_spectrum(q_vals, hq)
    return {"scales": scales_out, "q_vals": q_vals, "Fq": Fq, "segment_counts": segment_counts, "hq": hq, "hq_intercepts": hq_intercepts, "hq_r2": hq_r2, "hq_fit_points": hq_fit_points, "tau_q": tau_q, "alpha_q": alpha_q, "f_alpha_q": f_alpha_q, "summary": _mfdfa_summary(alpha_q, f_alpha_q, hq, hq_r2, q_vals)}


def _mfdfa_temporal(x: np.ndarray, channels: list[str], scales_base: np.ndarray, q_vals_temporal: np.ndarray, *, fs: float, poly_order: int, n_windows: int) -> pd.DataFrame:
    n = x.shape[0]
    window_len = max(int(np.max(scales_base) * 5), n // max(n_windows, 1), 2048)
    rows: list[dict[str, Any]] = []
    for w in range(n_windows):
        s0 = w * window_len
        e0 = min(n, s0 + window_len)
        if e0 - s0 < 2048:
            continue
        local_scales = scales_base[scales_base < (e0 - s0) // 4]
        if len(local_scales) < 5:
            continue
        for ch_idx, ch in enumerate(channels):
            try:
                res = _mfdfa_channel(x[s0:e0, ch_idx], local_scales, q_vals_temporal, poly_order)
                summ = res["summary"]
                rows.append({"window": w + 1, "channel": ch, "start_sample": s0, "end_sample": e0, "start_sec_relative": _safe_float(s0 / fs) if fs else None, "end_sec_relative": _safe_float(e0 / fs) if fs else None, "alpha_width": summ.get("alpha_width"), "asymmetry": summ.get("asymmetry"), "hq_q0": summ.get("hq_q0"), "mean_hq_fit_r2": summ.get("mean_hq_fit_r2"), "multifractal_complexity_score": summ.get("multifractal_complexity_score"), "status": "ok", "error": ""})
            except Exception as exc:
                rows.append({"window": w + 1, "channel": ch, "start_sample": s0, "end_sample": e0, "start_sec_relative": _safe_float(s0 / fs) if fs else None, "end_sec_relative": _safe_float(e0 / fs) if fs else None, "alpha_width": None, "asymmetry": None, "hq_q0": None, "mean_hq_fit_r2": None, "multifractal_complexity_score": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def _manual_expert_mfdfa(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "ultra").lower()
    if mode not in MFDFA_CONFIGS:
        mode = "ultra"
    cfg = dict(MFDFA_CONFIGS[mode])
    # Parameter overrides, retaining notebook defaults unless user changes them.
    for key in ("scale_min_power", "scale_max_power"):
        if params.get(key) not in (None, ""):
            cfg[key] = float(params[key])
    for key in ("n_scales", "poly_order", "max_analysis_samples", "raw_plot_max_points", "spectrum_gallery_top_n", "fq_gallery_top_n", "temporal_windows"):
        if params.get(key) not in (None, ""):
            cfg[key] = _safe_int(params[key], int(cfg[key] or 1))
    q_min = float(params.get("q_min") if params.get("q_min") not in (None, "") else cfg["q_min"])
    q_max = float(params.get("q_max") if params.get("q_max") not in (None, "") else cfg["q_max"])
    q_count = _safe_int(params.get("q_count"), int(cfg["q_count"]))
    q_vals = np.linspace(q_min, q_max, q_count)
    temporal_q_vals = np.asarray(cfg["temporal_q_vals"], dtype=float)
    max_channels = _safe_int(params.get("max_channels"), 999999) if params.get("max_channels") not in (None, "") else None
    temporal = str(params.get("temporal_stability", str(cfg["run_temporal_stability"]))).lower() not in ("0", "false", "no", "off")
    save_per_channel_plots = str(params.get("save_per_channel_plots", str(cfg["save_per_channel_plots"]))).lower() in ("1", "true", "yes", "on")
    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=cfg.get("max_analysis_samples"))
    root = Path(rec["recording_dir"])
    x = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels is not None:
        channels = channels[:max_channels]
        x = x[:, :len(channels)]
    segment_start = _safe_float(params.get("segment_start_sec"))
    segment_end = _safe_float(params.get("segment_end_sec"))
    if segment_start is not None or segment_end is not None:
        s0 = int(max(0, (segment_start or 0.0) * fs))
        s1 = int(min(x.shape[0], (segment_end if segment_end is not None else (x.shape[0] / fs)) * fs))
        if s1 > s0 + 32:
            x = x[s0:s1, :]
    poly_order = int(cfg["poly_order"])
    scales = _mfdfa_make_scales(x.shape[0], cfg, poly_order)
    plt, cmap = _mfdfa_import_plotting(); _mfdfa_apply_style(plt)
    out_dir = root / "advanced_methods" / "manual_expert_mfdfa"
    plots_dir = out_dir / "plots"
    per_channel_dir = plots_dir / "per_channel"
    gallery_dir = plots_dir / "galleries"
    for d in (out_dir, plots_dir, per_channel_dir, gallery_dir): d.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    hq_rows: list[dict[str, Any]] = []
    Fq_rows: list[dict[str, Any]] = []
    mfdfa_results: list[dict[str, Any]] = []
    start = __import__('time').perf_counter()
    for ch_idx, channel in enumerate(channels):
        sig = x[:, ch_idx]
        t0 = __import__('time').perf_counter()
        try:
            res = _mfdfa_channel(sig, scales, q_vals, poly_order)
            elapsed = __import__('time').perf_counter() - t0
            row: dict[str, Any] = {"channel": channel, "mean_signal": _safe_float(np.mean(sig)), "std_signal": _safe_float(np.std(sig)), "n_scales": int(len(res["scales"])), "n_q": int(len(q_vals)), "poly_order": poly_order, "elapsed_sec": _safe_float(elapsed), "status": "ok", "error": "", "hemisphere": _wh_hemisphere(channel), "region": _wh_region(channel)}
            row.update(res["summary"])
            summary_rows.append(row)
            mfdfa_results.append({"channel": channel, "scales": res["scales"], "q_vals": q_vals, "Fq": res["Fq"], "segment_counts": res["segment_counts"], "hq": res["hq"], "hq_r2": res["hq_r2"], "alpha_q": res["alpha_q"], "f_alpha_q": res["f_alpha_q"], "tau_q": res["tau_q"]})
            for qi, q in enumerate(q_vals):
                hq_rows.append({"channel": channel, "q": float(q), "hq": _safe_float(res["hq"][qi]), "hq_fit_r2": _safe_float(res["hq_r2"][qi]), "hq_fit_points": int(res["hq_fit_points"][qi]), "tau_q": _safe_float(res["tau_q"][qi]), "alpha_q": _safe_float(res["alpha_q"][qi]), "f_alpha_q": _safe_float(res["f_alpha_q"][qi])})
            for si, scale in enumerate(res["scales"]):
                for qi, q in enumerate(q_vals):
                    Fq_rows.append({"channel": channel, "scale": float(scale), "q": float(q), "Fq": _safe_float(res["Fq"][si, qi]), "segment_count": int(res["segment_counts"][si])})
            if save_per_channel_plots:
                try:
                    Xp_ch, idx_ch = _mfdfa_downsample_for_plot(np.asarray(sig, dtype=float), cfg.get("raw_plot_max_points"))
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=MFDFA_BLACK)
                    ax1, ax2 = axes
                    ax1.plot(idx_ch, _mfdfa_safe_zscore(Xp_ch), color=MFDFA_ACCENT, linewidth=0.9, alpha=0.85)
                    _mfdfa_style_ax(ax1, f"{channel}: EEG Segment", "Samples", "Z-scored amplitude")
                    scales_out = np.asarray(res["scales"], dtype=float)
                    Fq_arr = np.asarray(res["Fq"], dtype=float)
                    for qi in range(len(q_vals)):
                        y = Fq_arr[:, qi]
                        mask = np.isfinite(y) & (y > 0)
                        if np.sum(mask) >= 2:
                            alpha = 0.18 if qi not in (0, len(q_vals) // 2, len(q_vals) - 1) else 0.75
                            lw = 0.8 if alpha < 0.5 else 1.4
                            ax2.loglog(scales_out[mask], y[mask], color=MFDFA_ACCENT, alpha=alpha, linewidth=lw)
                    _mfdfa_style_ax(ax2, f"{channel}: Manual MFDFA Fq(s)", "Scale", "Fq(s)")
                    _mfdfa_save_fig(fig, per_channel_dir / f"{channel}_signal_and_Fq_{mode}.png")
                    fig, ax = plt.subplots(figsize=(5.5, 4.3), facecolor=MFDFA_BLACK)
                    hq_arr = np.asarray(res["hq"], dtype=float)
                    ax.plot(q_vals, hq_arr, marker="o", color=MFDFA_ACCENT, linewidth=1.3, markersize=3)
                    _mfdfa_style_ax(ax, f"{channel}: h(q)", "q", "Generalized Hurst exponent")
                    _mfdfa_save_fig(fig, per_channel_dir / f"{channel}_hq_{mode}.png")
                    fig, ax = plt.subplots(figsize=(5.5, 4.3), facecolor=MFDFA_BLACK)
                    alpha_q = np.asarray(res["alpha_q"], dtype=float); f_alpha_q = np.asarray(res["f_alpha_q"], dtype=float)
                    mask = np.isfinite(alpha_q) & np.isfinite(f_alpha_q)
                    ax.plot(alpha_q[mask], f_alpha_q[mask], marker="o", color=MFDFA_ACCENT, linewidth=1.3, markersize=3)
                    _mfdfa_style_ax(ax, f"{channel}: Multifractal Spectrum", "alpha", "f(alpha)")
                    _mfdfa_save_fig(fig, per_channel_dir / f"{channel}_multifractal_spectrum_{mode}.png")
                except Exception:
                    pass
        except Exception as exc:
            summary_rows.append({"channel": channel, "mean_signal": _safe_float(np.mean(sig)), "std_signal": _safe_float(np.std(sig)), "n_scales": int(len(scales)), "n_q": int(len(q_vals)), "poly_order": poly_order, "elapsed_sec": _safe_float(__import__('time').perf_counter() - t0), "status": "error", "error": f"{type(exc).__name__}: {exc}", "alpha_width": None, "asymmetry": None, "hq_q0": None, "mean_hq_fit_r2": None, "multifractal_complexity_score": None, "hemisphere": _wh_hemisphere(channel), "region": _wh_region(channel)})
    summary_df = pd.DataFrame(summary_rows)
    hq_df = pd.DataFrame(hq_rows)
    Fq_df = pd.DataFrame(Fq_rows)
    temporal_df = _mfdfa_temporal(x, channels, scales, temporal_q_vals, fs=fs, poly_order=poly_order, n_windows=int(cfg["temporal_windows"])) if temporal else pd.DataFrame()
    summary_csv = out_dir / f"mfdfa_summary_{mode}.csv"; summary_df.to_csv(summary_csv, index=False)
    hq_csv = out_dir / f"mfdfa_hq_tau_alpha_falpha_{mode}.csv"; hq_df.to_csv(hq_csv, index=False)
    Fq_csv = out_dir / f"mfdfa_Fq_values_{mode}.csv"; Fq_df.to_csv(Fq_csv, index=False)
    temporal_csv = out_dir / f"mfdfa_temporal_stability_{mode}.csv"
    if not temporal_df.empty: temporal_df.to_csv(temporal_csv, index=False)
    npz_path = out_dir / f"mfdfa_results_{mode}.npz"
    try:
        np.savez_compressed(npz_path, scales=scales, q_vals=q_vals, poly_order=poly_order, channels=np.array(channels, dtype=object), summary=summary_df.to_dict("records"))
    except Exception:
        pass
    plot_paths: list[dict[str, str]] = []
    def add_plot(title: str, path: str | None) -> None:
        if path: plot_paths.append({"title": title, "path": path, "url": f"/api/file?path={path}"})
    raw_max = cfg.get("raw_plot_max_points")
    # Raw offset traces
    Xp, idx = _mfdfa_downsample_for_plot(x, raw_max)
    t = idx / fs if fs else idx
    Xz = np.column_stack([_mfdfa_safe_zscore(Xp[:, i]) for i in range(Xp.shape[1])]) if Xp.size else np.empty_like(Xp)
    fig_h = min(10.5, max(6.0, 0.28 * len(channels))); fig, ax = plt.subplots(figsize=(13, fig_h)); offset = 5.0
    for i, ch in enumerate(channels): ax.plot(t, Xz[:, i] + i * offset, color=MFDFA_ACCENT, linewidth=0.55, alpha=0.65)
    ax.set_yticks(np.arange(len(channels))*offset); ax.set_yticklabels(channels, color=MFDFA_ACCENT, fontsize=8)
    _mfdfa_style_ax(ax, "EEG Segment Used for Manual MFDFA", "Time from segment start (s)", "Channel")
    add_plot("Raw EEG offset traces", _mfdfa_save_fig(fig, plots_dir / f"00_raw_eeg_segment_offset_{mode}.png"))
    # Alpha width barh
    def _barh_metric(df: pd.DataFrame, metric: str, title: str, xlabel: str, fname: str, zero: bool=False) -> str | None:
        if df.empty or metric not in df.columns: return None
        plot_df = df.copy(); plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce"); plot_df = plot_df.sort_values(metric, ascending=True, na_position="first")
        fig, ax = plt.subplots(figsize=(10, 10)); y = np.arange(len(plot_df))
        ax.barh(y, plot_df[metric].fillna(0.0).values, color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.86)
        ax.set_yticks(y); ax.set_yticklabels(plot_df["channel"].values, color=MFDFA_ACCENT, fontsize=8)
        if zero: ax.axvline(0.0, color=MFDFA_WHITE, linestyle="--", linewidth=1.0, alpha=0.75)
        else:
            mean_val = plot_df[metric].mean(); ax.axvline(mean_val, color=MFDFA_WHITE, linestyle="--", linewidth=1.1, alpha=0.85, label=f"Mean={mean_val:.3f}"); ax.legend(loc="best", frameon=True)
        _mfdfa_style_ax(ax, title, xlabel, "Channel")
        return _mfdfa_save_fig(fig, plots_dir / fname)
    add_plot("Multifractal spectrum width by channel", _barh_metric(summary_df, "alpha_width", "Multifractal Spectrum Width by Channel", "alpha width", f"01_mfdfa_alpha_width_barh_{mode}.png"))
    add_plot("Multifractal spectrum asymmetry by channel", _barh_metric(summary_df, "asymmetry", "Multifractal Spectrum Asymmetry by Channel", "Asymmetry: right width - left width", f"02_mfdfa_asymmetry_barh_{mode}.png", True))
    # h(q) all channels
    fig, ax = plt.subplots(figsize=(9.5,6.5));
    if mfdfa_results:
        stack = []
        for item in mfdfa_results:
            hq = np.asarray(item["hq"], dtype=float); mask = np.isfinite(hq)
            if np.sum(mask) >= 2: ax.plot(q_vals[mask], hq[mask], color=MFDFA_ACCENT, alpha=0.30, linewidth=1.0); stack.append(hq)
        if stack:
            hq_stack = np.array(stack); ax.plot(q_vals, np.nanmean(hq_stack, axis=0), color=MFDFA_WHITE, linewidth=2.4, label="Mean h(q)"); ax.plot(q_vals, np.nanmedian(hq_stack, axis=0), color=MFDFA_ACCENT_SOFT, linewidth=1.8, linestyle="--", label="Median h(q)"); ax.legend(loc="best", frameon=True)
    _mfdfa_style_ax(ax, "Generalized Hurst Exponent Curves Across Channels", "q", "h(q)")
    add_plot("h(q) curves across channels", _mfdfa_save_fig(fig, plots_dir / f"03_mfdfa_hq_all_channels_{mode}.png"))
    # Spectra all channels
    fig, ax = plt.subplots(figsize=(9.5,6.5))
    for item in mfdfa_results:
        a=np.asarray(item["alpha_q"],float); f=np.asarray(item["f_alpha_q"],float); mask=np.isfinite(a)&np.isfinite(f)
        if np.sum(mask)>=2: ax.plot(a[mask], f[mask], color=MFDFA_ACCENT, alpha=0.34, linewidth=1.0)
    _mfdfa_style_ax(ax, "Multifractal Spectra Across Channels", "alpha", "f(alpha)")
    add_plot("Multifractal spectra across channels", _mfdfa_save_fig(fig, plots_dir / f"04_mfdfa_spectra_all_channels_{mode}.png"))
    # hq heatmap
    if not hq_df.empty:
        pivot = hq_df.pivot_table(index="channel", columns="q", values="hq", aggfunc="mean")
        if not pivot.empty:
            order = pivot.mean(axis=1).sort_values(ascending=False).index; pivot=pivot.loc[order]; M=pivot.to_numpy(dtype=float); finite=M[np.isfinite(M)]
            fig_h=min(10.5,max(7.0,0.30*len(pivot.index))); fig, ax=plt.subplots(figsize=(12,fig_h)); im=ax.imshow(M,aspect="auto",interpolation="nearest",cmap=cmap,vmin=np.nanpercentile(finite,3) if finite.size else 0,vmax=np.nanpercentile(finite,97) if finite.size else 1)
            q_labels=[f"{q:.1f}" if i % max(1,len(pivot.columns)//10)==0 else "" for i,q in enumerate(pivot.columns)]
            ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(q_labels,color=MFDFA_ACCENT,fontsize=8); ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index,color=MFDFA_ACCENT,fontsize=8)
            _mfdfa_style_ax(ax,"Generalized Hurst Exponent Heatmap h(q)","q","Channel, sorted by mean h(q)",False); cbar=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02); cbar.set_label("h(q)",color=MFDFA_ACCENT); cbar.outline.set_edgecolor(MFDFA_ACCENT); cbar.ax.tick_params(color=MFDFA_ACCENT); [lab.set_color(MFDFA_ACCENT) for lab in cbar.ax.get_yticklabels()]
            add_plot("h(q) heatmap", _mfdfa_save_fig(fig, plots_dir / f"05_hq_heatmap_{mode}.png"))
    # summary metric heatmap
    metric_cols = [c for c in ["alpha_width","asymmetry","hq_qmin","hq_q0","hq_qmax","hq_range","mean_hq_fit_r2","multifractal_complexity_score"] if c in summary_df.columns]
    if metric_cols:
        dfm=summary_df.copy(); dfm=dfm.sort_values("multifractal_complexity_score", ascending=False, na_position="last"); M=dfm[metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float); M_norm=np.zeros_like(M)
        for j in range(M.shape[1]):
            col=M[:,j]; finite=col[np.isfinite(col)]
            if finite.size and np.nanmax(finite)!=np.nanmin(finite): M_norm[:,j]=(col-np.nanmin(finite))/(np.nanmax(finite)-np.nanmin(finite)+MFDFA_EPS)
        fig_h=min(10.5,max(7.0,0.30*len(dfm))); fig, ax=plt.subplots(figsize=(12,fig_h)); im=ax.imshow(M_norm,aspect="auto",interpolation="nearest",cmap=cmap,vmin=0,vmax=1)
        ax.set_xticks(np.arange(len(metric_cols))); ax.set_xticklabels(metric_cols,rotation=50,ha="right",color=MFDFA_ACCENT,fontsize=8); ax.set_yticks(np.arange(len(dfm))); ax.set_yticklabels(dfm["channel"],color=MFDFA_ACCENT,fontsize=8)
        _mfdfa_style_ax(ax,"MFDFA Summary Metric Heatmap, Column-Normalized","Metric","Channel, sorted by complexity score",False); cbar=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02); cbar.set_label("Column-normalized value",color=MFDFA_ACCENT); cbar.outline.set_edgecolor(MFDFA_ACCENT); cbar.ax.tick_params(color=MFDFA_ACCENT); [lab.set_color(MFDFA_ACCENT) for lab in cbar.ax.get_yticklabels()]
        add_plot("Summary metric heatmap", _mfdfa_save_fig(fig, plots_dir / f"06_mfdfa_summary_metric_heatmap_{mode}.png"))
    # width vs hq0
    dfw=summary_df.copy();
    if "alpha_width" in dfw and "hq_q0" in dfw:
        dfw["alpha_width"]=pd.to_numeric(dfw["alpha_width"], errors="coerce"); dfw["hq_q0"]=pd.to_numeric(dfw["hq_q0"], errors="coerce"); dfw=dfw[np.isfinite(dfw["alpha_width"]) & np.isfinite(dfw["hq_q0"])]
        if not dfw.empty:
            fig, ax=plt.subplots(figsize=(9,7)); sizes=70+420*np.clip(pd.to_numeric(dfw.get("mean_hq_fit_r2",0), errors="coerce").fillna(0).to_numpy(dtype=float),0,1); ax.scatter(dfw["hq_q0"], dfw["alpha_width"], s=sizes, c=MFDFA_ACCENT, edgecolors=MFDFA_WHITE, linewidths=0.8, alpha=0.82)
            for _, row in dfw.sort_values("multifractal_complexity_score", ascending=False, na_position="last").head(10).iterrows(): ax.annotate(str(row["channel"]), (row["hq_q0"], row["alpha_width"]), xytext=(5,5), textcoords="offset points", color=MFDFA_ACCENT_SOFT, fontsize=8)
            _mfdfa_style_ax(ax,"Multifractal Width vs h(0)","h(q=0)","Spectrum width Δα"); add_plot("Width vs h(0)", _mfdfa_save_fig(fig, plots_dir / f"07_width_vs_hq0_scatter_{mode}.png"))
    # galleries
    def _spectrum_gallery(kind: str) -> str | None:
        if summary_df.empty or not mfdfa_results: return None
        if kind == "spectrum":
            df = summary_df.copy(); df["alpha_width"] = pd.to_numeric(df.get("alpha_width"), errors="coerce"); df = df[np.isfinite(df["alpha_width"])].sort_values("alpha_width", ascending=False).head(int(cfg["spectrum_gallery_top_n"])); title_name="Spectrum gallery top width"; fname=f"08_spectrum_gallery_top_width_{mode}.png"
        else:
            df = summary_df.copy(); df["multifractal_complexity_score"] = pd.to_numeric(df.get("multifractal_complexity_score"), errors="coerce"); df = df[np.isfinite(df["multifractal_complexity_score"])].sort_values("multifractal_complexity_score", ascending=False).head(int(cfg["fq_gallery_top_n"])); title_name="Fq scaling gallery top complexity"; fname=f"09_fq_scaling_gallery_top_complexity_{mode}.png"
        if df.empty: return None
        rmap={item["channel"]: item for item in mfdfa_results}; n=len(df); ncols=3; nrows=int(math.ceil(n/ncols)); fig, axes=plt.subplots(nrows,ncols,figsize=(13,3.8*nrows),squeeze=False); fig.patch.set_facecolor(MFDFA_BLACK)
        for ax in axes.ravel(): ax.set_facecolor(MFDFA_BLACK); ax.axis("off")
        for ax, (_, row) in zip(axes.ravel(), df.iterrows()):
            ax.axis("on"); item=rmap.get(row["channel"]); 
            if item is None: continue
            if kind == "spectrum":
                a=np.asarray(item["alpha_q"],float); f=np.asarray(item["f_alpha_q"],float); mask=np.isfinite(a)&np.isfinite(f); ax.plot(a[mask],f[mask],marker="o",color=MFDFA_ACCENT,linewidth=1.2,markersize=3); _mfdfa_style_ax(ax,f"{row['channel']} | width={row.get('alpha_width', np.nan):.3f}, asym={row.get('asymmetry', np.nan):.3f}","alpha","f(alpha)")
            else:
                scales_i=np.asarray(item["scales"],float); Fq_i=np.asarray(item["Fq"],float); q_targets=[float(np.nanmin(q_vals)),0.0,float(np.nanmax(q_vals))]; qis=[int(np.argmin(np.abs(q_vals-qt))) for qt in q_targets]
                for qi in qis:
                    y=Fq_i[:,qi]; mask=np.isfinite(y)&(y>0)
                    if np.sum(mask)>=2: ax.plot(np.log(scales_i[mask]),np.log(y[mask]),marker="o",linewidth=1.1,markersize=3,color=MFDFA_ACCENT,alpha=0.45 if qi!=qis[1] else 0.95,label=f"q={q_vals[qi]:.1f}")
                _mfdfa_style_ax(ax,f"{row['channel']} | score={row.get('multifractal_complexity_score', np.nan):.3f}","log(scale)","log Fq(s)")
        return _mfdfa_save_fig(fig, gallery_dir / fname)
    add_plot("Spectrum gallery", _spectrum_gallery("spectrum")); add_plot("Fq scaling gallery", _spectrum_gallery("fq"))
    # Topographic alpha width
    topo=[]
    for _, row in summary_df.iterrows():
        ch=str(row.get("channel")); v=_safe_float(row.get("alpha_width"));
        if v is not None and ch in WH_ELECTRODE_XY: topo.append({"channel": ch, "x": WH_ELECTRODE_XY[ch][0], "y": WH_ELECTRODE_XY[ch][1], "alpha_width": v})
    if topo:
        fig, ax=plt.subplots(figsize=(8,8)); head=plt.Circle((0,0),1.15,color=MFDFA_ACCENT_DIM,fill=False,linewidth=1.2,alpha=0.9); ax.add_patch(head); ax.plot([0,-0.09,0.09,0],[1.15,1.28,1.28,1.15],color=MFDFA_ACCENT_DIM,linewidth=1.0)
        xs=[p["x"] for p in topo]; ys=[p["y"] for p in topo]; vals=[p["alpha_width"] for p in topo]
        sc=ax.scatter(xs,ys,c=vals,s=520,cmap=cmap,edgecolors=MFDFA_WHITE,linewidths=1.0,alpha=0.95)
        for pnt in topo: ax.text(pnt["x"], pnt["y"], pnt["channel"], ha="center", va="center", color=MFDFA_BLACK, fontsize=8, fontweight="bold")
        ax.set_aspect("equal"); ax.set_xlim(-1.35,1.35); ax.set_ylim(-1.35,1.35); ax.set_xticks([]); ax.set_yticks([]); _mfdfa_style_ax(ax,"Approximate Scalp Map of MFDFA Spectrum Width Δα","","",False); cbar=fig.colorbar(sc,ax=ax,fraction=0.040,pad=0.02); cbar.set_label("alpha width",color=MFDFA_ACCENT); cbar.outline.set_edgecolor(MFDFA_ACCENT); cbar.ax.tick_params(color=MFDFA_ACCENT); [lab.set_color(MFDFA_ACCENT) for lab in cbar.ax.get_yticklabels()]
        add_plot("Topographic alpha width", _mfdfa_save_fig(fig, plots_dir / f"10_topographic_alpha_width_{mode}.png"))
    # Region summary
    if "region" in summary_df.columns and "alpha_width" in summary_df.columns:
        rdf=summary_df.copy(); rdf["alpha_width"]=pd.to_numeric(rdf["alpha_width"], errors="coerce"); rdf=rdf[np.isfinite(rdf["alpha_width"])]
        if not rdf.empty:
            grouped=rdf.groupby("region").agg(mean_alpha_width=("alpha_width","mean"), std_alpha_width=("alpha_width","std"), mean_asymmetry=("asymmetry","mean"), mean_hq0=("hq_q0","mean"), mean_r2=("mean_hq_fit_r2","mean"), n=("channel","count")).reset_index().sort_values("mean_alpha_width", ascending=True)
            fig, ax=plt.subplots(figsize=(10,6)); y=np.arange(len(grouped)); ax.barh(y,grouped["mean_alpha_width"],xerr=grouped["std_alpha_width"].fillna(0.0),color=MFDFA_ACCENT,edgecolor=MFDFA_ACCENT_SOFT,alpha=0.86,capsize=4); ax.set_yticks(y); ax.set_yticklabels(grouped["region"],color=MFDFA_ACCENT)
            for yi,(_,row) in zip(y, grouped.iterrows()): ax.text(row["mean_alpha_width"], yi, f"  n={int(row['n'])}, R²={row['mean_r2']:.2f}", va="center", color=MFDFA_ACCENT_SOFT, fontsize=8)
            _mfdfa_style_ax(ax,"Regional Summary of MFDFA Spectrum Width","Mean alpha width ± SD","Region"); add_plot("Regional alpha-width summary", _mfdfa_save_fig(fig, plots_dir / f"11_region_alpha_width_summary_{mode}.png"))
    # Hemisphere asymmetry
    homolog=[("Fp1","Fp2"),("F7","F8"),("F3","F4"),("FC5","FC6"),("FC1","FC2"),("M1","M2"),("T7","T8"),("C3","C4"),("CP5","CP6"),("CP1","CP2"),("P7","P8"),("P3","P4"),("O1","O2")]
    valmap={str(r.get("channel")): _safe_float(r.get("alpha_width")) for _, r in summary_df.iterrows()}; asym=[]
    for l,r in homolog:
        if valmap.get(l) is not None and valmap.get(r) is not None: asym.append({"pair": f"{l}-{r}", "left_channel": l, "right_channel": r, "left_alpha_width": valmap[l], "right_alpha_width": valmap[r], "left_minus_right_alpha_width": valmap[l]-valmap[r]})
    asym_df=pd.DataFrame(asym)
    if not asym_df.empty:
        asym_csv=out_dir / f"mfdfa_homolog_alpha_width_asymmetry_{mode}.csv"; asym_df.to_csv(asym_csv,index=False); plot_df=asym_df.sort_values("left_minus_right_alpha_width",ascending=True); fig, ax=plt.subplots(figsize=(10,7)); y=np.arange(len(plot_df)); ax.barh(y,plot_df["left_minus_right_alpha_width"],color=MFDFA_ACCENT,edgecolor=MFDFA_ACCENT_SOFT,alpha=0.86); ax.axvline(0,color=MFDFA_WHITE,linewidth=1.0,alpha=0.85); ax.set_yticks(y); ax.set_yticklabels(plot_df["pair"],color=MFDFA_ACCENT); _mfdfa_style_ax(ax,"Homologous Left-Right MFDFA Width Asymmetry","Left alpha width - right alpha width","Homologous channel pair"); add_plot("Hemisphere alpha-width asymmetry", _mfdfa_save_fig(fig, plots_dir / f"12_hemisphere_alpha_width_asymmetry_{mode}.png"))
    # Temporal plots
    if not temporal_df.empty:
        grouped=temporal_df.groupby("window").agg(mean_alpha_width=("alpha_width","mean"), std_alpha_width=("alpha_width","std"), mean_hq0=("hq_q0","mean"), mean_r2=("mean_hq_fit_r2","mean"), n=("channel","count")).reset_index(); fig, ax=plt.subplots(figsize=(11,7)); ax.plot(grouped["window"],grouped["mean_alpha_width"],marker="o",color=MFDFA_ACCENT,linewidth=2,label="Mean alpha width"); ax.fill_between(grouped["window"],grouped["mean_alpha_width"]-grouped["std_alpha_width"],grouped["mean_alpha_width"]+grouped["std_alpha_width"],color=MFDFA_ACCENT,alpha=0.12,label="±1 SD"); ax.plot(grouped["window"],grouped["mean_hq0"],marker="s",color=MFDFA_ACCENT_SOFT,linewidth=1.6,alpha=0.9,label="Mean h(q=0)"); ax.plot(grouped["window"],grouped["mean_r2"],marker="^",color=MFDFA_WHITE,linewidth=1.4,alpha=0.85,label="Mean fit R²"); _mfdfa_style_ax(ax,"Temporal Stability of MFDFA Metrics","Window","Mean metric value"); ax.legend(loc="best",frameon=True); add_plot("Temporal MFDFA stability", _mfdfa_save_fig(fig, plots_dir / f"13_temporal_mfdfa_stability_{mode}.png"))
        pivot=temporal_df.pivot_table(index="channel", columns="window", values="alpha_width", aggfunc="mean")
        if not pivot.empty:
            order=pivot.mean(axis=1).sort_values(ascending=False).index; pivot=pivot.loc[order]; M=pivot.to_numpy(dtype=float); finite=M[np.isfinite(M)]; fig_h=min(10.5,max(7.0,0.30*len(pivot.index))); fig,ax=plt.subplots(figsize=(10,fig_h)); im=ax.imshow(M,aspect="auto",interpolation="nearest",cmap=cmap,vmin=np.nanpercentile(finite,3) if finite.size else 0,vmax=np.nanpercentile(finite,97) if finite.size else 1); ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns,color=MFDFA_ACCENT); ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index,color=MFDFA_ACCENT,fontsize=8); _mfdfa_style_ax(ax,"Temporal MFDFA Alpha-Width Heatmap by Channel","Window","Channel, sorted by mean alpha width",False); cbar=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02); cbar.set_label("alpha width",color=MFDFA_ACCENT); cbar.outline.set_edgecolor(MFDFA_ACCENT); cbar.ax.tick_params(color=MFDFA_ACCENT); [lab.set_color(MFDFA_ACCENT) for lab in cbar.ax.get_yticklabels()]; add_plot("Temporal channel alpha-width heatmap", _mfdfa_save_fig(fig, plots_dir / f"14_temporal_channel_alpha_width_heatmap_{mode}.png"))
    elapsed = __import__('time').perf_counter() - start
    ok = summary_df[summary_df["status"] == "ok"].copy()
    summary = {"mode": mode, "n_channels": int(len(channels)), "analysis_samples": int(x.shape[0]), "sampling_rate_hz": fs, "scales_used": [int(v) for v in scales.tolist()], "q_count": int(len(q_vals)), "q_min": q_min, "q_max": q_max, "poly_order": poly_order, "temporal_stability": bool(temporal), "temporal_windows": int(cfg["temporal_windows"]), "mean_alpha_width": _safe_float(pd.to_numeric(ok.get("alpha_width", pd.Series(dtype=float)), errors="coerce").mean()) if not ok.empty else None, "median_alpha_width": _safe_float(pd.to_numeric(ok.get("alpha_width", pd.Series(dtype=float)), errors="coerce").median()) if not ok.empty else None, "mean_hq_q0": _safe_float(pd.to_numeric(ok.get("hq_q0", pd.Series(dtype=float)), errors="coerce").mean()) if not ok.empty else None, "mean_hq_fit_r2": _safe_float(pd.to_numeric(ok.get("mean_hq_fit_r2", pd.Series(dtype=float)), errors="coerce").mean()) if not ok.empty else None, "elapsed_sec": _safe_float(elapsed), "plot_count": len(plot_paths)}
    summary_txt = out_dir / f"mfdfa_summary_{mode}.txt"
    summary_txt.write_text("Manual Expert MFDFA Summary\n==========================\n\n" + f"Recording: {root}\nMode: {mode}\nSampling rate analysis Hz: {fs}\nScales used: {summary['scales_used']}\nq values: {q_min} to {q_max} ({len(q_vals)} values)\nPoly order: {poly_order}\nChannels analyzed: {len(channels)}\nMean alpha width: {summary['mean_alpha_width']}\nMean h(q=0): {summary['mean_hq_q0']}\nMean h(q) fit R2: {summary['mean_hq_fit_r2']}\n", encoding="utf-8")
    return {"manual_expert_mfdfa": {"summary": summary, "rows": _json_safe(summary_df.to_dict("records")), "hq_rows": _json_safe(hq_df.head(5000).to_dict("records")), "temporal_rows": _json_safe(temporal_df.to_dict("records")) if not temporal_df.empty else [], "asymmetry": _json_safe(asym), "plot_paths": plot_paths, "outputs": {"summary_csv": str(summary_csv), "hq_csv": str(hq_csv), "Fq_csv": str(Fq_csv), "temporal_csv": str(temporal_csv) if not temporal_df.empty else "", "npz": str(npz_path), "summary_txt": str(summary_txt), "plots_dir": str(plots_dir)}}}




def _mfdfa_viewer_config(mode: str) -> dict[str, Any]:
    mode = (mode or "fast").lower()
    if mode == "ultra":
        return {"max_images_to_show": 16, "plots_per_page": 8, "ncols": 2, "figure_scale": 0.85, "dpi_save": 170, "sort_mode": "alpha_width", "make_ranked_galleries": True, "make_metric_plots": True, "make_image_inventory_plots": True}
    if mode == "balanced":
        return {"max_images_to_show": None, "plots_per_page": 8, "ncols": 2, "figure_scale": 1.05, "dpi_save": 220, "sort_mode": "multifractal_complexity_score", "make_ranked_galleries": True, "make_metric_plots": True, "make_image_inventory_plots": True}
    if mode == "full":
        return {"max_images_to_show": None, "plots_per_page": 6, "ncols": 2, "figure_scale": 1.15, "dpi_save": 240, "sort_mode": "multifractal_complexity_score", "make_ranked_galleries": True, "make_metric_plots": True, "make_image_inventory_plots": True}
    return {"max_images_to_show": 32, "plots_per_page": 8, "ncols": 2, "figure_scale": 1.00, "dpi_save": 190, "sort_mode": "multifractal_complexity_score", "make_ranked_galleries": True, "make_metric_plots": True, "make_image_inventory_plots": True}


def _mfdfa_viewer_channel_from_filename(path: Path) -> str:
    name = path.name
    suffixes = [
        "_all_41_Fq_curves.png",
        "_signal_and_Fq_fast.png", "_signal_and_Fq_balanced.png", "_signal_and_Fq_full.png", "_signal_and_Fq_ultra.png", "_signal_and_Fq.png",
        "_hq_fast.png", "_hq_balanced.png", "_hq_full.png", "_hq_ultra.png", "_hq.png",
        "_multifractal_spectrum_fast.png", "_multifractal_spectrum_balanced.png", "_multifractal_spectrum_full.png", "_multifractal_spectrum_ultra.png", "_multifractal_spectrum.png",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name.replace(suffix, "")
    return path.stem


def _mfdfa_viewer_collect_manifest(img_dir: Path, plt: Any) -> tuple[Path | None, pd.DataFrame]:
    pngs = sorted(img_dir.glob("*.png"))
    contact_sheet_path: Path | None = None
    for candidate in (img_dir / "all_channels_all_41_Fq_contact_sheet.png", img_dir / "all_channels_contact_sheet.png"):
        if candidate.exists():
            contact_sheet_path = candidate
            break
    image_paths = [p for p in pngs if "contact_sheet" not in p.name.lower()]
    rows: list[dict[str, Any]] = []
    for path in image_paths:
        width = height = float("nan")
        try:
            img = plt.imread(str(path))
            height, width = img.shape[:2]
        except Exception:
            pass
        rows.append({
            "channel": _mfdfa_viewer_channel_from_filename(path),
            "image_path": str(path),
            "filename": path.name,
            "file_size_kb": _safe_float(path.stat().st_size / 1024.0) if path.exists() else None,
            "image_width_px": _safe_float(width),
            "image_height_px": _safe_float(height),
            "aspect_ratio": _safe_float(width / height) if np.isfinite(width) and np.isfinite(height) and height > 0 else None,
            "modified_time": _safe_float(path.stat().st_mtime) if path.exists() else None,
        })
    return contact_sheet_path, pd.DataFrame(rows)


def _mfdfa_viewer_load_summary(candidates: list[Path]) -> tuple[Path | None, pd.DataFrame]:
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "channel" in df.columns:
                    return path, df
            except Exception:
                pass
    return None, pd.DataFrame()


def _mfdfa_viewer_add_plot(plot_paths: list[dict[str, str]], title: str, path: str | None) -> None:
    if path:
        plot_paths.append({"title": title, "path": path, "url": f"/api/file?path={path}"})


def _mfdfa_plot_viewer(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "fast").lower()
    if mode not in ("ultra", "fast", "balanced", "full"):
        mode = "fast"
    cfg = _mfdfa_viewer_config(mode)
    for key in ("max_images_to_show", "plots_per_page", "ncols", "dpi_save"):
        if params.get(key) not in (None, ""):
            cfg[key] = _safe_int(params[key], int(cfg[key] or 1))
    if params.get("figure_scale") not in (None, ""):
        cfg["figure_scale"] = float(params["figure_scale"])
    if params.get("sort_mode") not in (None, ""):
        cfg["sort_mode"] = str(params["sort_mode"])
    run_if_missing = str(params.get("run_mfdfa_if_missing", "true")).lower() not in ("0", "false", "no", "off")
    root = Path(recording_dir).expanduser().resolve()
    mfdfa_root = root / "advanced_methods" / "manual_expert_mfdfa"
    plots_root = mfdfa_root / "plots"
    candidate_img_dirs = [
        plots_root / "per_channel",
        plots_root / "per_channel_all_41_Fq",
        root / "results" / "mfdfa_manual_expert_fast" / "plots" / "per_channel",
        root / "results" / "mfdfa_manual_expert_fast" / "plots" / "per_channel_all_41_Fq",
        root / "results" / "mfdfa_manual" / "plots" / "per_channel_all_41_Fq",
        root / "results" / "mfdfa_manual" / "plots" / "per_channel",
    ]

    def existing_img_dir() -> Path | None:
        found: list[tuple[Path, int]] = []
        for d in candidate_img_dirs:
            if d.is_dir():
                count = len([p for p in d.glob("*.png") if "contact_sheet" not in p.name.lower()])
                if count > 0:
                    found.append((d, count))
        if not found:
            return None
        return sorted(found, key=lambda item: item[1], reverse=True)[0][0]

    img_dir = existing_img_dir()
    generated_mfdfa = False
    if img_dir is None and run_if_missing:
        mfdfa_params = {
            "mode": mode,
            "save_per_channel_plots": "true",
            "temporal_stability": str(params.get("temporal_stability", "false")).lower() not in ("0", "false", "no", "off"),
        }
        for key in ("max_channels", "max_analysis_samples", "sampling_rate", "segment_start_sec", "segment_end_sec", "q_min", "q_max", "q_count", "poly_order"):
            if params.get(key) not in (None, ""):
                mfdfa_params[key] = params[key]
        _manual_expert_mfdfa(root, mfdfa_params)
        generated_mfdfa = True
        img_dir = existing_img_dir()
    if img_dir is None:
        raise FileNotFoundError("No per-channel MFDFA PNG files were found. Run Manual Expert MFDFA with per-channel plots enabled first, or set run_mfdfa_if_missing=true.")

    plt, cmap = _mfdfa_import_plotting()
    _mfdfa_apply_style(plt)
    viewer_out_dir = root / "advanced_methods" / "mfdfa_plot_viewer"
    viewer_plot_dir = viewer_out_dir / "plots"
    viewer_contact_dir = viewer_plot_dir / "contact_sheets"
    viewer_expert_dir = viewer_plot_dir / "expert_summary"
    for d in (viewer_out_dir, viewer_plot_dir, viewer_contact_dir, viewer_expert_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary_candidates = [
        mfdfa_root / f"mfdfa_summary_{mode}.csv",
        mfdfa_root / "mfdfa_summary_fast.csv",
        mfdfa_root / "mfdfa_summary_balanced.csv",
        mfdfa_root / "mfdfa_summary_full.csv",
        mfdfa_root / "mfdfa_summary_ultra.csv",
        root / "results" / "mfdfa_manual_expert_fast" / f"mfdfa_summary_{mode}.csv",
        root / "results" / "mfdfa_manual_expert_fast" / "mfdfa_summary_fast.csv",
        root / "results" / "mfdfa_manual_expert_fast" / "mfdfa_summary_balanced.csv",
        root / "results" / "mfdfa_manual_expert_fast" / "mfdfa_summary_full.csv",
        root / "results" / "mfdfa_manual_expert_fast" / "mfdfa_summary_ultra.csv",
        root / "results" / "mfdfa_manual" / "mfdfa_summary.csv",
    ]
    summary_csv_path, summary_df = _mfdfa_viewer_load_summary(summary_candidates)
    contact_sheet_path, manifest_df = _mfdfa_viewer_collect_manifest(img_dir, plt)
    if manifest_df.empty and contact_sheet_path is None:
        raise FileNotFoundError(f"No saved PNG plots found in: {img_dir}")

    display_df = manifest_df.copy()
    if not display_df.empty and not summary_df.empty and "channel" in summary_df.columns:
        cols_to_use = [c for c in summary_df.columns if c not in ("image_path", "filename")]
        display_df = display_df.merge(summary_df[cols_to_use], on="channel", how="left")
    sort_mode = str(cfg.get("sort_mode") or "multifractal_complexity_score")
    if not display_df.empty:
        if sort_mode in display_df.columns:
            display_df[sort_mode] = pd.to_numeric(display_df[sort_mode], errors="coerce")
            display_df = display_df.sort_values(sort_mode, ascending=False, na_position="last")
        elif "alpha_width" in display_df.columns:
            display_df["alpha_width"] = pd.to_numeric(display_df["alpha_width"], errors="coerce")
            display_df = display_df.sort_values("alpha_width", ascending=False, na_position="last")
        elif "file_size_kb" in display_df.columns:
            display_df = display_df.sort_values("file_size_kb", ascending=False, na_position="last")
        max_show = cfg.get("max_images_to_show")
        if max_show is not None:
            display_df = display_df.head(int(max_show)).copy()

    manifest_csv = viewer_out_dir / f"mfdfa_image_manifest_{mode}.csv"
    display_csv = viewer_out_dir / f"mfdfa_display_manifest_ranked_{mode}.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    display_df.to_csv(display_csv, index=False)

    dpi_save = int(cfg["dpi_save"])
    figure_scale = float(cfg["figure_scale"])
    ncols = int(cfg["ncols"])
    plots_per_page = int(cfg["plots_per_page"])
    plot_paths: list[dict[str, str]] = []

    def save_fig(fig: Any, path: Path) -> str:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi_save, bbox_inches="tight", facecolor=MFDFA_BLACK)
        plt.close(fig)
        return str(path)

    def image_from_path(path: str | Path) -> Any:
        return plt.imread(str(path))

    def make_contact_sheet_from_paths(df: pd.DataFrame, title: str, save_name: str, max_items: int | None = None) -> str | None:
        if df.empty:
            return None
        plot_df = df.head(max_items).copy() if max_items is not None else df.copy()
        if plot_df.empty:
            return None
        n = len(plot_df); nrows = int(math.ceil(n / ncols))
        fig_w = 7.0 * ncols * figure_scale; fig_h = 4.8 * nrows * figure_scale
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), facecolor=MFDFA_BLACK, squeeze=False)
        for ax in axes.ravel():
            ax.set_facecolor(MFDFA_BLACK); ax.axis("off")
        for ax, (_, row) in zip(axes.ravel(), plot_df.iterrows()):
            try:
                ax.imshow(image_from_path(row["image_path"]))
            except Exception as exc:
                ax.text(0.5, 0.5, f"Could not load image\n{type(exc).__name__}", transform=ax.transAxes, ha="center", va="center", color=MFDFA_ACCENT)
            parts = [str(row.get("channel", "image"))]
            for col, label in (("alpha_width", "Δα"), ("multifractal_complexity_score", "score"), ("mean_hq_fit_r2", "R²")):
                val = _safe_float(row.get(col))
                if val is not None:
                    parts.append(f"{label}={val:.3f}" if label != "R²" else f"{label}={val:.2f}")
            ax.set_title(" | ".join(parts), color=MFDFA_ACCENT, fontsize=10)
            ax.axis("off")
        fig.suptitle(title, color=MFDFA_ACCENT, fontsize=16, y=0.995)
        return save_fig(fig, viewer_contact_dir / save_name)

    if contact_sheet_path is not None and contact_sheet_path.exists():
        try:
            fig, ax = plt.subplots(figsize=(16 * figure_scale, 20 * figure_scale), facecolor=MFDFA_BLACK)
            ax.imshow(image_from_path(contact_sheet_path)); ax.axis("off"); ax.set_title("Existing Saved Contact Sheet", color=MFDFA_ACCENT, pad=14)
            _mfdfa_viewer_add_plot(plot_paths, "Existing saved contact sheet", save_fig(fig, viewer_contact_dir / f"00_existing_contact_sheet_viewed_{mode}.png"))
        except Exception:
            pass

    _mfdfa_viewer_add_plot(plot_paths, "Ranked MFDFA per-channel contact sheet", make_contact_sheet_from_paths(display_df, f"Ranked MFDFA Per-Channel Plot Contact Sheet | sorted by {sort_mode} | {mode}", f"01_ranked_contact_sheet_{mode}.png", cfg.get("max_images_to_show")))

    if cfg.get("make_ranked_galleries") and not display_df.empty and "alpha_width" in display_df.columns:
        top_width = display_df.copy(); top_width["alpha_width"] = pd.to_numeric(top_width["alpha_width"], errors="coerce"); top_width = top_width.sort_values("alpha_width", ascending=False, na_position="last")
        _mfdfa_viewer_add_plot(plot_paths, "Top MFDFA images by alpha width", make_contact_sheet_from_paths(top_width, f"Top MFDFA Images by Alpha Width Δα | {mode}", f"02_top_alpha_width_contact_sheet_{mode}.png", min(12, len(top_width))))
    if cfg.get("make_ranked_galleries") and not display_df.empty and "asymmetry" in display_df.columns:
        top_asym = display_df.copy(); top_asym["abs_asymmetry"] = np.abs(pd.to_numeric(top_asym["asymmetry"], errors="coerce")); top_asym = top_asym.sort_values("abs_asymmetry", ascending=False, na_position="last")
        _mfdfa_viewer_add_plot(plot_paths, "Top MFDFA images by absolute asymmetry", make_contact_sheet_from_paths(top_asym, f"Top MFDFA Images by Absolute Spectrum Asymmetry | {mode}", f"03_top_asymmetry_contact_sheet_{mode}.png", min(12, len(top_asym))))

    # Individual pages, exactly like the notebook viewer.
    page_paths: list[str] = []
    if not display_df.empty:
        for start in range(0, len(display_df), plots_per_page):
            batch = display_df.iloc[start:start + plots_per_page].copy(); page_num = start // plots_per_page + 1
            nrows = int(math.ceil(len(batch) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(14 * figure_scale, 5.2 * nrows * figure_scale), facecolor=MFDFA_BLACK, squeeze=False)
            for ax in axes.ravel():
                ax.set_facecolor(MFDFA_BLACK); ax.axis("off")
            for ax, (_, row) in zip(axes.ravel(), batch.iterrows()):
                try:
                    ax.imshow(image_from_path(row["image_path"]))
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Could not load\n{Path(str(row.get('image_path',''))).name}\n{type(exc).__name__}", transform=ax.transAxes, ha="center", va="center", color=MFDFA_ACCENT)
                title = str(row.get("channel", "image")); bits=[]
                for col, label in (("alpha_width","Δα"), ("asymmetry","asym"), ("hq_q0","h0"), ("mean_hq_fit_r2","R²")):
                    val=_safe_float(row.get(col))
                    if val is not None: bits.append(f"{label}={val:.3f}")
                if bits: title += " | " + ", ".join(bits)
                ax.set_title(title, color=MFDFA_ACCENT, fontsize=10); ax.axis("off")
            fig.suptitle(f"MFDFA Per-Channel Plot Page {page_num} | FAST_OPTION={mode}", color=MFDFA_ACCENT, fontsize=15, y=0.995)
            path = save_fig(fig, viewer_contact_dir / f"individual_page_{page_num:02d}_{mode}.png")
            page_paths.append(path)
            _mfdfa_viewer_add_plot(plot_paths, f"Individual per-channel page {page_num}", path)

    if cfg.get("make_image_inventory_plots") and not manifest_df.empty:
        df_inv = manifest_df.copy(); df_inv["file_size_kb"] = pd.to_numeric(df_inv["file_size_kb"], errors="coerce"); df_inv = df_inv.sort_values("file_size_kb", ascending=True, na_position="first")
        fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(df_inv)))); y = np.arange(len(df_inv))
        ax.barh(y, df_inv["file_size_kb"].fillna(0.0), color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.84)
        ax.set_yticks(y); ax.set_yticklabels(df_inv["channel"], color=MFDFA_ACCENT, fontsize=8)
        for yi, (_, row) in zip(y, df_inv.iterrows()):
            w = _safe_float(row.get("image_width_px")); h = _safe_float(row.get("image_height_px")); label = f"  {int(w)}x{int(h)}" if w is not None and h is not None else ""
            ax.text(row.get("file_size_kb") if _safe_float(row.get("file_size_kb")) is not None else 0.0, yi, label, va="center", color=MFDFA_ACCENT_SOFT, fontsize=7)
        _mfdfa_style_ax(ax, "MFDFA Saved Image Inventory", "File size (KB)", "Channel / image")
        _mfdfa_viewer_add_plot(plot_paths, "MFDFA saved image inventory", save_fig(fig, viewer_expert_dir / f"image_inventory_{mode}.png"))

    def plot_metric_ranking(summary: pd.DataFrame, metric_col: str, title: str, xlabel: str, save_name: str) -> str | None:
        if summary.empty or metric_col not in summary.columns: return None
        df = summary.copy(); df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce"); df = df[np.isfinite(df[metric_col])].sort_values(metric_col, ascending=True)
        if df.empty: return None
        max_show = cfg.get("max_images_to_show")
        if max_show is not None: df = df.tail(int(max_show)).copy()
        fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(df)))); y = np.arange(len(df))
        ax.barh(y, df[metric_col], color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.86)
        ax.set_yticks(y); ax.set_yticklabels(df["channel"], color=MFDFA_ACCENT, fontsize=8)
        mean_val = df[metric_col].mean(); ax.axvline(mean_val, color=MFDFA_WHITE, linestyle="--", linewidth=1.0, alpha=0.80, label=f"Mean={mean_val:.3f}")
        _mfdfa_style_ax(ax, title, xlabel, "Channel"); ax.legend(loc="best", frameon=True)
        return save_fig(fig, viewer_expert_dir / save_name)

    if cfg.get("make_metric_plots") and not summary_df.empty:
        _mfdfa_viewer_add_plot(plot_paths, "MFDFA alpha width ranking", plot_metric_ranking(summary_df, "alpha_width", "MFDFA Alpha Width Ranking", "Alpha width Δα", f"alpha_width_ranking_{mode}.png"))
        _mfdfa_viewer_add_plot(plot_paths, "MFDFA spectrum asymmetry ranking", plot_metric_ranking(summary_df, "asymmetry", "MFDFA Spectrum Asymmetry Ranking", "Asymmetry", f"asymmetry_ranking_{mode}.png"))
        if "multifractal_complexity_score" in summary_df.columns:
            _mfdfa_viewer_add_plot(plot_paths, "MFDFA composite complexity score ranking", plot_metric_ranking(summary_df, "multifractal_complexity_score", "MFDFA Composite Complexity Score Ranking", "Multifractal complexity score", f"complexity_score_ranking_{mode}.png"))

        metric_cols = [c for c in ["alpha_width", "asymmetry", "hq_qmin", "hq_q0", "hq_qmax", "hq_range", "mean_hq_fit_r2", "multifractal_complexity_score"] if c in summary_df.columns]
        if metric_cols:
            dfh = summary_df.copy()
            if sort_mode in dfh.columns:
                dfh[sort_mode] = pd.to_numeric(dfh[sort_mode], errors="coerce"); dfh = dfh.sort_values(sort_mode, ascending=False, na_position="last")
            elif "alpha_width" in dfh.columns:
                dfh["alpha_width"] = pd.to_numeric(dfh["alpha_width"], errors="coerce"); dfh = dfh.sort_values("alpha_width", ascending=False, na_position="last")
            if cfg.get("max_images_to_show") is not None: dfh = dfh.head(int(cfg["max_images_to_show"])).copy()
            M = dfh[metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float); M_norm = np.zeros_like(M, dtype=float)
            for j in range(M.shape[1]):
                col = M[:, j]; finite = col[np.isfinite(col)]
                if finite.size and np.nanmax(finite) != np.nanmin(finite): M_norm[:, j] = (col - np.nanmin(finite)) / (np.nanmax(finite) - np.nanmin(finite) + 1e-12)
            fig, ax = plt.subplots(figsize=(12, max(7, 0.30 * len(dfh))))
            im = ax.imshow(M_norm, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(metric_cols))); ax.set_xticklabels(metric_cols, rotation=50, ha="right", color=MFDFA_ACCENT, fontsize=8)
            ax.set_yticks(np.arange(len(dfh))); ax.set_yticklabels(dfh["channel"], color=MFDFA_ACCENT, fontsize=8)
            _mfdfa_style_ax(ax, "MFDFA Summary Metric Heatmap for Displayed Images", "Metric", "Channel", False)
            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cbar.ax.yaxis.set_tick_params(color=MFDFA_ACCENT); [lab.set_color(MFDFA_ACCENT) for lab in cbar.ax.get_yticklabels()]; cbar.outline.set_edgecolor(MFDFA_ACCENT); cbar.set_label("Column-normalized value", color=MFDFA_ACCENT)
            _mfdfa_viewer_add_plot(plot_paths, "MFDFA summary metric heatmap", save_fig(fig, viewer_expert_dir / f"summary_metric_heatmap_{mode}.png"))

        if all(c in summary_df.columns for c in ("alpha_width", "asymmetry")):
            dfs = summary_df.copy(); dfs["alpha_width"] = pd.to_numeric(dfs["alpha_width"], errors="coerce"); dfs["asymmetry"] = pd.to_numeric(dfs["asymmetry"], errors="coerce"); dfs = dfs[np.isfinite(dfs["alpha_width"]) & np.isfinite(dfs["asymmetry"])]
            if not dfs.empty:
                sizes = 70 + 420 * np.clip(pd.to_numeric(dfs.get("mean_hq_fit_r2", 0), errors="coerce").fillna(0).to_numpy(dtype=float), 0, 1)
                fig, ax = plt.subplots(figsize=(9, 7)); ax.scatter(dfs["asymmetry"], dfs["alpha_width"], s=sizes, c=MFDFA_ACCENT, edgecolors=MFDFA_WHITE, linewidths=0.8, alpha=0.82)
                ax.axvline(0, color=MFDFA_WHITE, linestyle="--", linewidth=1.0, alpha=0.75)
                rank_col = sort_mode if sort_mode in dfs.columns else "alpha_width"
                if rank_col in dfs.columns:
                    dfs[rank_col] = pd.to_numeric(dfs[rank_col], errors="coerce")
                    label_df = dfs.sort_values(rank_col, ascending=False, na_position="last").head(10)
                else:
                    label_df = dfs.head(10)
                for _, row in label_df.iterrows(): ax.annotate(str(row["channel"]), (row["asymmetry"], row["alpha_width"]), xytext=(5, 5), textcoords="offset points", color=MFDFA_ACCENT_SOFT, fontsize=8)
                _mfdfa_style_ax(ax, "MFDFA Spectrum Width vs Asymmetry", "Spectrum asymmetry", "Alpha width Δα")
                _mfdfa_viewer_add_plot(plot_paths, "MFDFA spectrum width vs asymmetry", save_fig(fig, viewer_expert_dir / f"width_vs_asymmetry_scatter_{mode}.png"))

    elapsed = __import__('time').perf_counter()
    summary = {
        "mode": mode,
        "image_dir": str(img_dir),
        "summary_csv": str(summary_csv_path) if summary_csv_path else "",
        "images_found": int(len(manifest_df)),
        "images_displayed": int(len(display_df)),
        "sort_mode": sort_mode,
        "generated_mfdfa_first": bool(generated_mfdfa),
        "viewer_plot_count": int(len(plot_paths)),
    }
    summary_txt = viewer_out_dir / f"mfdfa_plot_viewer_summary_{mode}.txt"
    lines = ["MFDFA Plot Viewer Summary", "=========================", "", f"FAST_OPTION: {mode}", f"Image directory: {img_dir}", f"Images found: {len(manifest_df)}", f"Images displayed: {len(display_df)}", f"Summary CSV: {summary_csv_path if summary_csv_path else 'None'}", f"Manifest CSV: {manifest_csv}", f"Ranked display CSV: {display_csv}", f"Sort mode: {sort_mode}", f"Generated MFDFA first: {generated_mfdfa}", "", "Saved viewer plots:"]
    lines.extend([p["path"] for p in plot_paths])
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["summary_txt"] = str(summary_txt)
    top_rows: list[dict[str, Any]] = []
    if not summary_df.empty:
        rank_col = sort_mode if sort_mode in summary_df.columns else "alpha_width"
        cols = [c for c in ["channel", "alpha_width", "asymmetry", "hq_q0", "mean_hq_fit_r2", "multifractal_complexity_score"] if c in summary_df.columns]
        sdf = summary_df.copy()
        if rank_col in sdf.columns:
            sdf[rank_col] = pd.to_numeric(sdf[rank_col], errors="coerce"); sdf = sdf.sort_values(rank_col, ascending=False, na_position="last")
        top_rows = _json_safe(sdf[cols].head(15).to_dict("records"))
    return {"mfdfa_plot_viewer": {"summary": summary, "manifest_rows": _json_safe(manifest_df.head(5000).to_dict("records")), "display_rows": _json_safe(display_df.to_dict("records")), "top_rows": top_rows, "plot_paths": plot_paths, "outputs": {"manifest_csv": str(manifest_csv), "display_csv": str(display_csv), "summary_txt": str(summary_txt), "viewer_plots_dir": str(viewer_plot_dir)}}}



# ---- v0.11.22 Manual MFDFA Spectrum analysis ----
# Adapted from the user's compact MFDFA spectrum notebook. This is intentionally
# separate from Manual Expert MFDFA: it focuses on per-channel triptych plots of
# h(q), tau(q), and f(alpha), plus the alpha-width ranking barplot.
MFDFA_SPECTRUM_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {
        "scale_min_power": 0.8, "scale_max_power": 3.45, "n_scales": 14,
        "q_min": -4.0, "q_max": 4.0, "q_count": 17,
        "poly_order": 1, "max_analysis_samples": 90_000,
    },
    "fast": {
        "scale_min_power": 0.7, "scale_max_power": 3.75, "n_scales": 20,
        "q_min": -5.0, "q_max": 5.0, "q_count": 25,
        "poly_order": 1, "max_analysis_samples": 120_000,
    },
    "balanced": {
        # Matches the notebook code parameters: logspace(0.7, 4, 30), q=-5..5 with 41 points.
        "scale_min_power": 0.7, "scale_max_power": 4.0, "n_scales": 30,
        "q_min": -5.0, "q_max": 5.0, "q_count": 41,
        "poly_order": 1, "max_analysis_samples": 180_000,
    },
    "full": {
        "scale_min_power": 0.7, "scale_max_power": 4.0, "n_scales": 36,
        "q_min": -6.0, "q_max": 6.0, "q_count": 49,
        "poly_order": 2, "max_analysis_samples": 240_000,
    },
}


def _manual_mfdfa_spectrum(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "ultra").lower()
    if mode not in MFDFA_SPECTRUM_CONFIGS:
        mode = "ultra"
    cfg = dict(MFDFA_SPECTRUM_CONFIGS[mode])
    for key in ("scale_min_power", "scale_max_power"):
        if params.get(key) not in (None, ""):
            cfg[key] = float(params[key])
    for key in ("n_scales", "poly_order", "max_analysis_samples"):
        if params.get(key) not in (None, ""):
            cfg[key] = _safe_int(params[key], int(cfg[key] or 1))
    q_min = float(params.get("q_min") if params.get("q_min") not in (None, "") else cfg["q_min"])
    q_max = float(params.get("q_max") if params.get("q_max") not in (None, "") else cfg["q_max"])
    q_count = _safe_int(params.get("q_count"), int(cfg["q_count"]))
    q_vals = np.linspace(q_min, q_max, q_count)
    max_channels = _safe_int(params.get("max_channels"), 999999) if params.get("max_channels") not in (None, "") else None

    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=cfg.get("max_analysis_samples"))
    root = Path(rec["recording_dir"])
    x = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels is not None:
        channels = channels[:max_channels]
        x = x[:, :len(channels)]
    segment_start = _safe_float(params.get("segment_start_sec"))
    segment_end = _safe_float(params.get("segment_end_sec"))
    if segment_start is not None or segment_end is not None:
        s0 = int(max(0, (segment_start or 0.0) * fs))
        s1 = int(min(x.shape[0], (segment_end if segment_end is not None else (x.shape[0] / fs)) * fs))
        if s1 > s0 + 32:
            x = x[s0:s1, :]

    poly_order = int(cfg["poly_order"])
    scales = _mfdfa_make_scales(x.shape[0], cfg, poly_order)
    plt, _ = _mfdfa_import_plotting()
    _mfdfa_apply_style(plt)
    out_dir = root / "advanced_methods" / "manual_mfdfa_spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    per_channel_npz: list[str] = []
    plot_paths: list[dict[str, str]] = []
    start = __import__('time').perf_counter()

    for ch_idx, channel in enumerate(channels):
        sig = x[:, ch_idx].astype(float)
        res = _mfdfa_channel(sig, scales, q_vals, poly_order)
        summ = dict(res["summary"])
        row = {
            "channel": channel,
            "alpha_min": summ.get("alpha_min"),
            "alpha_max": summ.get("alpha_max"),
            "alpha_width": summ.get("alpha_width"),
            "alpha_at_peak": summ.get("alpha_at_peak"),
            "f_peak": summ.get("f_peak"),
            "asymmetry": summ.get("asymmetry"),
            "hq_q0": summ.get("hq_q0"),
            "mean_hq_fit_r2": summ.get("mean_hq_fit_r2"),
            "multifractal_complexity_score": summ.get("multifractal_complexity_score"),
        }
        summary_rows.append(row)

        npz_path = out_dir / f"{channel}_mfdfa_manual_spectrum.npz"
        np.savez_compressed(
            npz_path,
            channel=channel,
            sampling_rate=fs,
            scales=res["scales"],
            q_vals=q_vals,
            Fq=res["Fq"],
            hq=res["hq"],
            tau=res["tau_q"],
            alpha=res["alpha_q"],
            f_alpha=res["f_alpha_q"],
            alpha_min=row["alpha_min"],
            alpha_max=row["alpha_max"],
            alpha_width=row["alpha_width"],
            alpha_at_peak=row["alpha_at_peak"],
            f_peak=row["f_peak"],
            asymmetry=row["asymmetry"],
        )
        per_channel_npz.append(str(npz_path))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=MFDFA_BLACK)
        for ax in axes:
            _mfdfa_style_ax(ax)
        hq = np.asarray(res["hq"], dtype=float)
        tau = np.asarray(res["tau_q"], dtype=float)
        alpha = np.asarray(res["alpha_q"], dtype=float)
        f_alpha = np.asarray(res["f_alpha_q"], dtype=float)
        axes[0].plot(q_vals, hq, color=MFDFA_ACCENT, marker="o", markerfacecolor=MFDFA_ACCENT, markeredgecolor=MFDFA_ACCENT)
        _mfdfa_style_ax(axes[0], f"{channel}: h(q)", "q", "h(q)")
        axes[1].plot(q_vals, tau, color=MFDFA_ACCENT, marker="o", markerfacecolor=MFDFA_ACCENT, markeredgecolor=MFDFA_ACCENT)
        _mfdfa_style_ax(axes[1], f"{channel}: tau(q)", "q", "tau(q)")
        mask = np.isfinite(alpha) & np.isfinite(f_alpha)
        axes[2].plot(alpha[mask], f_alpha[mask], color=MFDFA_ACCENT, marker="o", markerfacecolor=MFDFA_ACCENT, markeredgecolor=MFDFA_ACCENT)
        _mfdfa_style_ax(axes[2], f"{channel}: singularity spectrum", "alpha", "f(alpha)")
        fig_path = out_dir / f"{channel}_mfdfa_spectrum.png"
        plot_paths.append({"title": f"{channel}: h(q), tau(q), and singularity spectrum", "path": _mfdfa_save_fig(fig, fig_path, dpi=150)})

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "mfdfa_manual_spectrum_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=MFDFA_BLACK)
    plot_df = summary_df.copy()
    ax.bar(plot_df["channel"], pd.to_numeric(plot_df["alpha_width"], errors="coerce"), color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT)
    _mfdfa_style_ax(ax, "Multifractal spectrum width by channel", "Channel", "Spectrum width (alpha_max - alpha_min)")
    ax.tick_params(axis="x", rotation=90, colors=MFDFA_ACCENT)
    width_plot = out_dir / "mfdfa_spectrum_width_by_channel.png"
    plot_paths.append({"title": "Multifractal spectrum width by channel", "path": _mfdfa_save_fig(fig, width_plot, dpi=150)})

    summary_txt = out_dir / "mfdfa_manual_spectrum_summary.txt"
    alpha_vals = pd.to_numeric(summary_df.get("alpha_width"), errors="coerce") if "alpha_width" in summary_df else pd.Series(dtype=float)
    lines = [
        "Manual MFDFA Spectrum Summary",
        "=============================",
        "",
        f"Recording: {root}",
        f"Mode: {mode}",
        f"Sampling rate: {fs}",
        f"Channels analyzed: {len(channels)}",
        f"Scales used: {scales.tolist()}",
        f"q values: {float(q_vals.min())} to {float(q_vals.max())} ({len(q_vals)} values)",
        f"Mean alpha width: {float(np.nanmean(alpha_vals)) if len(alpha_vals) else np.nan}",
        "",
        "Per-channel summary:",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"{row['channel']}: alpha_width={row.get('alpha_width')}, asymmetry={row.get('asymmetry')}, hq_q0={row.get('hq_q0')}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    elapsed = __import__('time').perf_counter() - start
    summary = {
        "mode": mode,
        "n_channels": int(len(channels)),
        "analysis_samples": int(x.shape[0]),
        "sampling_rate_hz": fs,
        "scale_count": int(len(scales)),
        "q_count": int(len(q_vals)),
        "poly_order": poly_order,
        "mean_alpha_width": _safe_float(np.nanmean(alpha_vals)) if len(alpha_vals) else None,
        "median_alpha_width": _safe_float(np.nanmedian(alpha_vals)) if len(alpha_vals) else None,
        "elapsed_sec": _safe_float(elapsed),
    }
    return {"manual_mfdfa_spectrum": {"summary": summary, "rows": _json_safe(summary_df.to_dict("records")), "plot_paths": plot_paths, "per_channel_npz": per_channel_npz, "outputs": {"summary_csv": str(summary_csv), "summary_txt": str(summary_txt), "output_dir": str(out_dir)}}}

def _method_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "higuchi_fractal_dimension",
            "name": "Higuchi fractal dimension",
            "description": "Computes per-channel Higuchi fractal dimension with log-log fit diagnostics, scalp/regional/asymmetry summaries, and rolling temporal-stability plots.",
            "panel": {"kind": "custom", "field": "higuchi_fractal_dimension", "title": "Higuchi Fractal Dimension"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""},
                {"name": "k_max", "label": "k max override", "type": "number", "default": ""},
                {"name": "decimate", "label": "Decimate override", "type": "number", "default": ""},
                {"name": "max_samples", "label": "Max samples override", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "rolling", "label": "Rolling stability", "type": "select", "default": "true", "options": ["true", "false"]},
                {"name": "rolling_window_sec", "label": "Rolling window sec", "type": "number", "default": ""},
                {"name": "rolling_step_sec", "label": "Rolling step sec", "type": "number", "default": ""},
                {"name": "rolling_max_windows", "label": "Rolling max windows", "type": "number", "default": ""},
            ],
        },
        {
            "id": "embedded_fractal_dimension",
            "name": "Embedded fractal dimension",
            "description": "Computes box-counting fractal dimension and sampled Grassberger-Procaccia correlation dimension D2 across 2D..10D attractor embeddings, using precomputed embedding files when present or generated delay embeddings from signal.npy.",
            "panel": {"kind": "custom", "field": "embedded_fractal_dimension", "title": "Embedded Fractal Dimension"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "embedding_dims", "label": "Embedding dimensions comma list", "type": "text", "default": "2,3,4,5,6,7,8,9,10"},
                {"name": "tau_ms", "label": "Delay tau ms", "type": "number", "default": 10.0},
                {"name": "tau_samples", "label": "Delay tau samples override", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": 120000},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "random_seed", "label": "Random seed", "type": "number", "default": 0}
            ],
        },
        {
            "id": "dimension_saturation_profiling",
            "name": "Dimension saturation profiling",
            "description": "Profiles whether embedded correlation dimension D2 saturates across embedding dimensions, with ΔD2 heatmaps, saturation markers, tail slopes, and class counts using the same black/cyan notebook style.",
            "panel": {"kind": "custom", "field": "dimension_saturation_profiling", "title": "Dimension Saturation Profiling"},
            "parameters": [
                {"name": "mode", "label": "Embedded FD mode if it must run first", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "use_metric", "label": "Metric", "type": "select", "default": "corr_dim_d2", "options": ["corr_dim_d2", "boxcount_fd"]},
                {"name": "abs_delta_thresh", "label": "Absolute Δ threshold", "type": "number", "default": 0.12},
                {"name": "rel_delta_thresh", "label": "Relative Δ threshold", "type": "number", "default": 0.04},
                {"name": "n_consecutive", "label": "Consecutive small increments", "type": "number", "default": 2},
                {"name": "tail_points", "label": "Tail points for slope", "type": "number", "default": 3},
                {"name": "run_embedded_if_missing", "label": "Run Embedded FD if missing", "type": "select", "default": "true", "options": ["true", "false"]},
                {"name": "embedding_dims", "label": "Embedding dimensions comma list", "type": "text", "default": "2,3,4,5,6,7,8,9,10"},
                {"name": "tau_ms", "label": "Delay tau ms", "type": "number", "default": 10.0},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": 120000},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "random_seed", "label": "Random seed", "type": "number", "default": 0}
            ],
        },
        {
            "id": "katz_fractal_dimension",
            "name": "Katz fractal dimension",
            "description": "Computes Katz fractal dimension for each embedded trajectory across embedding dimensions, using precomputed embedding .npy files when present or generated delay embeddings from signal.npy. Plots follow the same black/cyan notebook style.",
            "panel": {"kind": "custom", "field": "katz_fractal_dimension", "title": "Katz Fractal Dimension"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "embedding_dims", "label": "Embedding dimensions comma list", "type": "text", "default": "2,3,4,5,6,7,8,9,10"},
                {"name": "tau_ms", "label": "Delay tau ms", "type": "number", "default": 10.0},
                {"name": "tau_samples", "label": "Delay tau samples override", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": 120000},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "random_seed", "label": "Random seed", "type": "number", "default": 0}
            ],
        },
        {
            "id": "manual_mfdfa_spectrum",
            "name": "Manual MFDFA Spectrum",
            "description": "Compact manual MFDFA spectrum workflow from signal.npy. It computes Fq(s), h(q), tau(q), alpha/f(alpha), per-channel NPZ outputs, triptych h/tau/spectrum plots, and alpha-width ranking in the black/cyan notebook style.",
            "panel": {"kind": "custom", "field": "manual_mfdfa_spectrum", "title": "Manual MFDFA Spectrum"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "q_min", "label": "q min override", "type": "number", "default": ""},
                {"name": "q_max", "label": "q max override", "type": "number", "default": ""},
                {"name": "q_count", "label": "q count override", "type": "number", "default": ""},
                {"name": "poly_order", "label": "Polynomial order", "type": "number", "default": ""},
                {"name": "scale_min_power", "label": "Scale min power", "type": "number", "default": ""},
                {"name": "scale_max_power", "label": "Scale max power", "type": "number", "default": ""},
                {"name": "n_scales", "label": "Scale count", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""}
            ],
        },
        {
            "id": "manual_mfdfa_shuffle_surrogate",
            "name": "MFDFA shuffle surrogate",
            "description": "Compares original multifractal spectrum width against shuffled surrogates for every available channel by default. Generates per-channel h(q), f(alpha), shuffle-width histograms, original-vs-shuffle bars, and width-excess rankings in the black/cyan notebook style.",
            "panel": {"kind": "custom", "field": "manual_mfdfa_shuffle_surrogate", "title": "MFDFA Shuffle Surrogate"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "n_shuffles", "label": "Shuffle count", "type": "number", "default": ""},
                {"name": "random_seed", "label": "Random seed", "type": "number", "default": 0},
                {"name": "q_min", "label": "q min override", "type": "number", "default": ""},
                {"name": "q_max", "label": "q max override", "type": "number", "default": ""},
                {"name": "q_count", "label": "q count override", "type": "number", "default": ""},
                {"name": "poly_order", "label": "Polynomial order", "type": "number", "default": ""},
                {"name": "scale_min_power", "label": "Scale min power", "type": "number", "default": ""},
                {"name": "scale_max_power", "label": "Scale max power", "type": "number", "default": ""},
                {"name": "n_scales", "label": "Scale count", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""}
            ],
        },
        {
            "id": "manual_mfdfa_iaaft_surrogate",
            "name": "MFDFA IAAFT surrogate",
            "description": "Compares original multifractal spectrum width against IAAFT surrogates for every available channel by default. Generates per-channel h(q), f(alpha), IAAFT alpha-width histograms, original-vs-IAAFT bars, and width-excess rankings in the black/cyan notebook style.",
            "panel": {"kind": "custom", "field": "manual_mfdfa_iaaft_surrogate", "title": "MFDFA IAAFT Surrogate"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "n_surrogates", "label": "IAAFT surrogate count", "type": "number", "default": ""},
                {"name": "iaaft_iters", "label": "IAAFT iterations", "type": "number", "default": ""},
                {"name": "random_seed", "label": "Random seed", "type": "number", "default": 0},
                {"name": "q_min", "label": "q min override", "type": "number", "default": ""},
                {"name": "q_max", "label": "q max override", "type": "number", "default": ""},
                {"name": "q_count", "label": "q count override", "type": "number", "default": ""},
                {"name": "poly_order", "label": "Polynomial order", "type": "number", "default": ""},
                {"name": "scale_min_power", "label": "Scale min power", "type": "number", "default": ""},
                {"name": "scale_max_power", "label": "Scale max power", "type": "number", "default": ""},
                {"name": "n_scales", "label": "Scale count", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""}
            ],
        },
        {
            "id": "wavelet_leader_multifractal",
            "name": "Wavelet leader multifractal",
            "description": "Manual Haar wavelet-leader multifractal analysis for all available channels by default. Computes leader structure functions, zeta(q), alpha(q), f(alpha), per-channel NPZ outputs, spectrum triptychs, and alpha-width rankings in the black/cyan notebook style.",
            "panel": {"kind": "custom", "field": "wavelet_leader_multifractal", "title": "Wavelet Leader Multifractal"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "q_min", "label": "q min override", "type": "number", "default": ""},
                {"name": "q_max", "label": "q max override", "type": "number", "default": ""},
                {"name": "q_count", "label": "q count override", "type": "number", "default": ""},
                {"name": "fit_j_min", "label": "Fit j min", "type": "number", "default": ""},
                {"name": "fit_j_max", "label": "Fit j max", "type": "number", "default": ""},
                {"name": "min_leaders_per_scale", "label": "Min leaders per scale", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""}
            ],
        },

        {
            "id": "lyapunov_spectrum_custom",
            "name": "Lyapunov spectrum custom",
            "description": "Computes local Jacobian/QR Lyapunov spectra from precomputed or generated delay embeddings for all available channels by default. Adds expert chaos metrics and the same black/cyan LLE, spectrum, complexity, runtime, correlation, and positive-exponent plots from the notebook workflow.",
            "panel": {"kind": "custom", "field": "lyapunov_spectrum_custom", "title": "Lyapunov Spectrum"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "embedding_dims", "label": "Embedding dimensions comma list", "type": "text", "default": "2,3"},
                {"name": "tau_ms", "label": "Delay tau ms", "type": "number", "default": 10.0},
                {"name": "tau_samples", "label": "Delay tau samples override", "type": "number", "default": ""},
                {"name": "k_neighbors", "label": "k neighbors", "type": "number", "default": ""},
                {"name": "theiler", "label": "Theiler window", "type": "number", "default": ""},
                {"name": "stride", "label": "Anchor stride", "type": "number", "default": ""},
                {"name": "max_steps_per_file", "label": "Max steps per embedding", "type": "number", "default": ""},
                {"name": "max_points", "label": "Max points per embedding", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_files", "label": "Max embedding files", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""},
                {"name": "random_seed", "label": "Random seed", "type": "number", "default": 0}
            ],
        },

        {
            "id": "arnold_tongues_kuramoto",
            "name": "Arnold tongues / Kuramoto",
            "description": "Builds EEG-derived natural frequencies from dominant channel FFT peaks, runs a phase-forced Kuramoto model over an Arnold tongue amplitude/frequency grid, and saves drive-locking, collective locking, frequency-error, synchronization, contrast, frequency, omega, and line-summary black/cyan plots.",
            "panel": {"kind": "custom", "field": "arnold_tongues_kuramoto", "title": "Arnold Tongues / Kuramoto"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "K", "label": "Kuramoto coupling K", "type": "number", "default": ""},
                {"name": "a_min", "label": "Drive amplitude min", "type": "number", "default": ""},
                {"name": "a_max", "label": "Drive amplitude max", "type": "number", "default": ""},
                {"name": "a_count", "label": "Drive amplitude grid count", "type": "number", "default": ""},
                {"name": "b_min", "label": "Drive frequency min", "type": "number", "default": ""},
                {"name": "b_max", "label": "Drive frequency max", "type": "number", "default": ""},
                {"name": "b_count", "label": "Drive frequency grid count", "type": "number", "default": ""},
                {"name": "t_end", "label": "Simulation end time", "type": "number", "default": ""},
                {"name": "n_t_eval", "label": "Time samples", "type": "number", "default": ""},
                {"name": "transient_fraction", "label": "Transient fraction", "type": "number", "default": ""},
                {"name": "freq_min", "label": "Dominant-frequency band min Hz", "type": "number", "default": ""},
                {"name": "freq_max", "label": "Dominant-frequency band max Hz", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""},
                {"name": "random_seed", "label": "Random seed", "type": "number", "default": 42}
            ],
        },
        {
            "id": "circle_map_arnold_tongues",
            "name": "Circle map Arnold tongues",
            "description": "Computes EEG-derived circular mean phase with the Hilbert transform, runs a vectorized circle-map fixed-point-locking grid over Ω and K, and saves all black/cyan plots from the notebook: locking heatmaps, theoretical boundaries, mean locking summaries, phase histogram/samples, and thresholded locking map.",
            "panel": {"kind": "custom", "field": "circle_map_arnold_tongues", "title": "Circle Map Arnold Tongues"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "n_omega", "label": "Omega grid count", "type": "number", "default": ""},
                {"name": "n_K", "label": "K grid count", "type": "number", "default": ""},
                {"name": "omega_min", "label": "Omega min", "type": "number", "default": ""},
                {"name": "omega_max", "label": "Omega max", "type": "number", "default": ""},
                {"name": "K_min", "label": "K min", "type": "number", "default": ""},
                {"name": "K_max", "label": "K max", "type": "number", "default": ""},
                {"name": "iterations", "label": "Circle-map iterations", "type": "number", "default": ""},
                {"name": "tol", "label": "Fixed-point tolerance", "type": "number", "default": ""},
                {"name": "max_phase_samples", "label": "Max phase samples", "type": "number", "default": ""},
                {"name": "lock_threshold", "label": "Binary lock threshold", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""}
            ],
        },
        {
            "id": "circle_map_converged_density",
            "name": "Circle map converged density",
            "description": "Uses bandpassed EEG Hilbert phase as per-channel initial conditions for the circle map, iterates final-state distributions across K at fixed Ω, and saves all black/cyan plots from the notebook: aggregate final-state probability, circular mean/resultant/concentration curves, channel summaries, initial phase distribution, and selected final-state distributions.",
            "panel": {"kind": "custom", "field": "circle_map_converged_density", "title": "Circle Map Converged Density"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "Omega", "label": "Circle-map Ω", "type": "number", "default": ""},
                {"name": "n_K", "label": "K grid count", "type": "number", "default": ""},
                {"name": "K_min", "label": "K min", "type": "number", "default": ""},
                {"name": "K_max", "label": "K max", "type": "number", "default": ""},
                {"name": "iterations", "label": "Circle-map iterations", "type": "number", "default": ""},
                {"name": "n_bins", "label": "Histogram bins", "type": "number", "default": ""},
                {"name": "max_samples_per_channel", "label": "Max phase samples per channel", "type": "number", "default": ""},
                {"name": "use_bandpass", "label": "Bandpass before Hilbert phase", "type": "select", "default": "true", "options": ["true", "false"]},
                {"name": "phase_min_hz", "label": "Phase band min Hz", "type": "number", "default": ""},
                {"name": "phase_max_hz", "label": "Phase band max Hz", "type": "number", "default": ""},
                {"name": "filter_order", "label": "Filter order", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""},
                {"name": "random_seed", "label": "Random seed", "type": "number", "default": 42}
            ],
        },
        {
            "id": "mfdfa_plot_viewer",
            "name": "MFDFA plot viewer",
            "description": "Viewer/organizer for Manual Expert MFDFA outputs. It scans per-channel MFDFA PNGs and summary CSVs, then creates the same contact sheets, ranked galleries, individual pages, image inventory, metric rankings, summary heatmap, and width-vs-asymmetry scatter from the notebook viewer.",
            "panel": {"kind": "custom", "field": "mfdfa_plot_viewer", "title": "MFDFA Plot Viewer"},
            "parameters": [
                {"name": "mode", "label": "Viewer speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "run_mfdfa_if_missing", "label": "Run Manual MFDFA first if per-channel images are missing", "type": "select", "default": "true", "options": ["true", "false"]},
                {"name": "sort_mode", "label": "Sort mode", "type": "select", "default": "multifractal_complexity_score", "options": ["multifractal_complexity_score", "alpha_width", "asymmetry", "file_size_kb"]},
                {"name": "max_images_to_show", "label": "Max images to show", "type": "number", "default": ""},
                {"name": "plots_per_page", "label": "Plots per individual page", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels if running MFDFA first", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples if running MFDFA first", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""}
            ],
        },
        {
            "id": "manual_expert_mfdfa",
            "name": "Manual Expert MFDFA",
            "description": "Manual multifractal detrended fluctuation analysis from signal.npy, computing Fq(s), h(q), tau(q), alpha/f(alpha), alpha width, asymmetry, complexity score, topographic map, galleries, regional/hemisphere summaries, and temporal stability plots in the black/cyan notebook style.",
            "panel": {"kind": "custom", "field": "manual_expert_mfdfa", "title": "Manual Expert MFDFA"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "temporal_stability", "label": "Temporal stability", "type": "select", "default": "false", "options": ["true", "false"]},
                {"name": "save_per_channel_plots", "label": "Save per-channel plots", "type": "select", "default": "false", "options": ["true", "false"]},
                {"name": "q_min", "label": "q min override", "type": "number", "default": ""},
                {"name": "q_max", "label": "q max override", "type": "number", "default": ""},
                {"name": "q_count", "label": "q count override", "type": "number", "default": ""},
                {"name": "poly_order", "label": "Polynomial order", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "segment_start_sec", "label": "Segment start sec", "type": "number", "default": ""},
                {"name": "segment_end_sec", "label": "Segment end sec", "type": "number", "default": ""}
            ],
        },
        {
            "id": "wavelet_hurst_exponent",
            "name": "Manual wavelet Hurst exponent",
            "description": "Manual Haar/db4 wavelet Hurst exponent analysis with fractal-dimension proxy, scaling curves, detail-energy maps, topographic Hurst map, regional/hemisphere summaries, and temporal stability plots in the same black/cyan notebook style.",
            "panel": {"kind": "custom", "field": "wavelet_hurst_exponent", "title": "Manual Wavelet Hurst Exponent"},
            "parameters": [
                {"name": "mode", "label": "Speed mode", "type": "select", "default": "ultra", "options": ["ultra", "fast", "balanced", "full"]},
                {"name": "wavelet", "label": "Wavelet", "type": "select", "default": "haar", "options": ["haar", "db4"]},
                {"name": "max_level", "label": "Max DWT level", "type": "number", "default": ""},
                {"name": "temporal_stability", "label": "Temporal stability", "type": "select", "default": "true", "options": ["true", "false"]},
                {"name": "temporal_windows", "label": "Temporal windows", "type": "number", "default": ""},
                {"name": "max_channels", "label": "Max channels", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""}
            ],
        },
        {
            "id": "neuromouse_advanced_plots",
            "name": "NeuroMouse advanced plots",
            "description": "Builds the original NeuroMouse advanced analysis data objects and opens the guaranteed server-rendered plot page: Polar Alpha Chronomap, Kuramoto Animation, Channel Network, and TDA View.",
            "panel": {"kind": "summary", "field": "neuromouse_advanced_plots.summary", "title": "NeuroMouse Advanced Plots"},
            "parameters": [
                {"name": "dataset_id", "label": "Dataset ID", "type": "text", "default": ""},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": 240000},
                {"name": "max_windows", "label": "Max sliding windows", "type": "number", "default": 600},
            ],
        },
        {
            "id": "band_power_summary",
            "name": "Band power summary",
            "description": "Welch PSD band-power table for a chosen frequency band.",
            "panel": {"kind": "table", "field": "band_power_summary.rows", "title": "Band Power Summary"},
            "parameters": [
                {"name": "band", "label": "Preset band", "type": "select", "default": "alpha", "options": ["custom", "delta", "theta", "alpha", "beta", "gamma"]},
                {"name": "min_hz", "label": "Min Hz", "type": "number", "default": 8.0},
                {"name": "max_hz", "label": "Max Hz", "type": "number", "default": 13.0},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": 240000},
            ],
        },
        {
            "id": "spike_detect",
            "name": "Spike detection",
            "description": "Threshold-based spike/event detection from continuous channels.",
            "panel": {"kind": "heatmap_table", "field": "spike_detect.rows", "title": "Spike Detection"},
            "parameters": [
                {"name": "threshold_multiplier", "label": "Threshold multiplier", "type": "number", "default": 5.0},
                {"name": "bandpass_low_hz", "label": "Bandpass low Hz", "type": "number", "default": ""},
                {"name": "bandpass_high_hz", "label": "Bandpass high Hz", "type": "number", "default": ""},
                {"name": "refractory_ms", "label": "Refractory ms", "type": "number", "default": 2.0},
                {"name": "polarity", "label": "Polarity", "type": "select", "default": "negative", "options": ["negative", "positive", "both"]},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": 240000},
                {"name": "max_spikes_per_channel", "label": "Max stored spikes/channel", "type": "number", "default": 5000},
            ],
        },
        {
            "id": "network_burst",
            "name": "Network burst detection",
            "description": "Detects multi-channel burst epochs after threshold-based spike detection.",
            "panel": {"kind": "timeline", "field": "network_burst.timeline", "title": "Network Bursts"},
            "parameters": [
                {"name": "threshold_multiplier", "label": "Spike threshold multiplier", "type": "number", "default": 5.0},
                {"name": "refractory_ms", "label": "Spike refractory ms", "type": "number", "default": 2.0},
                {"name": "polarity", "label": "Spike polarity", "type": "select", "default": "negative", "options": ["negative", "positive", "both"]},
                {"name": "bin_size_ms", "label": "Burst bin size ms", "type": "number", "default": 10.0},
                {"name": "threshold_count", "label": "Minimum spikes per active bin", "type": "number", "default": 3},
                {"name": "merge_gap_ms", "label": "Merge gap ms", "type": "number", "default": 50.0},
                {"name": "min_spikes", "label": "Minimum spikes per burst", "type": "number", "default": 5},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": 240000},
            ],
        },
        {
            "id": "electrode_connectivity",
            "name": "Electrode connectivity",
            "description": "Pairwise lagged cross-correlation matrix over detected spike trains.",
            "panel": {"kind": "matrix", "field": "electrode_connectivity.matrix", "title": "Electrode Connectivity Matrix"},
            "parameters": [
                {"name": "threshold_multiplier", "label": "Spike threshold multiplier", "type": "number", "default": 5.0},
                {"name": "refractory_ms", "label": "Spike refractory ms", "type": "number", "default": 2.0},
                {"name": "polarity", "label": "Spike polarity", "type": "select", "default": "negative", "options": ["negative", "positive", "both"]},
                {"name": "bin_size_ms", "label": "Connectivity bin size ms", "type": "number", "default": 5.0},
                {"name": "max_lag_ms", "label": "Max lag ms", "type": "number", "default": 20.0},
                {"name": "sampling_rate", "label": "Sampling rate override Hz", "type": "number", "default": ""},
                {"name": "max_analysis_samples", "label": "Max analysis samples", "type": "number", "default": 240000},
                {"name": "max_links", "label": "Max stored links", "type": "number", "default": 200},
            ],
        },
    ]


# ---- v0.11.23 MFDFA Shuffle Surrogate analysis ----
# Adapted from the user's shuffle-surrogate MFDFA notebook. Runs all available
# channels by default and compares the original multifractal spectrum width to
# shuffled surrogates in the same black/cyan Matplotlib style.
MFDFA_SHUFFLE_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {
        "scale_min_power": 0.8, "scale_max_power": 3.45, "n_scales": 14,
        "q_min": -4.0, "q_max": 4.0, "q_count": 17,
        "poly_order": 1, "n_shuffles": 5, "max_analysis_samples": 90_000,
    },
    "fast": {
        "scale_min_power": 0.7, "scale_max_power": 3.75, "n_scales": 20,
        "q_min": -5.0, "q_max": 5.0, "q_count": 25,
        "poly_order": 1, "n_shuffles": 10, "max_analysis_samples": 120_000,
    },
    "balanced": {
        # Matches the notebook code most closely: logspace(0.7, 4, 30), q=-5..5 with 41 q-values, 20 shuffles.
        "scale_min_power": 0.7, "scale_max_power": 4.0, "n_scales": 30,
        "q_min": -5.0, "q_max": 5.0, "q_count": 41,
        "poly_order": 1, "n_shuffles": 20, "max_analysis_samples": 180_000,
    },
    "full": {
        "scale_min_power": 0.7, "scale_max_power": 4.0, "n_scales": 36,
        "q_min": -6.0, "q_max": 6.0, "q_count": 49,
        "poly_order": 2, "n_shuffles": 50, "max_analysis_samples": 240_000,
    },
}


def _manual_mfdfa_shuffle_surrogate(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "ultra").lower()
    if mode not in MFDFA_SHUFFLE_CONFIGS:
        mode = "ultra"
    cfg = dict(MFDFA_SHUFFLE_CONFIGS[mode])
    for key in ("scale_min_power", "scale_max_power"):
        if params.get(key) not in (None, ""):
            cfg[key] = float(params[key])
    for key in ("n_scales", "poly_order", "n_shuffles", "max_analysis_samples"):
        if params.get(key) not in (None, ""):
            cfg[key] = _safe_int(params[key], int(cfg[key] or 1))
    q_min = float(params.get("q_min") if params.get("q_min") not in (None, "") else cfg["q_min"])
    q_max = float(params.get("q_max") if params.get("q_max") not in (None, "") else cfg["q_max"])
    q_count = _safe_int(params.get("q_count"), int(cfg["q_count"]))
    q_vals = np.linspace(q_min, q_max, q_count)
    n_shuffles = int(max(1, cfg["n_shuffles"]))
    seed = _safe_int(params.get("random_seed"), 0)
    max_channels = _safe_int(params.get("max_channels"), 999999) if params.get("max_channels") not in (None, "") else None

    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=cfg.get("max_analysis_samples"))
    root = Path(rec["recording_dir"])
    x = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels is not None:
        channels = channels[:max_channels]
        x = x[:, :len(channels)]
    segment_start = _safe_float(params.get("segment_start_sec"))
    segment_end = _safe_float(params.get("segment_end_sec"))
    if segment_start is not None or segment_end is not None:
        s0 = int(max(0, (segment_start or 0.0) * fs))
        s1 = int(min(x.shape[0], (segment_end if segment_end is not None else (x.shape[0] / fs)) * fs))
        if s1 > s0 + 32:
            x = x[s0:s1, :]

    poly_order = int(cfg["poly_order"])
    scales = _mfdfa_make_scales(x.shape[0], cfg, poly_order)
    plt, _ = _mfdfa_import_plotting()
    _mfdfa_apply_style(plt)
    out_dir = root / "advanced_methods" / "manual_mfdfa_shuffle_surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    per_channel_dir = plots_dir / "per_channel"
    per_channel_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    plot_paths: list[dict[str, str]] = []
    per_channel_npz: list[str] = []

    for ch_idx, channel in enumerate(channels):
        signal = np.asarray(x[:, ch_idx], dtype=float)
        original = _mfdfa_channel(signal, scales, q_vals, poly_order)
        orig_summary = original["summary"]
        orig_width = _safe_float(orig_summary.get("alpha_width"))

        shuffle_widths: list[float] = []
        for shuffle_idx in range(n_shuffles):
            shuffled_signal = rng.permutation(signal)
            try:
                shuffled = _mfdfa_channel(shuffled_signal, scales, q_vals, poly_order)
                width = _safe_float(shuffled["summary"].get("alpha_width"))
            except Exception:
                width = None
            if width is None:
                shuffle_widths.append(float("nan"))
            else:
                shuffle_widths.append(float(width))
            shuffle_rows.append({"channel": channel, "channel_index": ch_idx, "shuffle_index": shuffle_idx + 1, "alpha_width_shuffle": width})

        shuffle_arr = np.asarray(shuffle_widths, dtype=float)
        finite = shuffle_arr[np.isfinite(shuffle_arr)]
        shuffle_mean = float(np.nanmean(shuffle_arr)) if finite.size else None
        shuffle_std = float(np.nanstd(shuffle_arr, ddof=1)) if finite.size > 1 else None
        p_upper = None
        delta_width = None
        if orig_width is not None and finite.size:
            p_upper = float((np.sum(finite >= orig_width) + 1.0) / (finite.size + 1.0))
            delta_width = float(orig_width - float(np.nanmean(finite)))

        summary_rows.append({
            "channel": channel,
            "channel_index": ch_idx,
            "alpha_width_original": orig_width,
            "alpha_width_shuffle_mean": shuffle_mean,
            "alpha_width_shuffle_std": shuffle_std,
            "delta_width": delta_width,
            "p_upper_shuffle_ge_original": p_upper,
            "alpha_min_original": orig_summary.get("alpha_min"),
            "alpha_max_original": orig_summary.get("alpha_max"),
            "alpha_at_peak_original": orig_summary.get("alpha_at_peak"),
            "f_peak_original": orig_summary.get("f_peak"),
            "asymmetry_original": orig_summary.get("asymmetry"),
            "n_shuffles": n_shuffles,
            "mode": mode,
            "status": "ok",
        })

        npz_path = out_dir / f"{channel}_shuffle_mfdfa_comparison.npz"
        np.savez_compressed(
            npz_path,
            channel=channel,
            sampling_rate=fs,
            q_vals=q_vals,
            scales=original["scales"],
            hq_original=original["hq"],
            tau_original=original["tau_q"],
            alpha_original=original["alpha_q"],
            f_alpha_original=original["f_alpha_q"],
            alpha_width_original=orig_width if orig_width is not None else np.nan,
            shuffle_widths=shuffle_arr,
            alpha_width_shuffle_mean=shuffle_mean if shuffle_mean is not None else np.nan,
            alpha_width_shuffle_std=shuffle_std if shuffle_std is not None else np.nan,
            p_upper_shuffle_ge_original=p_upper if p_upper is not None else np.nan,
            n_shuffles=n_shuffles,
            seed=seed,
            mode=mode,
        )
        per_channel_npz.append(str(npz_path))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="#000000")
        for ax in axes:
            _mfdfa_style_ax(ax)
        axes[0].plot(q_vals, original["hq"], color="#00FFFF", marker="o", markerfacecolor="#00FFFF", markeredgecolor="#00FFFF")
        _mfdfa_style_ax(axes[0], title=f"{channel}: original h(q)", xlabel="q", ylabel="h(q)")
        mask = np.isfinite(original["alpha_q"]) & np.isfinite(original["f_alpha_q"])
        axes[1].plot(original["alpha_q"][mask], original["f_alpha_q"][mask], color="#00FFFF", marker="o", markerfacecolor="#00FFFF", markeredgecolor="#00FFFF")
        _mfdfa_style_ax(axes[1], title=f"{channel}: original f(alpha)", xlabel="alpha", ylabel="f(alpha)")
        if finite.size:
            axes[2].hist(finite, bins=min(12, max(3, finite.size)), color="#00FFFF", edgecolor="#00FFFF", alpha=0.80)
        if orig_width is not None:
            axes[2].axvline(orig_width, color="#E8FFFF", linestyle="--", linewidth=2.0, label="original")
        _mfdfa_style_ax(axes[2], title=f"{channel}: shuffle alpha-widths", xlabel="alpha width", ylabel="count")
        if orig_width is not None:
            leg = axes[2].legend(loc="best", frameon=True)
            for text in leg.get_texts():
                text.set_color("#00FFFF")
        fig_path = per_channel_dir / f"{channel}_shuffle_mfdfa_comparison.png"
        saved = _mfdfa_save_fig(fig, fig_path, dpi=170 if mode == "ultra" else 190)
        plot_paths.append({"title": f"{channel}: original vs shuffled MFDFA width", "path": saved})

    summary_df = pd.DataFrame(summary_rows)
    shuffle_df = pd.DataFrame(shuffle_rows)
    summary_csv = out_dir / "shuffle_mfdfa_summary.csv"
    shuffle_csv = out_dir / "shuffle_mfdfa_widths_long.csv"
    summary_df.to_csv(summary_csv, index=False)
    shuffle_df.to_csv(shuffle_csv, index=False)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(summary_df)), 5.0), facecolor="#000000")
        _mfdfa_style_ax(ax, title="Original vs shuffled multifractal width", xlabel="Channel", ylabel="Alpha width")
        xloc = np.arange(len(summary_df))
        width = 0.38
        ax.bar(xloc - width / 2, pd.to_numeric(summary_df["alpha_width_original"], errors="coerce"), width=width, color="#00FFFF", edgecolor="#00FFFF", label="Original")
        ax.bar(xloc + width / 2, pd.to_numeric(summary_df["alpha_width_shuffle_mean"], errors="coerce"), width=width, color="none", edgecolor="#00FFFF", hatch="//", label="Shuffle mean")
        ax.set_xticks(xloc)
        ax.set_xticklabels(summary_df["channel"].astype(str).tolist(), rotation=90, color="#00FFFF")
        leg = ax.legend(loc="best", frameon=True)
        for text in leg.get_texts():
            text.set_color("#00FFFF")
        saved = _mfdfa_save_fig(fig, plots_dir / "shuffle_mfdfa_summary_plot.png", dpi=190)
        plot_paths.insert(0, {"title": "Original vs shuffled multifractal width", "path": saved})

        delta_df = summary_df.copy()
        delta_df["delta_width"] = pd.to_numeric(delta_df["delta_width"], errors="coerce")
        delta_df = delta_df.sort_values("delta_width", ascending=True, na_position="first")
        fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(delta_df))), facecolor="#000000")
        _mfdfa_style_ax(ax, title="MFDFA width excess over shuffled surrogates", xlabel="Original Δα - shuffle mean Δα", ylabel="Channel")
        y = np.arange(len(delta_df))
        ax.barh(y, delta_df["delta_width"], color="#00FFFF", edgecolor="#66FFFF", alpha=0.86)
        ax.axvline(0.0, color="#E8FFFF", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(delta_df["channel"].astype(str).tolist(), color="#00FFFF", fontsize=8)
        saved = _mfdfa_save_fig(fig, plots_dir / "shuffle_mfdfa_delta_width_barh.png", dpi=190)
        plot_paths.insert(1, {"title": "Width excess over shuffled surrogates", "path": saved})

    summary_txt = out_dir / "shuffle_mfdfa_summary.txt"
    lines = [
        "MFDFA Shuffle Surrogate Summary",
        "===============================",
        "",
        f"Mode: {mode}",
        f"Channels analyzed: {len(channels)}",
        f"Shuffles per channel: {n_shuffles}",
        f"q range: {q_min:g} to {q_max:g} ({q_count} values)",
        f"Scales used: {scales.tolist()}",
        f"Poly order: {poly_order}",
        f"Random seed: {seed}",
        f"Analysis samples: {x.shape[0]}",
        "",
        "Top channels by width excess:",
    ]
    if not summary_df.empty and "delta_width" in summary_df.columns:
        top = summary_df.copy()
        top["delta_width"] = pd.to_numeric(top["delta_width"], errors="coerce")
        for _, row in top.sort_values("delta_width", ascending=False, na_position="last").head(20).iterrows():
            lines.append(f"  {row['channel']}: original={row['alpha_width_original']}, shuffle_mean={row['alpha_width_shuffle_mean']}, delta={row['delta_width']}, p_upper={row['p_upper_shuffle_ge_original']}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "mode": mode,
        "n_channels": len(channels),
        "n_shuffles": n_shuffles,
        "scale_count": int(len(scales)),
        "q_count": int(len(q_vals)),
        "poly_order": poly_order,
        "mean_original_alpha_width": _safe_float(np.nanmean(pd.to_numeric(summary_df.get("alpha_width_original", pd.Series(dtype=float)), errors="coerce"))) if not summary_df.empty else None,
        "mean_shuffle_alpha_width": _safe_float(np.nanmean(pd.to_numeric(summary_df.get("alpha_width_shuffle_mean", pd.Series(dtype=float)), errors="coerce"))) if not summary_df.empty else None,
        "mean_delta_width": _safe_float(np.nanmean(pd.to_numeric(summary_df.get("delta_width", pd.Series(dtype=float)), errors="coerce"))) if not summary_df.empty else None,
        "top_delta_channel": str(summary_df.sort_values("delta_width", ascending=False, na_position="last").iloc[0]["channel"]) if (not summary_df.empty and "delta_width" in summary_df.columns) else None,
        "plot_count": len(plot_paths),
        "sampling_rate_analysis_hz": fs,
    }
    return {"manual_mfdfa_shuffle_surrogate": {"summary": summary, "rows": _json_safe(summary_df.to_dict("records")), "shuffle_rows": _json_safe(shuffle_df.head(5000).to_dict("records")), "plot_paths": plot_paths, "per_channel_npz": per_channel_npz, "outputs": {"summary_csv": str(summary_csv), "shuffle_csv": str(shuffle_csv), "summary_txt": str(summary_txt), "output_dir": str(out_dir), "plots_dir": str(plots_dir)}}}


# ---- v0.11.24 MFDFA IAAFT Surrogate analysis ----
# Adapted from the user's IAAFT-surrogate MFDFA notebook. Runs all available
# channels by default and compares the original multifractal spectrum width to
# IAAFT surrogate widths in the same black/cyan Matplotlib style.
MFDFA_IAAFT_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {
        "scale_min_power": 0.8, "scale_max_power": 3.45, "n_scales": 14,
        "q_min": -4.0, "q_max": 4.0, "q_count": 17,
        "poly_order": 1, "n_surrogates": 3, "iaaft_iters": 30, "max_analysis_samples": 60_000,
    },
    "fast": {
        "scale_min_power": 0.7, "scale_max_power": 3.75, "n_scales": 20,
        "q_min": -5.0, "q_max": 5.0, "q_count": 25,
        "poly_order": 1, "n_surrogates": 6, "iaaft_iters": 75, "max_analysis_samples": 90_000,
    },
    "balanced": {
        # Matches the notebook code most closely: logspace(0.7, 4, 30), q=-5..5 with 41 q-values, 20 IAAFT surrogates and 200 iterations.
        "scale_min_power": 0.7, "scale_max_power": 4.0, "n_scales": 30,
        "q_min": -5.0, "q_max": 5.0, "q_count": 41,
        "poly_order": 1, "n_surrogates": 20, "iaaft_iters": 200, "max_analysis_samples": 160_000,
    },
    "full": {
        "scale_min_power": 0.7, "scale_max_power": 4.0, "n_scales": 36,
        "q_min": -6.0, "q_max": 6.0, "q_count": 49,
        "poly_order": 2, "n_surrogates": 40, "iaaft_iters": 300, "max_analysis_samples": 220_000,
    },
}


def _iaaft_surrogate_signal(signal: np.ndarray, *, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """Generate an IAAFT surrogate preserving the amplitude distribution and approximate power spectrum."""
    x = np.asarray(signal, dtype=float).reshape(-1)
    if x.size < 4:
        return x.copy()
    med = float(np.nanmedian(x)) if np.any(np.isfinite(x)) else 0.0
    x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)
    x_sorted = np.sort(x)
    target_mag = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    n_iter = int(max(1, n_iter))
    for _ in range(n_iter):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(target_mag * np.exp(1j * np.angle(Y)), n=x.size)
        ranks = np.argsort(y)
        y2 = np.empty_like(y)
        y2[ranks] = x_sorted
        y = y2
    return y


def _manual_mfdfa_iaaft_surrogate(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "ultra").lower()
    if mode not in MFDFA_IAAFT_CONFIGS:
        mode = "ultra"
    cfg = dict(MFDFA_IAAFT_CONFIGS[mode])
    for key in ("scale_min_power", "scale_max_power"):
        if params.get(key) not in (None, ""):
            cfg[key] = float(params[key])
    for key in ("n_scales", "poly_order", "n_surrogates", "iaaft_iters", "max_analysis_samples"):
        if params.get(key) not in (None, ""):
            cfg[key] = _safe_int(params[key], int(cfg[key] or 1))
    q_min = float(params.get("q_min") if params.get("q_min") not in (None, "") else cfg["q_min"])
    q_max = float(params.get("q_max") if params.get("q_max") not in (None, "") else cfg["q_max"])
    q_count = _safe_int(params.get("q_count"), int(cfg["q_count"]))
    q_vals = np.linspace(q_min, q_max, q_count)
    n_surrogates = int(max(1, cfg["n_surrogates"]))
    iaaft_iters = int(max(1, cfg["iaaft_iters"]))
    seed = _safe_int(params.get("random_seed"), 0)
    max_channels = _safe_int(params.get("max_channels"), 999999) if params.get("max_channels") not in (None, "") else None

    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=cfg.get("max_analysis_samples"))
    root = Path(rec["recording_dir"])
    x = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels is not None:
        channels = channels[:max_channels]
        x = x[:, :len(channels)]
    segment_start = _safe_float(params.get("segment_start_sec"))
    segment_end = _safe_float(params.get("segment_end_sec"))
    if segment_start is not None or segment_end is not None:
        s0 = int(max(0, (segment_start or 0.0) * fs))
        s1 = int(min(x.shape[0], (segment_end if segment_end is not None else (x.shape[0] / fs)) * fs))
        if s1 > s0 + 32:
            x = x[s0:s1, :]

    poly_order = int(cfg["poly_order"])
    scales = _mfdfa_make_scales(x.shape[0], cfg, poly_order)
    plt, _ = _mfdfa_import_plotting()
    _mfdfa_apply_style(plt)
    out_dir = root / "advanced_methods" / "manual_mfdfa_iaaft_surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    per_channel_dir = plots_dir / "per_channel"
    per_channel_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, Any]] = []
    surrogate_rows: list[dict[str, Any]] = []
    plot_paths: list[dict[str, str]] = []
    per_channel_npz: list[str] = []

    for ch_idx, channel in enumerate(channels):
        signal = np.asarray(x[:, ch_idx], dtype=float)
        original = _mfdfa_channel(signal, scales, q_vals, poly_order)
        orig_summary = original["summary"]
        orig_width = _safe_float(orig_summary.get("alpha_width"))

        surrogate_widths: list[float] = []
        for surrogate_idx in range(n_surrogates):
            try:
                surrogate_signal = _iaaft_surrogate_signal(signal, n_iter=iaaft_iters, rng=rng)
                surrogate = _mfdfa_channel(surrogate_signal, scales, q_vals, poly_order)
                width = _safe_float(surrogate["summary"].get("alpha_width"))
            except Exception:
                width = None
            surrogate_widths.append(float(width) if width is not None else float("nan"))
            surrogate_rows.append({"channel": channel, "channel_index": ch_idx, "surrogate_index": surrogate_idx + 1, "alpha_width_iaaft": width})

        surrogate_arr = np.asarray(surrogate_widths, dtype=float)
        finite = surrogate_arr[np.isfinite(surrogate_arr)]
        surr_mean = float(np.nanmean(surrogate_arr)) if finite.size else None
        surr_std = float(np.nanstd(surrogate_arr, ddof=1)) if finite.size > 1 else None
        p_upper = None
        delta_width = None
        if orig_width is not None and finite.size:
            p_upper = float((np.sum(finite >= orig_width) + 1.0) / (finite.size + 1.0))
            delta_width = float(orig_width - float(np.nanmean(finite)))

        summary_rows.append({
            "channel": channel,
            "channel_index": ch_idx,
            "alpha_width_original": orig_width,
            "alpha_width_iaaft_mean": surr_mean,
            "alpha_width_iaaft_std": surr_std,
            "delta_width": delta_width,
            "p_upper_iaaft_ge_original": p_upper,
            "alpha_min_original": orig_summary.get("alpha_min"),
            "alpha_max_original": orig_summary.get("alpha_max"),
            "alpha_at_peak_original": orig_summary.get("alpha_at_peak"),
            "f_peak_original": orig_summary.get("f_peak"),
            "asymmetry_original": orig_summary.get("asymmetry"),
            "n_surrogates": n_surrogates,
            "iaaft_iters": iaaft_iters,
            "mode": mode,
            "status": "ok",
        })

        npz_path = out_dir / f"{channel}_iaaft_mfdfa_comparison.npz"
        np.savez_compressed(
            npz_path,
            channel=channel,
            sampling_rate=fs,
            q_vals=q_vals,
            scales=original["scales"],
            hq_original=original["hq"],
            tau_original=original["tau_q"],
            alpha_original=original["alpha_q"],
            f_alpha_original=original["f_alpha_q"],
            alpha_width_original=orig_width if orig_width is not None else np.nan,
            iaaft_widths=surrogate_arr,
            alpha_width_iaaft_mean=surr_mean if surr_mean is not None else np.nan,
            alpha_width_iaaft_std=surr_std if surr_std is not None else np.nan,
            p_upper_iaaft_ge_original=p_upper if p_upper is not None else np.nan,
            n_surrogates=n_surrogates,
            iaaft_iters=iaaft_iters,
            seed=seed,
            mode=mode,
        )
        per_channel_npz.append(str(npz_path))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="#000000")
        for ax in axes:
            _mfdfa_style_ax(ax)
        axes[0].plot(q_vals, original["hq"], color="#00FFFF", marker="o", markerfacecolor="#00FFFF", markeredgecolor="#00FFFF")
        _mfdfa_style_ax(axes[0], title=f"{channel}: original h(q)", xlabel="q", ylabel="h(q)")
        mask = np.isfinite(original["alpha_q"]) & np.isfinite(original["f_alpha_q"])
        axes[1].plot(original["alpha_q"][mask], original["f_alpha_q"][mask], color="#00FFFF", marker="o", markerfacecolor="#00FFFF", markeredgecolor="#00FFFF")
        _mfdfa_style_ax(axes[1], title=f"{channel}: original f(alpha)", xlabel="alpha", ylabel="f(alpha)")
        if finite.size:
            axes[2].hist(finite, bins=min(12, max(3, finite.size)), color="#00FFFF", edgecolor="#00FFFF", alpha=0.80)
        if orig_width is not None:
            axes[2].axvline(orig_width, color="#E8FFFF", linestyle="--", linewidth=2.0, label="original")
        _mfdfa_style_ax(axes[2], title=f"{channel}: IAAFT alpha-widths", xlabel="alpha width", ylabel="count")
        if orig_width is not None:
            leg = axes[2].legend(loc="best", frameon=True)
            for text in leg.get_texts():
                text.set_color("#00FFFF")
        fig_path = per_channel_dir / f"{channel}_iaaft_mfdfa_comparison.png"
        saved = _mfdfa_save_fig(fig, fig_path, dpi=170 if mode == "ultra" else 190)
        plot_paths.append({"title": f"{channel}: original vs IAAFT MFDFA width", "path": saved})

    summary_df = pd.DataFrame(summary_rows)
    surrogate_df = pd.DataFrame(surrogate_rows)
    summary_csv = out_dir / "iaaft_mfdfa_summary.csv"
    surrogate_csv = out_dir / "iaaft_mfdfa_widths_long.csv"
    summary_df.to_csv(summary_csv, index=False)
    surrogate_df.to_csv(surrogate_csv, index=False)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(summary_df)), 5.0), facecolor="#000000")
        _mfdfa_style_ax(ax, title="Original vs IAAFT multifractal width", xlabel="Channel", ylabel="Alpha width")
        xloc = np.arange(len(summary_df))
        width = 0.38
        ax.bar(xloc - width / 2, pd.to_numeric(summary_df["alpha_width_original"], errors="coerce"), width=width, color="#00FFFF", edgecolor="#00FFFF", label="Original")
        ax.bar(xloc + width / 2, pd.to_numeric(summary_df["alpha_width_iaaft_mean"], errors="coerce"), width=width, color="none", edgecolor="#00FFFF", hatch="//", label="IAAFT mean")
        ax.set_xticks(xloc)
        ax.set_xticklabels(summary_df["channel"].astype(str).tolist(), rotation=90, color="#00FFFF")
        leg = ax.legend(loc="best", frameon=True)
        for text in leg.get_texts():
            text.set_color("#00FFFF")
        saved = _mfdfa_save_fig(fig, plots_dir / "iaaft_mfdfa_summary_plot.png", dpi=190)
        plot_paths.insert(0, {"title": "Original vs IAAFT multifractal width", "path": saved})

        delta_df = summary_df.copy()
        delta_df["delta_width"] = pd.to_numeric(delta_df["delta_width"], errors="coerce")
        delta_df = delta_df.sort_values("delta_width", ascending=True, na_position="first")
        fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(delta_df))), facecolor="#000000")
        _mfdfa_style_ax(ax, title="MFDFA width excess over IAAFT surrogates", xlabel="Original Δα - IAAFT mean Δα", ylabel="Channel")
        y = np.arange(len(delta_df))
        ax.barh(y, delta_df["delta_width"], color="#00FFFF", edgecolor="#66FFFF", alpha=0.86)
        ax.axvline(0.0, color="#E8FFFF", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(delta_df["channel"].astype(str).tolist(), color="#00FFFF", fontsize=8)
        saved = _mfdfa_save_fig(fig, plots_dir / "iaaft_mfdfa_delta_width_barh.png", dpi=190)
        plot_paths.insert(1, {"title": "Width excess over IAAFT surrogates", "path": saved})

    summary_txt = out_dir / "iaaft_mfdfa_summary.txt"
    lines = [
        "MFDFA IAAFT Surrogate Summary",
        "==============================",
        "",
        f"Mode: {mode}",
        f"Channels analyzed: {len(channels)}",
        f"IAAFT surrogates per channel: {n_surrogates}",
        f"IAAFT iterations: {iaaft_iters}",
        f"q range: {q_min:g} to {q_max:g} ({q_count} values)",
        f"Scales used: {scales.tolist()}",
        f"Poly order: {poly_order}",
        f"Random seed: {seed}",
        f"Analysis samples: {x.shape[0]}",
        "",
        "Top channels by width excess:",
    ]
    if not summary_df.empty and "delta_width" in summary_df.columns:
        top = summary_df.copy()
        top["delta_width"] = pd.to_numeric(top["delta_width"], errors="coerce")
        for _, row in top.sort_values("delta_width", ascending=False, na_position="last").head(20).iterrows():
            lines.append(f"  {row['channel']}: original={row['alpha_width_original']}, iaaft_mean={row['alpha_width_iaaft_mean']}, delta={row['delta_width']}, p_upper={row['p_upper_iaaft_ge_original']}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "mode": mode,
        "n_channels": len(channels),
        "n_surrogates": n_surrogates,
        "iaaft_iters": iaaft_iters,
        "scale_count": int(len(scales)),
        "q_count": int(len(q_vals)),
        "poly_order": poly_order,
        "mean_original_alpha_width": _safe_float(np.nanmean(pd.to_numeric(summary_df.get("alpha_width_original", pd.Series(dtype=float)), errors="coerce"))) if not summary_df.empty else None,
        "mean_iaaft_alpha_width": _safe_float(np.nanmean(pd.to_numeric(summary_df.get("alpha_width_iaaft_mean", pd.Series(dtype=float)), errors="coerce"))) if not summary_df.empty else None,
        "mean_delta_width": _safe_float(np.nanmean(pd.to_numeric(summary_df.get("delta_width", pd.Series(dtype=float)), errors="coerce"))) if not summary_df.empty else None,
        "top_delta_channel": str(summary_df.sort_values("delta_width", ascending=False, na_position="last").iloc[0]["channel"]) if (not summary_df.empty and "delta_width" in summary_df.columns) else None,
        "plot_count": len(plot_paths),
        "sampling_rate_analysis_hz": fs,
    }
    return {"manual_mfdfa_iaaft_surrogate": {"summary": summary, "rows": _json_safe(summary_df.to_dict("records")), "surrogate_rows": _json_safe(surrogate_df.head(5000).to_dict("records")), "plot_paths": plot_paths, "per_channel_npz": per_channel_npz, "outputs": {"summary_csv": str(summary_csv), "surrogate_csv": str(surrogate_csv), "summary_txt": str(summary_txt), "output_dir": str(out_dir), "plots_dir": str(plots_dir)}}}


# ---- v0.11.26 Wavelet Leader Multifractal analysis ----
# Adapted from the user's manual Haar wavelet-leader notebook. It analyzes all
# available channels by default, computes leader structure functions, zeta(q),
# alpha(q), f(alpha), and saves the same black/cyan per-channel and summary plots.

WAVELET_LEADER_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {"q_min": -4.0, "q_max": 4.0, "q_count": 17, "fit_j_min": 2, "fit_j_max": 9, "min_leaders_per_scale": 16, "max_analysis_samples": 32768, "dpi": 150},
    "fast": {"q_min": -4.0, "q_max": 4.0, "q_count": 25, "fit_j_min": 2, "fit_j_max": 11, "min_leaders_per_scale": 16, "max_analysis_samples": 65536, "dpi": 170},
    "balanced": {"q_min": -4.0, "q_max": 4.0, "q_count": 33, "fit_j_min": 2, "fit_j_max": None, "min_leaders_per_scale": 16, "max_analysis_samples": 120000, "dpi": 190},
    "full": {"q_min": -6.0, "q_max": 6.0, "q_count": 49, "fit_j_min": 2, "fit_j_max": None, "min_leaders_per_scale": 8, "max_analysis_samples": 240000, "dpi": 220},
}


def _wl_trim_to_power_of_two(x: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 16:
        raise ValueError("Signal is too short for wavelet leader analysis.")
    j = int(np.floor(np.log2(n)))
    n2 = 2 ** j
    return x[:n2], j


def _wl_haar_details(x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    x_use, _ = _wl_trim_to_power_of_two(x)
    approx = x_use.copy()
    details: list[np.ndarray] = []
    while len(approx) >= 2:
        n = len(approx) // 2
        a = (approx[0:2*n:2] + approx[1:2*n:2]) / np.sqrt(2.0)
        d = (approx[0:2*n:2] - approx[1:2*n:2]) / np.sqrt(2.0)
        details.append(d)
        approx = a
    return x_use, details, approx


def _wl_compute_leaders(details: list[np.ndarray], eps: float = 1e-12) -> list[np.ndarray]:
    leaders: list[np.ndarray] = []
    for j in range(1, len(details) + 1):
        coeffs_j = details[j - 1]
        n_blocks = len(coeffs_j)
        block_len_j = 2 ** j
        L_j = np.full(n_blocks, eps, dtype=float)
        for k in range(n_blocks):
            start = (k - 1) * block_len_j
            end = (k + 2) * block_len_j - 1
            max_val = eps
            for jp in range(1, j + 1):
                coeffs_jp = details[jp - 1]
                block_len_jp = 2 ** jp
                kp_min = int(np.floor(start / block_len_jp))
                kp_max = int(np.floor(end / block_len_jp))
                kp_min = max(0, kp_min)
                kp_max = min(len(coeffs_jp) - 1, kp_max)
                if kp_min <= kp_max:
                    local = float(np.max(np.abs(coeffs_jp[kp_min:kp_max + 1])))
                    if local > max_val:
                        max_val = local
            L_j[k] = max(max_val, eps)
        leaders.append(L_j)
    return leaders


def _wl_structure_functions(leaders: list[np.ndarray], q_vals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    S = np.full((len(leaders), len(q_vals)), np.nan, dtype=float)
    for j, Lj_raw in enumerate(leaders):
        Lj = np.maximum(np.asarray(Lj_raw, dtype=float), eps)
        log_mean = float(np.mean(np.log(Lj)))
        for qi, q in enumerate(q_vals):
            if np.isclose(q, 0.0):
                S[j, qi] = np.exp(log_mean)
            else:
                S[j, qi] = np.mean(Lj ** q)
    return S


def _wl_estimate_zeta(scale_exponents: np.ndarray, S_qj: np.ndarray, q_vals: np.ndarray, *, fit_j_min: int, fit_j_max: int | None, min_leaders_per_scale: int, leaders: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeta = np.full(len(q_vals), np.nan, dtype=float)
    fit_r2 = np.full(len(q_vals), np.nan, dtype=float)
    valid_scale_mask = np.ones_like(scale_exponents, dtype=bool)
    valid_scale_mask &= scale_exponents >= fit_j_min
    if fit_j_max is not None:
        valid_scale_mask &= scale_exponents <= fit_j_max
    counts = np.array([len(L) for L in leaders], dtype=int)
    valid_scale_mask &= counts >= min_leaders_per_scale
    for qi in range(len(q_vals)):
        y = S_qj[:, qi]
        mask = valid_scale_mask & np.isfinite(y) & (y > 0)
        if np.sum(mask) >= 4:
            xfit = scale_exponents[mask].astype(float)
            yfit = np.log2(y[mask])
            coef = np.polyfit(xfit, yfit, 1)
            yhat = coef[0] * xfit + coef[1]
            ss_res = float(np.sum((yfit - yhat) ** 2))
            ss_tot = float(np.sum((yfit - np.mean(yfit)) ** 2))
            zeta[qi] = float(coef[0])
            fit_r2[qi] = 1.0 - ss_res / (ss_tot + 1e-12)
    return zeta, fit_r2, valid_scale_mask


def _wl_legendre_spectrum(q_vals: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.full_like(zeta, np.nan, dtype=float)
    f_alpha = np.full_like(zeta, np.nan, dtype=float)
    valid = np.isfinite(q_vals) & np.isfinite(zeta)
    if np.sum(valid) >= 5:
        qv = q_vals[valid]
        zv = zeta[valid]
        av = np.gradient(zv, qv)
        fv = qv * av - zv + 1.0
        alpha[valid] = av
        f_alpha[valid] = fv
    return alpha, f_alpha


def _wl_summary(alpha: np.ndarray, f_alpha: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(alpha) & np.isfinite(f_alpha)
    if np.sum(mask) < 5:
        return {"alpha_min": None, "alpha_max": None, "alpha_width": None, "alpha_at_peak": None, "f_peak": None, "asymmetry": None}
    a = np.asarray(alpha[mask], dtype=float)
    f = np.asarray(f_alpha[mask], dtype=float)
    alpha_min = float(np.min(a))
    alpha_max = float(np.max(a))
    alpha_width = float(alpha_max - alpha_min)
    peak_idx = int(np.argmax(f))
    alpha_at_peak = float(a[peak_idx])
    f_peak = float(f[peak_idx])
    left_width = float(alpha_at_peak - alpha_min)
    right_width = float(alpha_max - alpha_at_peak)
    return {
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_width": alpha_width,
        "alpha_at_peak": alpha_at_peak,
        "f_peak": f_peak,
        "asymmetry": float(right_width - left_width),
    }


def _wavelet_leader_multifractal(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "ultra").lower()
    if mode not in WAVELET_LEADER_CONFIGS:
        mode = "ultra"
    cfg = dict(WAVELET_LEADER_CONFIGS[mode])
    for key in ("q_min", "q_max"):
        if params.get(key) not in (None, ""):
            cfg[key] = float(params[key])
    for key in ("q_count", "fit_j_min", "min_leaders_per_scale", "max_analysis_samples"):
        if params.get(key) not in (None, ""):
            cfg[key] = _safe_int(params[key], int(cfg[key] or 1))
    if params.get("fit_j_max") not in (None, ""):
        cfg["fit_j_max"] = _safe_int(params.get("fit_j_max"), int(cfg.get("fit_j_max") or 99))
    q_vals = np.linspace(float(cfg["q_min"]), float(cfg["q_max"]), int(cfg["q_count"]))
    max_channels = _safe_int(params.get("max_channels"), 999999) if params.get("max_channels") not in (None, "") else None

    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=cfg.get("max_analysis_samples"))
    root = Path(rec["recording_dir"])
    x = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels is not None:
        channels = channels[:max_channels]
        x = x[:, :len(channels)]
    segment_start = _safe_float(params.get("segment_start_sec"))
    segment_end = _safe_float(params.get("segment_end_sec"))
    if segment_start is not None or segment_end is not None:
        s0 = int(max(0, (segment_start or 0.0) * fs))
        s1 = int(min(x.shape[0], (segment_end if segment_end is not None else (x.shape[0] / fs)) * fs))
        if s1 > s0 + 32:
            x = x[s0:s1, :]

    plt, _ = _mfdfa_import_plotting()
    _mfdfa_apply_style(plt)
    out_dir = root / "advanced_methods" / "wavelet_leader_multifractal"
    plots_dir = out_dir / "plots"
    per_channel_dir = plots_dir / "per_channel"
    for d in (out_dir, plots_dir, per_channel_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    zeta_rows: list[dict[str, Any]] = []
    plot_paths: list[dict[str, str]] = []
    per_channel_npz: list[str] = []
    fit_j_min = int(cfg["fit_j_min"])
    fit_j_max = cfg.get("fit_j_max")
    fit_j_max = int(fit_j_max) if fit_j_max not in (None, "") else None
    min_leaders_per_scale = int(cfg["min_leaders_per_scale"])
    start = __import__('time').perf_counter()

    for ch_idx, channel in enumerate(channels):
        sig = _mfdfa_safe_zscore(x[:, ch_idx].astype(float))
        x_use, details, approx_last = _wl_haar_details(sig)
        leaders = _wl_compute_leaders(details, eps=1e-12)
        S_qj = _wl_structure_functions(leaders, q_vals, eps=1e-12)
        scale_exponents = np.arange(1, len(leaders) + 1)
        zeta, zeta_fit_r2, fit_mask = _wl_estimate_zeta(scale_exponents, S_qj, q_vals, fit_j_min=fit_j_min, fit_j_max=fit_j_max, min_leaders_per_scale=min_leaders_per_scale, leaders=leaders)
        alpha, f_alpha = _wl_legendre_spectrum(q_vals, zeta)
        summ = _wl_summary(alpha, f_alpha)
        row = {
            "channel": channel,
            "trimmed_length": int(len(x_use)),
            "n_levels": int(len(leaders)),
            "fit_levels": ",".join([str(int(v)) for v in scale_exponents[fit_mask]]),
            "mean_zeta_fit_r2": _safe_float(np.nanmean(zeta_fit_r2)) if np.any(np.isfinite(zeta_fit_r2)) else None,
            **summ,
        }
        summary_rows.append(row)
        for qi, q in enumerate(q_vals):
            zeta_rows.append({
                "channel": channel,
                "q": float(q),
                "zeta": _safe_float(zeta[qi]),
                "zeta_fit_r2": _safe_float(zeta_fit_r2[qi]),
                "alpha": _safe_float(alpha[qi]),
                "f_alpha": _safe_float(f_alpha[qi]),
            })
        npz_path = out_dir / f"{channel}_wavelet_leader_manual.npz"
        # Save structure functions as a rectangular array plus metadata; leaders are ragged, so save counts only.
        np.savez_compressed(
            npz_path,
            channel=channel,
            sampling_rate=fs,
            trimmed_length=len(x_use),
            scale_exponents=scale_exponents,
            leader_counts=np.array([len(L) for L in leaders], dtype=int),
            q_vals=q_vals,
            S_qj=S_qj,
            zeta=zeta,
            zeta_fit_r2=zeta_fit_r2,
            alpha=alpha,
            f_alpha=f_alpha,
            fit_mask=fit_mask,
            alpha_min=summ.get("alpha_min") if summ.get("alpha_min") is not None else np.nan,
            alpha_max=summ.get("alpha_max") if summ.get("alpha_max") is not None else np.nan,
            alpha_width=summ.get("alpha_width") if summ.get("alpha_width") is not None else np.nan,
            alpha_at_peak=summ.get("alpha_at_peak") if summ.get("alpha_at_peak") is not None else np.nan,
            f_peak=summ.get("f_peak") if summ.get("f_peak") is not None else np.nan,
            asymmetry=summ.get("asymmetry") if summ.get("asymmetry") is not None else np.nan,
        )
        per_channel_npz.append(str(npz_path))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=MFDFA_BLACK)
        for ax in axes:
            _mfdfa_style_ax(ax)
        axes[0].plot(q_vals, zeta, color=MFDFA_ACCENT, marker="o", markerfacecolor=MFDFA_ACCENT, markeredgecolor=MFDFA_ACCENT)
        _mfdfa_style_ax(axes[0], f"{channel}: zeta(q)", "q", "zeta(q)")
        axes[1].plot(q_vals, alpha, color=MFDFA_ACCENT, marker="o", markerfacecolor=MFDFA_ACCENT, markeredgecolor=MFDFA_ACCENT)
        _mfdfa_style_ax(axes[1], f"{channel}: alpha(q)", "q", "alpha(q)")
        mask = np.isfinite(alpha) & np.isfinite(f_alpha)
        axes[2].plot(alpha[mask], f_alpha[mask], color=MFDFA_ACCENT, marker="o", markerfacecolor=MFDFA_ACCENT, markeredgecolor=MFDFA_ACCENT)
        _mfdfa_style_ax(axes[2], f"{channel}: wavelet leader spectrum", "alpha", "f(alpha)")
        fig_path = per_channel_dir / f"{channel}_wavelet_leader_spectrum.png"
        plot_paths.append({"title": f"{channel}: zeta(q), alpha(q), and wavelet leader spectrum", "path": _mfdfa_save_fig(fig, fig_path, dpi=int(cfg["dpi"]))})

    summary_df = pd.DataFrame(summary_rows)
    zeta_df = pd.DataFrame(zeta_rows)
    summary_csv = out_dir / "wavelet_leader_summary.csv"
    zeta_csv = out_dir / "wavelet_leader_zeta_alpha_falpha.csv"
    summary_df.to_csv(summary_csv, index=False)
    zeta_df.to_csv(zeta_csv, index=False)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(summary_df)), 5.0), facecolor=MFDFA_BLACK)
        ax.bar(summary_df["channel"], pd.to_numeric(summary_df["alpha_width"], errors="coerce"), color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT)
        _mfdfa_style_ax(ax, "Wavelet leader multifractal width by channel", "Channel", "alpha width")
        ax.tick_params(axis="x", rotation=90, colors=MFDFA_ACCENT)
        plot_paths.insert(0, {"title": "Wavelet leader multifractal width by channel", "path": _mfdfa_save_fig(fig, plots_dir / "wavelet_leader_width_by_channel.png", dpi=190)})

        width_df = summary_df.copy()
        width_df["alpha_width"] = pd.to_numeric(width_df["alpha_width"], errors="coerce")
        width_df = width_df.sort_values("alpha_width", ascending=True, na_position="first")
        fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(width_df))), facecolor=MFDFA_BLACK)
        y = np.arange(len(width_df))
        ax.barh(y, width_df["alpha_width"], color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.86)
        ax.set_yticks(y)
        ax.set_yticklabels(width_df["channel"].astype(str).tolist(), color=MFDFA_ACCENT, fontsize=8)
        mean_width = float(np.nanmean(width_df["alpha_width"])) if len(width_df) else np.nan
        if np.isfinite(mean_width):
            ax.axvline(mean_width, color=MFDFA_WHITE, linestyle="--", linewidth=1.0, alpha=0.80, label=f"Mean={mean_width:.3f}")
            leg = ax.legend(loc="best", frameon=True)
            for text in leg.get_texts():
                text.set_color(MFDFA_ACCENT)
        _mfdfa_style_ax(ax, "Wavelet leader alpha-width ranking", "alpha width", "Channel")
        plot_paths.insert(1, {"title": "Wavelet leader alpha-width ranking", "path": _mfdfa_save_fig(fig, plots_dir / "wavelet_leader_alpha_width_ranking.png", dpi=190)})

    alpha_vals = pd.to_numeric(summary_df.get("alpha_width", pd.Series(dtype=float)), errors="coerce") if not summary_df.empty else pd.Series(dtype=float)
    summary_txt = out_dir / "wavelet_leader_summary.txt"
    lines = [
        "Wavelet Leader Multifractal Summary",
        "====================================",
        "",
        f"Mode: {mode}",
        f"Recording: {root}",
        f"Channels analyzed: {len(channels)}",
        f"Analysis samples: {x.shape[0]}",
        f"q range: {float(q_vals.min())} to {float(q_vals.max())} ({len(q_vals)} values)",
        f"Fit j min: {fit_j_min}",
        f"Fit j max: {fit_j_max if fit_j_max is not None else 'auto'}",
        f"Min leaders per fit scale: {min_leaders_per_scale}",
        f"Mean alpha width: {float(np.nanmean(alpha_vals)) if len(alpha_vals) else np.nan}",
        "",
        "Top channels by alpha width:",
    ]
    if not summary_df.empty:
        top = summary_df.copy()
        top["alpha_width"] = pd.to_numeric(top["alpha_width"], errors="coerce")
        for _, row in top.sort_values("alpha_width", ascending=False, na_position="last").head(20).iterrows():
            lines.append(f"  {row['channel']}: alpha_width={row['alpha_width']}, asymmetry={row['asymmetry']}, mean_zeta_fit_r2={row['mean_zeta_fit_r2']}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    elapsed = __import__('time').perf_counter() - start
    summary = {
        "mode": mode,
        "n_channels": int(len(channels)),
        "analysis_samples": int(x.shape[0]),
        "sampling_rate_hz": fs,
        "q_count": int(len(q_vals)),
        "fit_j_min": fit_j_min,
        "fit_j_max": fit_j_max,
        "min_leaders_per_scale": min_leaders_per_scale,
        "mean_alpha_width": _safe_float(np.nanmean(alpha_vals)) if len(alpha_vals) else None,
        "median_alpha_width": _safe_float(np.nanmedian(alpha_vals)) if len(alpha_vals) else None,
        "top_alpha_width_channel": str(summary_df.sort_values("alpha_width", ascending=False, na_position="last").iloc[0]["channel"]) if (not summary_df.empty and "alpha_width" in summary_df.columns) else None,
        "plot_count": len(plot_paths),
        "elapsed_sec": _safe_float(elapsed),
    }
    return {"wavelet_leader_multifractal": {"summary": summary, "rows": _json_safe(summary_df.to_dict("records")), "zeta_rows": _json_safe(zeta_df.head(5000).to_dict("records")), "plot_paths": plot_paths, "per_channel_npz": per_channel_npz, "outputs": {"summary_csv": str(summary_csv), "zeta_csv": str(zeta_csv), "summary_txt": str(summary_txt), "output_dir": str(out_dir), "plots_dir": str(plots_dir)}}}


# v0.11.26 local Lyapunov spectrum analysis for generated/precomputed embeddings.
# Adapted from the user's custom Lyapunov spectrum notebook. It runs all
# available channels by default, uses ultra/fast/balanced/full speed profiles,
# and saves the same black/cyan expert summary plots as backend PNGs.
LYAP_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {
        "embedding_dims": [2, 3], "tau_ms": 10.0, "k_neighbors": 6,
        "theiler": 8, "stride": 25, "max_steps_per_file": 80,
        "max_points": 900, "query_extra": 40, "chunk_size": 512,
        "max_analysis_samples": 20_000, "max_files": None,
    },
    "fast": {
        "embedding_dims": [2, 3, 4], "tau_ms": 10.0, "k_neighbors": 10,
        "theiler": 10, "stride": 15, "max_steps_per_file": 160,
        "max_points": 1500, "query_extra": 60, "chunk_size": 768,
        "max_analysis_samples": 60_000, "max_files": None,
    },
    "balanced": {
        "embedding_dims": [2, 3, 4, 5], "tau_ms": 10.0, "k_neighbors": 12,
        "theiler": 10, "stride": 10, "max_steps_per_file": 320,
        "max_points": 3000, "query_extra": 80, "chunk_size": 1024,
        "max_analysis_samples": 120_000, "max_files": None,
    },
    "full": {
        "embedding_dims": [2, 3, 4, 5, 6, 7, 8, 9, 10], "tau_ms": 10.0, "k_neighbors": 25,
        "theiler": 10, "stride": 5, "max_steps_per_file": 1000,
        "max_points": 5000, "query_extra": 80, "chunk_size": 1024,
        "max_analysis_samples": 240_000, "max_files": None,
    },
}


def _lyap_delay_embed_1d(x: np.ndarray, emb_dim: int, tau: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    n_eff = len(x) - (int(emb_dim) - 1) * int(tau)
    if n_eff <= max(8, int(emb_dim) + 2):
        return np.empty((0, int(emb_dim)), dtype=float)
    return np.column_stack([x[i:i+n_eff] for i in range(0, int(emb_dim) * int(tau), int(tau))])


def _lyap_maybe_subsample(X: np.ndarray, dt: float, max_points: int | None) -> tuple[np.ndarray, float, int]:
    X = np.asarray(X, dtype=float)
    if max_points is None or X.shape[0] <= int(max_points):
        return X, float(dt), 1
    step = int(math.ceil(X.shape[0] / int(max_points)))
    return X[::step], float(dt) * step, step


def _lyap_spectrum_from_trajectory(
    X: np.ndarray,
    *,
    k_neighbors: int = 12,
    theiler: int = 10,
    stride: int = 20,
    dt: float = 1.0,
    max_steps: int | None = 500,
    query_extra: int = 80,
    chunk_size: int = 1024,
    random_seed: int = 0,
) -> tuple[np.ndarray, int]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D: samples x state_dim")
    n, d = X.shape
    if n < max(8, d + 4):
        raise ValueError(f"Trajectory too short for dimension={d}")
    X0 = X[:-1]
    X1 = X[1:]
    n0 = X0.shape[0]
    if n0 < max(d + 2, int(k_neighbors) + 2):
        raise ValueError(f"Trajectory too short for dimension={d} and k_neighbors={k_neighbors}")
    anchor_idx = np.arange(0, max(0, n0 - 1), max(1, int(stride)))
    if anchor_idx.size == 0:
        raise ValueError("No anchor points available")
    if max_steps is not None and int(max_steps) > 0 and anchor_idx.size > int(max_steps):
        rng = np.random.default_rng(int(random_seed))
        anchor_idx = np.sort(rng.choice(anchor_idx, size=int(max_steps), replace=False))

    try:
        from scipy.spatial import cKDTree as KDTree  # type: ignore
    except Exception:
        KDTree = None  # type: ignore

    Q = np.eye(d)
    sums = np.zeros(d, dtype=float)
    used = 0

    if KDTree is not None:
        tree = KDTree(X0)
        qk = min(n0, max(int(k_neighbors) + int(query_extra), int(k_neighbors) * 4 + 2 * int(theiler) + 5, d + int(query_extra)))
        for start in range(0, anchor_idx.size, int(chunk_size)):
            inds = anchor_idx[start:start + int(chunk_size)]
            points = X0[inds]
            try:
                _, all_nbrs = tree.query(points, k=qk, workers=-1)
            except TypeError:
                _, all_nbrs = tree.query(points, k=qk)
            all_nbrs = np.asarray(all_nbrs)
            if all_nbrs.ndim == 1:
                all_nbrs = all_nbrs[:, None]
            for row, i in zip(all_nbrs, inds):
                row = np.asarray(row)
                nbrs = row[(row >= 0) & (row < n0 - 1) & (np.abs(row - i) > int(theiler))]
                if nbrs.size < max(d + 1, int(k_neighbors)):
                    continue
                nbrs = nbrs[:int(k_neighbors)]
                A = X0[nbrs] - X0[i]
                B = X1[nbrs] - X1[i]
                try:
                    JT, *_ = np.linalg.lstsq(A, B, rcond=None)
                    J = JT.T
                except np.linalg.LinAlgError:
                    continue
                Z = J @ Q
                Q, R = np.linalg.qr(Z)
                diag = np.abs(np.diag(R))
                diag[diag == 0] = np.finfo(float).tiny
                sums += np.log(diag)
                used += 1
    else:
        for i in anchor_idx:
            diffs = X0 - X0[i]
            d2 = np.einsum("ij,ij->i", diffs, diffs)
            lo = max(0, int(i) - int(theiler))
            hi = min(n0, int(i) + int(theiler) + 1)
            d2[lo:hi] = np.inf
            d2[n0 - 1:] = np.inf
            kk = min(n0, max(int(k_neighbors) + int(query_extra), int(k_neighbors) * 4 + 2 * int(theiler) + 5, d + int(query_extra)))
            cand = np.argpartition(d2, kk - 1)[:kk]
            cand = cand[np.argsort(d2[cand])]
            nbrs = cand[np.isfinite(d2[cand])][:int(k_neighbors)]
            if nbrs.size < max(d + 1, int(k_neighbors)):
                continue
            A = X0[nbrs] - X0[i]
            B = X1[nbrs] - X1[i]
            try:
                JT, *_ = np.linalg.lstsq(A, B, rcond=None)
                J = JT.T
            except np.linalg.LinAlgError:
                continue
            Z = J @ Q
            Q, R = np.linalg.qr(Z)
            diag = np.abs(np.diag(R))
            diag[diag == 0] = np.finfo(float).tiny
            sums += np.log(diag)
            used += 1

    if used == 0:
        raise ValueError("No usable Lyapunov steps; try lowering k_neighbors/theiler/stride or increasing max_steps")
    return sums / (used * max(float(dt), 1e-12)), int(used)


def _lyap_kaplan_yorke_dimension(exps: np.ndarray) -> float:
    exps = np.asarray(exps, dtype=float)
    exps = exps[np.isfinite(exps)]
    if exps.size == 0:
        return float("nan")
    lam = np.sort(exps)[::-1]
    csum = np.cumsum(lam)
    if csum[0] < 0:
        return 0.0
    nonneg = np.where(csum >= 0)[0]
    if nonneg.size == 0:
        return float("nan")
    j_idx = int(nonneg[-1])
    j = j_idx + 1
    if j >= lam.size:
        return float(lam.size)
    denom = abs(float(lam[j]))
    if denom == 0 or not np.isfinite(denom):
        return float(j)
    return float(j + csum[j_idx] / denom)


def _lyap_expansion_entropy(exps: np.ndarray) -> float:
    exps = np.asarray(exps, dtype=float)
    pos = exps[np.isfinite(exps) & (exps > 0)]
    if pos.size <= 1:
        return 0.0
    p = pos / (np.sum(pos) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(pos.size))


def _lyap_exp_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if str(c).startswith("Exp")], key=lambda c: int(str(c).replace("Exp", "")) if str(c).replace("Exp", "").isdigit() else 10**9)


def _lyap_add_expert_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    exp_cols = _lyap_exp_cols(df)
    out_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        exps = row[exp_cols].to_numpy(dtype=float) if exp_cols else np.array([], dtype=float)
        exps = exps[np.isfinite(exps)]
        if row.get("status") != "ok" or exps.size == 0:
            out_rows.append({"LLE": np.nan, "MLE": np.nan, "most_negative_exp": np.nan, "sum_exponents": np.nan, "mean_exponent": np.nan, "std_exponent": np.nan, "spectral_radius_abs": np.nan, "positive_count": np.nan, "negative_count": np.nan, "near_zero_count": np.nan, "KS_entropy_proxy": np.nan, "Kaplan_Yorke_dim": np.nan, "expansion_entropy": np.nan, "chaos_timescale": np.nan, "hyperchaos_flag": False, "dissipative_flag": False, "reliable_steps_flag": False, "quality_score": np.nan})
            continue
        sorted_exps = np.sort(exps)[::-1]
        lle = float(np.max(exps))
        eps_zero = 1e-3
        positive_count = int(np.sum(exps > eps_zero))
        sum_exps = float(np.sum(exps))
        steps_used = _safe_float(row.get("steps_used")) or float("nan")
        state_dim = _safe_float(row.get("state_dim")) or float("nan")
        subsample_step = _safe_float(row.get("subsample_step")) or 1.0
        reliable = bool(np.isfinite(steps_used) and np.isfinite(state_dim) and steps_used >= max(40, 12 * state_dim))
        if np.isfinite(steps_used) and np.isfinite(state_dim):
            step_component = min(1.0, steps_used / max(40, 12 * state_dim))
            quality = float(100.0 * step_component / math.sqrt(max(subsample_step, 1.0)))
        else:
            quality = float("nan")
        out_rows.append({
            "LLE": lle,
            "MLE": float(np.mean(sorted_exps[:min(3, len(sorted_exps))])),
            "most_negative_exp": float(np.min(exps)),
            "sum_exponents": sum_exps,
            "mean_exponent": float(np.mean(exps)),
            "std_exponent": float(np.std(exps)),
            "spectral_radius_abs": float(np.max(np.abs(exps))),
            "positive_count": positive_count,
            "negative_count": int(np.sum(exps < -eps_zero)),
            "near_zero_count": int(np.sum(np.abs(exps) <= eps_zero)),
            "KS_entropy_proxy": float(np.sum(exps[exps > 0])) if np.any(exps > 0) else 0.0,
            "Kaplan_Yorke_dim": _lyap_kaplan_yorke_dimension(exps),
            "expansion_entropy": _lyap_expansion_entropy(exps),
            "chaos_timescale": float(1.0 / lle) if lle > 0 else np.nan,
            "hyperchaos_flag": bool(positive_count >= 2),
            "dissipative_flag": bool(sum_exps < 0),
            "reliable_steps_flag": reliable,
            "quality_score": quality,
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)


def _lyap_precomputed_embedding_paths(root: Path, channels: list[str], dims: list[int]) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    for ch in channels:
        for emb_dim in dims:
            p = _efd_embedding_file(root, ch, int(emb_dim))
            if p is not None:
                found[(ch, int(emb_dim))] = p
    return found


def _lyap_short_labels(series: pd.Series, max_len: int = 22) -> np.ndarray:
    return np.array([str(s) if len(str(s)) <= max_len else str(s)[:max_len-1] + "…" for s in series.astype(str).to_numpy()])


def _lyap_plot_images(df: pd.DataFrame, plots_dir: Path) -> list[dict[str, str]]:
    plt, _ = _mfdfa_import_plotting()
    from matplotlib.colors import LinearSegmentedColormap
    _mfdfa_apply_style(plt)
    cyan_div = LinearSegmentedColormap.from_list("cyan_diverging", [MFDFA_ACCENT_DIM, MFDFA_BLACK, MFDFA_ACCENT_SOFT], N=256)
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[dict[str, str]] = []
    ok = df[df["status"] == "ok"].copy()
    exp_cols = _lyap_exp_cols(ok)

    def add(title: str, fig: Any, filename: str):
        paths.append({"title": title, "path": _mfdfa_save_fig(fig, plots_dir / filename, dpi=220)})

    if not ok.empty and "LLE" in ok:
        sub = ok[np.isfinite(ok["LLE"])].sort_values("LLE", ascending=False).head(35).sort_values("LLE", ascending=True)
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(11, 8), facecolor=MFDFA_BLACK)
            y = np.arange(len(sub)); ax.barh(y, sub["LLE"], color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.85); ax.axvline(0, color=MFDFA_WHITE, linewidth=1.0, alpha=0.8)
            ax.set_yticks(y); ax.set_yticklabels(_lyap_short_labels(sub["channel"], 24), color=MFDFA_ACCENT, fontsize=8)
            _mfdfa_style_ax(ax, f"Top {len(sub)} Channels by Largest Lyapunov Exponent", "Largest Lyapunov exponent", "Channel")
            add("Top LLE ranking", fig, "01_top_lle_ranking_black_cyan.png")

        sub = ok[np.isfinite(ok["KS_entropy_proxy"]) & np.isfinite(ok["Kaplan_Yorke_dim"])].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(10, 7), facecolor=MFDFA_BLACK)
            sizes = np.clip(sub.get("quality_score", pd.Series(40, index=sub.index)).fillna(40).to_numpy(dtype=float), 18, 180)
            ax.scatter(sub["Kaplan_Yorke_dim"], sub["KS_entropy_proxy"], s=sizes, c=MFDFA_ACCENT, edgecolors=MFDFA_WHITE, linewidths=0.7, alpha=0.78)
            for _, r in sub.sort_values("KS_entropy_proxy", ascending=False).head(8).iterrows():
                ax.annotate(str(r["channel"])[:22], (r["Kaplan_Yorke_dim"], r["KS_entropy_proxy"]), xytext=(5,5), textcoords="offset points", fontsize=8, color=MFDFA_ACCENT_SOFT)
            _mfdfa_style_ax(ax, "Chaos Complexity Map", "Kaplan-Yorke dimension", "KS entropy proxy: sum of positive exponents")
            add("Chaos complexity map", fig, "02_complexity_map_ks_vs_ky_black_cyan.png")

        if exp_cols:
            sub = ok[np.isfinite(ok["LLE"])].sort_values("LLE", ascending=False).head(45).sort_values(["embedding_dimension", "LLE"], ascending=[True, False])
            M = sub[exp_cols].to_numpy(dtype=float) if not sub.empty else np.empty((0,0))
            if M.size and np.any(np.isfinite(M)):
                finite_abs = np.abs(M[np.isfinite(M)]); vmax = max(float(np.percentile(finite_abs, 95)), 1e-9) if finite_abs.size else 1.0
                fig, ax = plt.subplots(figsize=(11, 8), facecolor=MFDFA_BLACK)
                im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cyan_div, vmin=-vmax, vmax=vmax)
                ax.set_xticks(np.arange(len(exp_cols))); ax.set_xticklabels(exp_cols, rotation=45, ha="right", color=MFDFA_ACCENT)
                ax.set_yticks(np.arange(len(sub))); ax.set_yticklabels(_lyap_short_labels(sub["channel"], 24), color=MFDFA_ACCENT, fontsize=7)
                _mfdfa_style_ax(ax, f"Lyapunov Spectrum Heatmap: Top {len(sub)} by LLE", "Exponent index", "Channel")
                cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cbar.ax.yaxis.set_tick_params(color=MFDFA_ACCENT); plt.setp(cbar.ax.get_yticklabels(), color=MFDFA_ACCENT); cbar.outline.set_edgecolor(MFDFA_ACCENT); cbar.set_label("Lyapunov exponent", color=MFDFA_ACCENT)
                add("Lyapunov spectrum heatmap", fig, "03_spectrum_heatmap_black_cyan.png")

            dims = sorted([d for d in ok["embedding_dimension"].dropna().unique()]) if "embedding_dimension" in ok else []
            if dims:
                fig, ax = plt.subplots(figsize=(10, 7), facecolor=MFDFA_BLACK)
                max_dim = max(dims)
                for dim in dims:
                    subd = ok[ok["embedding_dimension"] == dim]
                    M = subd[exp_cols].to_numpy(dtype=float)
                    if M.size:
                        mean = np.nanmean(M, axis=0); x = np.arange(1, len(mean)+1); valid = np.isfinite(mean)
                        if valid.sum() >= 2:
                            ax.plot(x[valid], mean[valid], marker="o", linewidth=1.8, markersize=4, color=MFDFA_ACCENT, alpha=0.35 + 0.55 * (float(dim)/float(max_dim)), label=f"{int(dim)}D")
                ax.axhline(0, color=MFDFA_WHITE, linewidth=1.0, alpha=0.8)
                _mfdfa_style_ax(ax, "Mean Lyapunov Spectrum by Embedding Dimension", "Exponent rank", "Mean exponent")
                leg = ax.legend(loc="best", frameon=True)
                for text in leg.get_texts(): text.set_color(MFDFA_ACCENT)
                add("Mean spectrum by embedding dimension", fig, "04_mean_spectrum_by_embedding_dim_black_cyan.png")

        sub = ok[np.isfinite(ok["sum_exponents"]) & np.isfinite(ok["LLE"])].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(10, 7), facecolor=MFDFA_BLACK)
            ax.scatter(sub["sum_exponents"], sub["LLE"], s=np.clip(sub.get("quality_score", pd.Series(40, index=sub.index)).fillna(40).to_numpy(dtype=float), 20, 180), c=MFDFA_ACCENT, edgecolors=MFDFA_WHITE, linewidths=0.7, alpha=0.82)
            ax.axhline(0, color=MFDFA_WHITE, linewidth=1.0, alpha=0.75); ax.axvline(0, color=MFDFA_WHITE, linewidth=1.0, alpha=0.75)
            for _, r in sub.sort_values("LLE", ascending=False).head(8).iterrows(): ax.annotate(str(r["channel"])[:22], (r["sum_exponents"], r["LLE"]), xytext=(5,5), textcoords="offset points", fontsize=8, color=MFDFA_ACCENT_SOFT)
            _mfdfa_style_ax(ax, "Dissipation vs Instability Phase Plot", "Sum of Lyapunov exponents: volume expansion/contraction", "Largest Lyapunov exponent")
            add("Dissipation vs instability", fig, "05_dissipation_vs_instability_black_cyan.png")

        sub = ok[np.isfinite(ok["elapsed_sec"])].sort_values("elapsed_sec", ascending=False).head(35).sort_values("elapsed_sec", ascending=True)
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(11, 8), facecolor=MFDFA_BLACK); y=np.arange(len(sub)); ax.barh(y, sub["elapsed_sec"], color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.82); ax.set_yticks(y); ax.set_yticklabels(_lyap_short_labels(sub["channel"], 24), color=MFDFA_ACCENT, fontsize=8); ax.axvline(sub["elapsed_sec"].median(), color=MFDFA_WHITE, linewidth=1.0, alpha=0.75, linestyle="--")
            for yi, (_, r) in zip(y, sub.iterrows()):
                if np.isfinite(r.get("steps_used", np.nan)): ax.text(r["elapsed_sec"], yi, f"  {int(r['steps_used'])} steps", va="center", ha="left", color=MFDFA_ACCENT_SOFT, fontsize=7)
            _mfdfa_style_ax(ax, f"Runtime QC: Slowest {len(sub)} Files", "Elapsed seconds per file", "Channel")
            add("Runtime QC", fig, "06_runtime_qc_slowest_files_black_cyan.png")

        sub = ok[np.isfinite(ok["elapsed_sec"]) & np.isfinite(ok["embedding_dimension"])].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(10, 7), facecolor=MFDFA_BLACK); rng=np.random.default_rng(0)
            for dim in sorted(sub["embedding_dimension"].dropna().unique()):
                subd = sub[sub["embedding_dimension"] == dim]; x=np.full(len(subd), float(dim)); jitter=rng.normal(0,0.035,size=len(subd)); ax.scatter(x+jitter, subd["elapsed_sec"], s=45, c=MFDFA_ACCENT, edgecolors=MFDFA_WHITE, linewidths=0.5, alpha=0.68); med=subd["elapsed_sec"].median(); ax.plot([float(dim)-0.18,float(dim)+0.18],[med,med], color=MFDFA_WHITE, linewidth=2.0, alpha=0.9)
            _mfdfa_style_ax(ax, "Runtime Distribution by Embedding Dimension", "Embedding dimension", "Elapsed seconds per file")
            add("Runtime by dimension", fig, "07_runtime_distribution_by_dimension_black_cyan.png")

        sub = ok[np.isfinite(ok["LLE"])].sort_values("LLE", ascending=False).head(20).copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(13, 7), facecolor=MFDFA_BLACK); x=np.arange(len(sub)); w=0.28; labels=_lyap_short_labels(sub["channel"], 18); ax.bar(x-w, sub["LLE"], w, label="LLE", color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.85); ax.bar(x, sub["KS_entropy_proxy"], w, label="KS entropy proxy", color=MFDFA_ACCENT_SOFT, edgecolor=MFDFA_ACCENT, alpha=0.60); ax.bar(x+w, sub["Kaplan_Yorke_dim"], w, label="Kaplan-Yorke dim", color=MFDFA_ACCENT_DIM, edgecolor=MFDFA_ACCENT, alpha=0.75); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=65, ha="right", color=MFDFA_ACCENT, fontsize=8); _mfdfa_style_ax(ax, "Top Channels: Instability, Entropy Proxy, Dimension", "Channel", "Metric value"); leg=ax.legend(loc="best", frameon=True); [text.set_color(MFDFA_ACCENT) for text in leg.get_texts()]
            add("Top channels expert summary", fig, "08_top_channels_expert_summary_black_cyan.png")

        sub = ok.copy()
        if not sub.empty and all(c in sub.columns for c in ["embedding_dimension","LLE","KS_entropy_proxy","Kaplan_Yorke_dim"]):
            grouped = sub.groupby("embedding_dimension").agg(LLE_mean=("LLE","mean"), LLE_std=("LLE","std"), KS_mean=("KS_entropy_proxy","mean"), KY_mean=("Kaplan_Yorke_dim","mean"), n=("file","count")).reset_index().sort_values("embedding_dimension")
            if not grouped.empty:
                fig, ax = plt.subplots(figsize=(10, 7), facecolor=MFDFA_BLACK); ax.errorbar(grouped["embedding_dimension"], grouped["LLE_mean"], yerr=grouped["LLE_std"].fillna(0), marker="o", linewidth=2, markersize=6, color=MFDFA_ACCENT, ecolor=MFDFA_ACCENT_SOFT, capsize=4, label="Mean LLE ± SD"); ax.plot(grouped["embedding_dimension"], grouped["KS_mean"], marker="s", linewidth=1.8, markersize=5, color=MFDFA_ACCENT_SOFT, alpha=0.8, label="Mean KS entropy proxy"); ax.plot(grouped["embedding_dimension"], grouped["KY_mean"], marker="^", linewidth=1.8, markersize=5, color=MFDFA_WHITE, alpha=0.75, label="Mean Kaplan-Yorke dim"); ax.axhline(0, color=MFDFA_WHITE, linewidth=1.0, alpha=0.6); _mfdfa_style_ax(ax, "Embedding Dimension Sensitivity Summary", "Embedding dimension", "Aggregate metric"); leg=ax.legend(loc="best", frameon=True); [text.set_color(MFDFA_ACCENT) for text in leg.get_texts()]
                add("Embedding dimension summary", fig, "09_embedding_dimension_summary_black_cyan.png")

        metric_cols = [c for c in ["LLE","MLE","most_negative_exp","sum_exponents","mean_exponent","std_exponent","spectral_radius_abs","positive_count","negative_count","near_zero_count","KS_entropy_proxy","Kaplan_Yorke_dim","expansion_entropy","chaos_timescale","quality_score","elapsed_sec","steps_used"] if c in ok.columns]
        if len(metric_cols) >= 3:
            corr = ok[metric_cols].apply(pd.to_numeric, errors="coerce").corr()
            if not corr.empty:
                fig, ax = plt.subplots(figsize=(11, 9), facecolor=MFDFA_BLACK); im=ax.imshow(corr.to_numpy(), aspect="auto", interpolation="nearest", cmap=cyan_div, vmin=-1, vmax=1); ax.set_xticks(np.arange(len(metric_cols))); ax.set_yticks(np.arange(len(metric_cols))); ax.set_xticklabels(metric_cols, rotation=65, ha="right", color=MFDFA_ACCENT, fontsize=8); ax.set_yticklabels(metric_cols, color=MFDFA_ACCENT, fontsize=8); _mfdfa_style_ax(ax, "Expert Metric Correlation Matrix", "Metric", "Metric"); cbar=fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cbar.ax.yaxis.set_tick_params(color=MFDFA_ACCENT); plt.setp(cbar.ax.get_yticklabels(), color=MFDFA_ACCENT); cbar.outline.set_edgecolor(MFDFA_ACCENT); cbar.set_label("Pearson correlation", color=MFDFA_ACCENT)
                add("Metric correlation matrix", fig, "10_metric_correlation_matrix_black_cyan.png")

        sub = ok[np.isfinite(ok["LLE"])].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(10, 7), facecolor=MFDFA_BLACK); ax.hist(sub["LLE"], bins=min(40, max(8, int(math.sqrt(len(sub))))), color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.75); ax.axvline(0, color=MFDFA_WHITE, linewidth=1.2, alpha=0.85); ax.axvline(sub["LLE"].median(), color=MFDFA_ACCENT_SOFT, linewidth=1.5, linestyle="--", alpha=0.9); _mfdfa_style_ax(ax, "Distribution of Largest Lyapunov Exponents", "Largest Lyapunov exponent", "Count")
            add("LLE distribution", fig, "11_lle_distribution_black_cyan.png")

        if exp_cols:
            selected = []
            for _, row in ok[np.isfinite(ok["LLE"])].sort_values("LLE", ascending=False).iterrows():
                exps = row[exp_cols].to_numpy(dtype=float); exps = exps[np.isfinite(exps)]
                if exps.size >= 3: selected.append(row)
                if len(selected) >= 18: break
            if selected:
                selected_df = pd.DataFrame(selected); fig, ax = plt.subplots(figsize=(11, 7), facecolor=MFDFA_BLACK)
                for rank, (_, row) in enumerate(selected_df.iterrows(), start=1):
                    exps = row[exp_cols].to_numpy(dtype=float); exps = np.sort(exps[np.isfinite(exps)])[::-1]; xx=np.arange(1, len(exps)+1); ax.plot(xx, exps, marker="o" if rank <= 6 else None, linewidth=2.4 if rank <= 5 else 1.3, markersize=4, color=MFDFA_ACCENT, alpha=max(0.25, 1.0 - 0.035*rank));
                    if rank <= 8: ax.text(xx[-1]+0.08, exps[-1], str(row["channel"])[:18], color=MFDFA_ACCENT_SOFT, fontsize=8, va="center")
                ax.axhline(0, color=MFDFA_WHITE, linewidth=1.0, alpha=0.85); _mfdfa_style_ax(ax, f"Overlay of Top {len(selected_df)} Informative Lyapunov Spectra", "Exponent rank, sorted descending", "Lyapunov exponent")
                add("Top spectra overlay", fig, "12_top_spectra_overlay_black_cyan.png")

        sub = ok[np.isfinite(ok["positive_count"])].copy()
        if not sub.empty:
            counts = sub["positive_count"].astype(int).value_counts().sort_index(); fig, ax = plt.subplots(figsize=(9, 6), facecolor=MFDFA_BLACK); ax.bar(counts.index.astype(str), counts.values, color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.82); _mfdfa_style_ax(ax, "Count of Positive Lyapunov Directions", "Number of positive exponents", "Number of channels/files")
            add("Positive exponent count", fig, "13_positive_exponent_count_black_cyan.png")
    return paths


def _lyapunov_spectrum_custom(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    import time
    start = time.perf_counter()
    mode = str(params.get("mode") or "ultra").lower()
    if mode not in LYAP_MODE_CONFIGS:
        mode = "ultra"
    cfg = dict(LYAP_MODE_CONFIGS[mode])
    dims = _efd_parse_dims(params.get("embedding_dims"), list(cfg["embedding_dims"]))
    max_channels_raw = params.get("max_channels")
    max_channels = None if max_channels_raw in (None, "", "all") else _safe_int(max_channels_raw, 0)
    max_files_raw = params.get("max_files")
    max_files = None if max_files_raw in (None, "", "all") else _safe_int(max_files_raw, 0)
    max_analysis_samples = _safe_int(params.get("max_analysis_samples"), int(cfg["max_analysis_samples"]))
    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=max_analysis_samples)
    root = Path(rec["recording_dir"])
    signal = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels and max_channels > 0:
        signal = signal[:, :max_channels]
        channels = channels[:max_channels]
    seg_start = _safe_float(params.get("segment_start_sec"))
    seg_end = _safe_float(params.get("segment_end_sec"))
    if seg_start is not None or seg_end is not None:
        s0 = max(0, int(round((seg_start or 0.0) * fs)))
        s1 = int(round(seg_end * fs)) if seg_end is not None else signal.shape[0]
        s1 = min(signal.shape[0], max(s0 + 2, s1))
        signal = signal[s0:s1, :]
    k_neighbors = _safe_int(params.get("k_neighbors"), int(cfg["k_neighbors"]))
    theiler = _safe_int(params.get("theiler"), int(cfg["theiler"]))
    stride = _safe_int(params.get("stride"), int(cfg["stride"]))
    max_steps = _safe_int(params.get("max_steps_per_file"), int(cfg["max_steps_per_file"])) if params.get("max_steps_per_file") not in (None, "") else cfg["max_steps_per_file"]
    max_points = _safe_int(params.get("max_points"), int(cfg["max_points"])) if params.get("max_points") not in (None, "") else int(cfg["max_points"])
    query_extra = _safe_int(params.get("query_extra"), int(cfg["query_extra"]))
    chunk_size = _safe_int(params.get("chunk_size"), int(cfg["chunk_size"]))
    random_seed = _safe_int(params.get("random_seed"), 0)
    tau_samples = params.get("tau_samples")
    if tau_samples not in (None, ""):
        tau = _safe_int(tau_samples, 1)
    else:
        tau_ms = _positive_float(params.get("tau_ms"), float(cfg.get("tau_ms", 10.0)))
        tau = max(1, int(round(fs * tau_ms / 1000.0)))

    out_dir = root / "advanced_methods" / "lyapunov_spectrum_custom"
    plots_dir = out_dir / "plots_black_cyan"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    precomputed = _lyap_precomputed_embedding_paths(root, channels, dims)
    records: list[dict[str, Any]] = []
    file_counter = 0
    use_precomputed_count = 0
    generated_count = 0
    for ch_idx, channel in enumerate(channels):
        x = signal[:, ch_idx]
        for emb_dim in dims:
            if max_files and file_counter >= max_files:
                break
            file_counter += 1
            row: dict[str, Any] = {
                "file": f"{int(emb_dim)}dembedded_{channel}.npy", "stem": f"{int(emb_dim)}dembedded_{channel}",
                "path": "", "source_dir": "generated_delay_embedding", "channel": channel, "channel_index": ch_idx,
                "embedding_dimension": int(emb_dim), "status": "ok", "error": "", "n_samples_original": None,
                "n_samples_used": None, "state_dim": int(emb_dim), "subsample_step": 1, "dt_used": 1.0 / fs if fs > 0 else 1.0,
                "steps_used": None, "elapsed_sec": None, "embedding_source": "generated_delay_embedding",
            }
            t0 = time.perf_counter()
            try:
                p = precomputed.get((channel, int(emb_dim)))
                if p is not None:
                    X = np.asarray(np.load(p, allow_pickle=True), dtype=float)
                    if X.ndim != 2:
                        raise ValueError(f"bad embedding shape {X.shape}")
                    row.update({"path": str(p), "source_dir": p.parent.name, "embedding_source": "precomputed_embedding"})
                    use_precomputed_count += 1
                else:
                    X = _lyap_delay_embed_1d(x, int(emb_dim), tau)
                    generated_count += 1
                row["n_samples_original"] = int(X.shape[0]) if X.ndim == 2 else 0
                X, local_dt, subsample_step = _lyap_maybe_subsample(X, row["dt_used"], max_points)
                if X.ndim != 2 or X.shape[0] < max(8, int(emb_dim) + 4):
                    raise ValueError("insufficient embedding points")
                row["n_samples_used"] = int(X.shape[0]); row["state_dim"] = int(X.shape[1]); row["subsample_step"] = int(subsample_step); row["dt_used"] = float(local_dt)
                exps, used = _lyap_spectrum_from_trajectory(X, k_neighbors=k_neighbors, theiler=theiler, stride=stride, dt=local_dt, max_steps=max_steps, query_extra=query_extra, chunk_size=chunk_size, random_seed=random_seed + file_counter)
                row["steps_used"] = int(used)
                for i, val in enumerate(exps, start=1):
                    row[f"Exp{i}"] = _safe_float(val)
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            row["elapsed_sec"] = _safe_float(time.perf_counter() - t0)
            records.append(row)
        if max_files and file_counter >= max_files:
            break
    raw_df = pd.DataFrame(records)
    raw_csv = out_dir / "lyap_spectrum_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    expert_df = _lyap_add_expert_metrics(raw_df)
    enriched_csv = out_dir / "lyap_spectrum_with_expert_metrics.csv"
    expert_df.to_csv(enriched_csv, index=False)
    ok = expert_df[expert_df["status"] == "ok"].copy()
    outputs: dict[str, str] = {"raw_csv": str(raw_csv), "enriched_csv": str(enriched_csv), "output_dir": str(out_dir), "plots_dir": str(plots_dir)}
    if not ok.empty:
        top_lle_csv = out_dir / "top_channels_by_LLE.csv"; ok.sort_values("LLE", ascending=False).to_csv(top_lle_csv, index=False); outputs["top_lle_csv"] = str(top_lle_csv)
        top_ks_csv = out_dir / "top_channels_by_KS_entropy_proxy.csv"; ok.sort_values("KS_entropy_proxy", ascending=False).to_csv(top_ks_csv, index=False); outputs["top_complexity_csv"] = str(top_ks_csv)
        hyper_csv = out_dir / "hyperchaotic_candidates.csv"; ok[ok["hyperchaos_flag"] == True].sort_values("positive_count", ascending=False).to_csv(hyper_csv, index=False); outputs["hyperchaos_csv"] = str(hyper_csv)
        qc_cols = [c for c in ["file","channel","embedding_dimension","n_samples_original","n_samples_used","subsample_step","steps_used","elapsed_sec","quality_score","reliable_steps_flag","status","error"] if c in ok.columns]
        qc_csv = out_dir / "quality_control_flags.csv"; ok[qc_cols].sort_values("quality_score", ascending=True).to_csv(qc_csv, index=False); outputs["qc_csv"] = str(qc_csv)
        dim_csv = out_dir / "embedding_dimension_summary.csv"
        ok.groupby("embedding_dimension").agg(n_files=("file","count"), LLE_mean=("LLE","mean"), LLE_median=("LLE","median"), LLE_std=("LLE","std"), KS_entropy_mean=("KS_entropy_proxy","mean"), KY_dim_mean=("Kaplan_Yorke_dim","mean"), positive_count_mean=("positive_count","mean"), sum_exponents_mean=("sum_exponents","mean"), quality_score_mean=("quality_score","mean"), elapsed_sec_mean=("elapsed_sec","mean")).reset_index().to_csv(dim_csv, index=False); outputs["dim_summary_csv"] = str(dim_csv)
    plot_paths = _lyap_plot_images(expert_df, plots_dir)
    summary_txt = out_dir / "lyapunov_spectrum_summary.txt"
    lines = ["Custom local Lyapunov spectrum analysis", "=======================================", "", f"Mode: {mode}", f"Embedding dims: {dims}", f"Channels: {len(channels)}", f"Successful files: {int((expert_df['status'] == 'ok').sum())}", f"Errored files: {int((expert_df['status'] == 'error').sum())}", f"Precomputed embeddings used: {use_precomputed_count}", f"Generated embeddings used: {generated_count}", ""]
    if not ok.empty:
        lines.append("Top channels/files by LLE:")
        for _, r in ok.sort_values("LLE", ascending=False).head(15).iterrows():
            lines.append(f"  {r['channel']} {int(r['embedding_dimension'])}D: LLE={r['LLE']:.6g}, KS={r['KS_entropy_proxy']:.6g}, KY={r['Kaplan_Yorke_dim']:.6g}, quality={r['quality_score']:.3g}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs["summary_txt"] = str(summary_txt)
    elapsed = time.perf_counter() - start
    summary = {
        "mode": mode, "n_channels": int(len(channels)), "embedding_dims": dims, "n_files": int(len(expert_df)),
        "successful_files": int((expert_df["status"] == "ok").sum()) if "status" in expert_df else 0,
        "errored_files": int((expert_df["status"] == "error").sum()) if "status" in expert_df else 0,
        "precomputed_embeddings_used": int(use_precomputed_count), "generated_embeddings_used": int(generated_count),
        "k_neighbors": int(k_neighbors), "theiler": int(theiler), "stride": int(stride), "max_steps_per_file": max_steps,
        "max_points": int(max_points), "tau_samples": int(tau), "analysis_samples": int(signal.shape[0]),
        "mean_LLE": _safe_float(np.nanmean(ok["LLE"])) if not ok.empty else None,
        "max_LLE": _safe_float(np.nanmax(ok["LLE"])) if not ok.empty else None,
        "mean_KS_entropy_proxy": _safe_float(np.nanmean(ok["KS_entropy_proxy"])) if not ok.empty else None,
        "mean_Kaplan_Yorke_dim": _safe_float(np.nanmean(ok["Kaplan_Yorke_dim"])) if not ok.empty else None,
        "hyperchaotic_candidates": int(ok["hyperchaos_flag"].sum()) if not ok.empty else 0,
        "plot_count": int(len(plot_paths)), "elapsed_sec": _safe_float(elapsed),
    }
    return {"lyapunov_spectrum_custom": {"summary": summary, "rows": _json_safe(expert_df.head(5000).to_dict("records")), "plot_paths": plot_paths, "outputs": outputs}}



ARNOLD_KURAMOTO_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {
        "a_count": 12, "b_count": 12, "a_min": 0.1, "a_max": 2.0, "b_min": 0.1, "b_max": 2.0,
        "t_end": 8.0, "n_t_eval": 120, "transient_fraction": 0.5,
        "K": 1.0, "freq_min": 1.0, "freq_max": 45.0, "max_analysis_samples": 60_000,
        "rtol": 1e-3, "atol": 1e-5, "solver_method": "RK45", "dpi": 140, "contours": False,
        "description": "Very fast preview with drive-locking diagnostics.",
    },
    "fast": {
        "a_count": 25, "b_count": 25, "a_min": 0.1, "a_max": 2.0, "b_min": 0.1, "b_max": 2.0,
        "t_end": 20.0, "n_t_eval": 250, "transient_fraction": 0.5,
        "K": 1.0, "freq_min": 1.0, "freq_max": 45.0, "max_analysis_samples": 100_000,
        "rtol": 3e-4, "atol": 1e-6, "solver_method": "RK45", "dpi": 170, "contours": False,
        "description": "Fast exploratory phase-forced Kuramoto run.",
    },
    "balanced": {
        "a_count": 40, "b_count": 40, "a_min": 0.1, "a_max": 2.0, "b_min": 0.1, "b_max": 2.0,
        "t_end": 40.0, "n_t_eval": 450, "transient_fraction": 0.5,
        "K": 1.0, "freq_min": 1.0, "freq_max": 45.0, "max_analysis_samples": 160_000,
        "rtol": 1e-4, "atol": 1e-7, "solver_method": "RK45", "dpi": 200, "contours": True,
        "description": "Good default run with contour overlays.",
    },
    "full": {
        "a_count": 50, "b_count": 50, "a_min": 0.1, "a_max": 2.0, "b_min": 0.1, "b_max": 2.0,
        "t_end": 60.0, "n_t_eval": 600, "transient_fraction": 0.5,
        "K": 1.0, "freq_min": 1.0, "freq_max": 45.0, "max_analysis_samples": 240_000,
        "rtol": 1e-5, "atol": 1e-7, "solver_method": "RK45", "dpi": 220, "contours": True,
        "description": "Full-quality phase-forced Arnold tongue run.",
    },
}


def _arnold_estimate_dominant_frequencies(eeg_segment: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    eeg_segment = np.asarray(eeg_segment, dtype=float)
    if eeg_segment.ndim != 2 or eeg_segment.shape[0] < 4:
        raise ValueError("Need a 2D samples x channels EEG segment with at least 4 samples.")
    X = eeg_segment - np.mean(eeg_segment, axis=0, keepdims=True)
    fft_vals = np.fft.rfft(X, axis=0)
    freqs = np.fft.rfftfreq(X.shape[0], d=1.0 / max(float(fs), 1e-12))
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        raise ValueError(f"No FFT frequencies found inside band {band}")
    band_freqs = freqs[band_mask]
    band_power = np.abs(fft_vals[band_mask, :])
    peak_idx = np.argmax(band_power, axis=0)
    return band_freqs[peak_idx]


def _arnold_kuramoto_rhs(t: float, theta: np.ndarray, omega: np.ndarray, K: float, a: float, b: float) -> np.ndarray:
    z = np.mean(np.exp(1j * theta))
    r = np.abs(z)
    psi = np.angle(z)
    coupling = K * r * np.sin(psi - theta)
    drive_phase = 2.0 * np.pi * b * t
    drive = a * np.sin(drive_phase - theta)
    return omega + coupling + drive


def _arnold_simulate_kuramoto(omega: np.ndarray, K: float, a: float, b: float, theta0: np.ndarray, t_span: tuple[float, float], t_eval: np.ndarray, solver_method: str, rtol: float, atol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.integrate import solve_ivp
    sol = solve_ivp(
        _arnold_kuramoto_rhs,
        t_span=t_span,
        y0=theta0,
        t_eval=t_eval,
        args=(omega, K, a, b),
        method=solver_method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    theta = np.asarray(sol.y, dtype=float)
    z = np.mean(np.exp(1j * theta), axis=0)
    r = np.abs(z)
    return np.asarray(sol.t, dtype=float), theta, r


def _arnold_apply_colorbar_style(cbar: Any, label: str) -> None:
    cbar.ax.yaxis.set_tick_params(color=MFDFA_ACCENT)
    for tick in cbar.ax.get_yticklabels():
        tick.set_color(MFDFA_ACCENT)
    cbar.outline.set_edgecolor(MFDFA_ACCENT)
    cbar.set_label(label, color=MFDFA_ACCENT)


def _arnold_robust_limits(arr: Any, lower: float = 2, upper: float = 98, pad_fraction: float = 0.08, min_span: float = 1e-8, bounds: tuple[float, float] | None = None) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return bounds if bounds is not None else (0.0, 1.0)
    vmin, vmax = np.nanpercentile(finite, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    center = 0.5 * (vmin + vmax)
    span = vmax - vmin
    if not np.isfinite(span) or span < min_span:
        span = max(abs(center) * 1e-6, min_span)
        vmin = center - 0.5 * span
        vmax = center + 0.5 * span
    else:
        pad = pad_fraction * span
        vmin -= pad
        vmax += pad
    if bounds is not None:
        lo, hi = bounds
        vmin = max(lo, vmin)
        vmax = min(hi, vmax)
    if vmax <= vmin:
        if bounds is not None:
            return bounds
        delta = max(abs(center) * 1e-6, min_span)
        return float(center - delta), float(center + delta)
    return float(vmin), float(vmax)


def _arnold_tongues_kuramoto(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    import time
    start_clock = time.perf_counter()
    mode = str(params.get("mode") or "ultra").lower()
    if mode not in ARNOLD_KURAMOTO_MODE_CONFIGS:
        mode = "ultra"
    cfg = dict(ARNOLD_KURAMOTO_MODE_CONFIGS[mode])

    for key in ("a_min", "a_max", "b_min", "b_max", "t_end", "transient_fraction", "K", "freq_min", "freq_max", "rtol", "atol"):
        if params.get(key) not in (None, ""):
            cfg[key] = float(params[key])
    for key in ("a_count", "b_count", "n_t_eval", "max_analysis_samples"):
        if params.get(key) not in (None, ""):
            cfg[key] = _safe_int(params.get(key), int(cfg[key]))
    if params.get("solver_method") not in (None, ""):
        cfg["solver_method"] = str(params.get("solver_method"))
    if params.get("contours") not in (None, ""):
        cfg["contours"] = str(params.get("contours")).lower() in {"1", "true", "yes", "y", "on"}

    max_channels_raw = params.get("max_channels")
    max_channels = None if max_channels_raw in (None, "", "all") else _safe_int(max_channels_raw, 0)
    random_seed = _safe_int(params.get("random_seed"), 42)

    rec = _load_recording(recording_dir, sampling_rate=params.get("sampling_rate"), max_samples=cfg["max_analysis_samples"])
    root = Path(rec["recording_dir"])
    signal = np.asarray(rec["signal"], dtype=float)
    fs = float(rec["sampling_rate_hz"] or 1.0)
    channels = list(rec["channels"])
    if max_channels and max_channels > 0:
        signal = signal[:, :max_channels]
        channels = channels[:max_channels]

    seg_start = _safe_float(params.get("segment_start_sec"))
    seg_end = _safe_float(params.get("segment_end_sec"))
    if seg_start is not None or seg_end is not None:
        s0 = max(0, int(round((seg_start or 0.0) * fs)))
        s1 = int(round(seg_end * fs)) if seg_end is not None else signal.shape[0]
        s1 = min(signal.shape[0], max(s0 + 4, s1))
        signal = signal[s0:s1, :]

    if signal.ndim != 2 or signal.shape[1] == 0:
        raise ValueError("No channels available for Arnold/Kuramoto analysis.")
    if signal.shape[0] < 8:
        raise ValueError("Recording is too short for FFT-based frequency estimation.")

    band = (float(cfg["freq_min"]), float(cfg["freq_max"]))
    dominant_freqs = _arnold_estimate_dominant_frequencies(signal, fs, band)
    omega = (dominant_freqs - np.mean(dominant_freqs)) / (np.std(dominant_freqs) + 1e-12)

    a_values = np.linspace(float(cfg["a_min"]), float(cfg["a_max"]), max(2, int(cfg["a_count"])))
    b_values = np.linspace(float(cfg["b_min"]), float(cfg["b_max"]), max(2, int(cfg["b_count"])))
    t_eval = np.linspace(0.0, float(cfg["t_end"]), max(8, int(cfg["n_t_eval"])))
    transient_idx = int(np.clip(float(cfg["transient_fraction"]), 0.0, 0.95) * len(t_eval))
    K = float(cfg["K"])
    solver_method = str(cfg["solver_method"])
    rtol = float(cfg["rtol"])
    atol = float(cfg["atol"])

    rng = np.random.default_rng(random_seed)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, len(channels))
    sync = np.zeros((len(a_values), len(b_values)), dtype=float)
    sync_std = np.zeros_like(sync)
    drive_lock = np.zeros_like(sync)
    collective_drive_lock = np.zeros_like(sync)
    freq_error = np.zeros_like(sync)

    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            t, theta, r = _arnold_simulate_kuramoto(omega, K, float(a), float(b), theta0, (0.0, float(cfg["t_end"])), t_eval, solver_method, rtol, atol)
            r_ss = r[transient_idx:] if transient_idx < len(r) else r
            theta_ss = theta[:, transient_idx:] if transient_idx < theta.shape[1] else theta
            t_ss = t[transient_idx:] if transient_idx < len(t) else t
            sync[i, j] = float(np.mean(r_ss))
            sync_std[i, j] = float(np.std(r_ss))

            drive_phase = 2.0 * np.pi * float(b) * t_ss
            relative_phase = np.exp(1j * (theta_ss - drive_phase[None, :]))
            drive_lock[i, j] = float(np.mean(np.abs(np.mean(relative_phase, axis=1))))

            z_ss = np.mean(np.exp(1j * theta_ss), axis=0)
            collective_phase = np.angle(z_ss)
            collective_drive_lock[i, j] = float(np.abs(np.mean(np.exp(1j * (collective_phase - drive_phase)))))

            if t_ss.size >= 2 and t_ss[-1] > t_ss[0]:
                unwrapped_theta = np.unwrap(theta_ss, axis=1)
                observed_omega_each = (unwrapped_theta[:, -1] - unwrapped_theta[:, 0]) / (t_ss[-1] - t_ss[0])
                drive_omega = 2.0 * np.pi * float(b)
                freq_error[i, j] = float(np.mean(np.abs(observed_omega_each - drive_omega)))
            else:
                freq_error[i, j] = np.nan

    out_dir = root / "advanced_methods" / "arnold_tongues_kuramoto"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "arnold_tongues_results.npz"
    np.savez_compressed(
        npz_path,
        synchronization_array=sync,
        sync_std_array=sync_std,
        drive_lock_array=drive_lock,
        collective_drive_lock_array=collective_drive_lock,
        freq_error_array=freq_error,
        a_values=a_values,
        b_values=b_values,
        omega=omega,
        dominant_freqs_hz=dominant_freqs,
        channels=np.array(channels, dtype=object),
        K=K,
        t_eval=t_eval,
        transient_fraction=float(cfg["transient_fraction"]),
        sampling_rate=fs,
        mode=mode,
        speed_cfg=str(cfg),
    )

    grid_rows = [
        {
            "mode": mode,
            "a": float(a), "b": float(b),
            "mean_sync": float(sync[i, j]),
            "std_sync": float(sync_std[i, j]),
            "drive_lock": float(drive_lock[i, j]),
            "collective_drive_lock": float(collective_drive_lock[i, j]),
            "frequency_error": float(freq_error[i, j]),
        }
        for i, a in enumerate(a_values)
        for j, b in enumerate(b_values)
    ]
    grid_df = pd.DataFrame(grid_rows)
    grid_csv = out_dir / "arnold_tongues_grid.csv"
    grid_df.to_csv(grid_csv, index=False)

    omega_df = pd.DataFrame({"channel": channels, "dominant_frequency_hz": dominant_freqs, "omega_dimensionless": omega})
    omega_csv = out_dir / "kuramoto_natural_frequencies.csv"
    omega_df.to_csv(omega_csv, index=False)

    plt, cyan_seq = _mfdfa_import_plotting()
    _mfdfa_apply_style(plt)
    from matplotlib.colors import LinearSegmentedColormap
    cyan_div = LinearSegmentedColormap.from_list("arnold_cyan_diverging", [MFDFA_ACCENT_DIM, MFDFA_BLACK, MFDFA_ACCENT_SOFT], N=256)
    plot_paths: list[dict[str, str]] = []

    def add_plot(title: str, fig: Any, filename: str, dpi: int | None = None) -> None:
        plot_paths.append({"title": title, "path": _mfdfa_save_fig(fig, plots_dir / filename, dpi=int(dpi or cfg.get("dpi", 190)))})

    def style_colorbar(cbar: Any, label: str) -> None:
        _arnold_apply_colorbar_style(cbar, label)

    def maybe_contour(ax: Any, arr: np.ndarray, vmin: float | None, vmax: float | None, color: str | None = None) -> None:
        if not bool(cfg.get("contours", False)):
            return
        try:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return
            cmin = float(np.nanmin(finite) if vmin is None else vmin)
            cmax = float(np.nanmax(finite) if vmax is None else vmax)
            if cmax <= cmin:
                return
            B, A = np.meshgrid(b_values, a_values)
            levels = np.linspace(cmin, cmax, 8)
            ax.contour(B, A, arr, levels=levels, colors=color or MFDFA_ACCENT, linewidths=0.45, alpha=0.35)
        except Exception:
            return

    def heatmap(title: str, colorbar_label: str, filename: str, data: np.ndarray, cmap: Any = cyan_seq, vmin: float | None = None, vmax: float | None = None, robust: bool = False, bounds: tuple[float, float] | None = None, contour: bool = False) -> None:
        arr = np.asarray(data, dtype=float)
        if robust:
            vmin, vmax = _arnold_robust_limits(arr, bounds=bounds)
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=MFDFA_BLACK)
        im = ax.imshow(np.ma.masked_invalid(arr), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, extent=[b_values.min(), b_values.max(), a_values.min(), a_values.max()], vmin=vmin, vmax=vmax)
        if contour:
            maybe_contour(ax, arr, vmin, vmax)
        _mfdfa_style_ax(ax, title, "Driving frequency parameter b", "Driving amplitude a", grid=True)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        style_colorbar(cbar, colorbar_label)
        add_plot(title, fig, filename)

    def line_band(title: str, xlabel: str, ylabel: str, filename: str, x: np.ndarray, y: np.ndarray, ystd: np.ndarray | None = None, ylim_bounds: tuple[float, float] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=MFDFA_BLACK)
        ax.plot(x, y, color=MFDFA_ACCENT, marker="o", linewidth=2.0)
        if ystd is not None:
            ax.fill_between(x, y - ystd, y + ystd, color=MFDFA_ACCENT, alpha=0.15)
        if ylim_bounds is not None:
            ymin, ymax = _arnold_robust_limits(y, lower=0, upper=100, pad_fraction=0.15, bounds=ylim_bounds)
            ax.set_ylim(ymin, ymax)
        _mfdfa_style_ax(ax, title, xlabel, ylabel)
        add_plot(title, fig, filename)

    sync_mean = float(np.nanmean(sync)); sync_range = float(np.nanmax(sync) - np.nanmin(sync))
    std_mean = float(np.nanmean(sync_std)); drive_lock_mean = float(np.nanmean(drive_lock)); collective_lock_mean = float(np.nanmean(collective_drive_lock)); freq_error_mean = float(np.nanmean(freq_error))
    sync_contrast = sync - sync_mean
    std_contrast = sync_std - std_mean
    drive_lock_contrast = drive_lock - drive_lock_mean
    freq_error_contrast = freq_error - freq_error_mean

    def absmax(arr: np.ndarray) -> float:
        val = float(np.nanmax(np.abs(arr))) if np.any(np.isfinite(arr)) else 1e-12
        return val if np.isfinite(val) and val > 0 else 1e-12

    heatmap("Drive-Locking Arnold Tongue", "Mean oscillator-to-drive locking index", "01_kuramoto_drive_locking_arnold_tongue.png", drive_lock, cmap=plt.get_cmap("viridis"), robust=True, bounds=(0.0, 1.0), contour=True)
    heatmap("Collective Phase Drive Locking", "Collective phase-to-drive locking index", "02_kuramoto_collective_drive_locking.png", collective_drive_lock, cmap=plt.get_cmap("viridis"), robust=True, bounds=(0.0, 1.0), contour=True)
    heatmap("Frequency Error Relative to External Drive", "Mean |observed omega - drive omega|", "03_kuramoto_drive_frequency_error.png", freq_error, cmap=plt.get_cmap("magma_r"), robust=True, contour=True)
    heatmap("Kuramoto Mean Synchronization, Visible Scale", "Mean oscillator-to-oscillator synchronization r", "04_kuramoto_mean_sync_visible_scale.png", sync, cmap=plt.get_cmap("viridis"), robust=True, bounds=(0.0, 1.0), contour=True)
    heatmap("Kuramoto Mean Synchronization, Honest 0-to-1 Scale", "Mean synchronization r", "05_kuramoto_mean_sync_honest_scale.png", sync, cmap=plt.get_cmap("viridis"), vmin=0.0, vmax=1.0, contour=False)
    heatmap("Mean Synchronization Contrast", "Mean sync minus global mean", "06_kuramoto_mean_sync_contrast.png", sync_contrast, cmap=plt.get_cmap("coolwarm"), vmin=-absmax(sync_contrast), vmax=absmax(sync_contrast))
    heatmap("Synchronization Variability", "Synchronization std", "07_kuramoto_sync_std_heatmap.png", sync_std, cmap=plt.get_cmap("viridis"), robust=True, bounds=(0.0, 1.0), contour=True)
    heatmap("Synchronization Variability Contrast", "Std sync minus global mean std", "08_kuramoto_sync_std_contrast.png", std_contrast, cmap=plt.get_cmap("coolwarm"), vmin=-absmax(std_contrast), vmax=absmax(std_contrast))
    heatmap("Drive-Locking Contrast", "Drive lock minus global mean drive lock", "09_kuramoto_drive_locking_contrast.png", drive_lock_contrast, cmap=plt.get_cmap("coolwarm"), vmin=-absmax(drive_lock_contrast), vmax=absmax(drive_lock_contrast))
    heatmap("Frequency Error Contrast", "Frequency error minus global mean frequency error", "10_kuramoto_frequency_error_contrast.png", freq_error_contrast, cmap=plt.get_cmap("coolwarm"), vmin=-absmax(freq_error_contrast), vmax=absmax(freq_error_contrast))

    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.28 * len(channels))), facecolor=MFDFA_BLACK)
    order = np.argsort(dominant_freqs); y = np.arange(len(channels))
    ax.barh(y, dominant_freqs[order], color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.86)
    ax.set_yticks(y); ax.set_yticklabels(np.array(channels)[order], color=MFDFA_ACCENT, fontsize=8)
    _mfdfa_style_ax(ax, "Dominant EEG Frequency by Channel", "Dominant frequency (Hz)", "Channel")
    add_plot("Dominant EEG Frequency by Channel", fig, "11_dominant_eeg_frequencies_by_channel.png")

    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.28 * len(channels))), facecolor=MFDFA_BLACK)
    order = np.argsort(omega); y = np.arange(len(channels))
    ax.barh(y, omega[order], color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.86); ax.axvline(0, color=MFDFA_WHITE, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_yticks(y); ax.set_yticklabels(np.array(channels)[order], color=MFDFA_ACCENT, fontsize=8)
    _mfdfa_style_ax(ax, "Kuramoto Natural Frequencies Derived from EEG", "Dimensionless omega", "Channel")
    add_plot("Kuramoto Natural Frequencies Derived from EEG", fig, "12_kuramoto_omega_by_channel.png")

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=MFDFA_BLACK)
    ax.scatter(dominant_freqs, omega, color=MFDFA_ACCENT, edgecolors=MFDFA_WHITE, s=60, alpha=0.85)
    for ch, fdom, om in zip(channels, dominant_freqs, omega):
        ax.text(fdom, om, str(ch)[:12], color=MFDFA_ACCENT_SOFT, fontsize=8, alpha=0.78)
    ax.axhline(0.0, color=MFDFA_WHITE, linestyle="--", alpha=0.6); ax.axvline(float(np.mean(dominant_freqs)), color=MFDFA_WHITE, linestyle="--", alpha=0.6)
    _mfdfa_style_ax(ax, "Dominant EEG Frequency vs Standardized Omega", "Dominant frequency (Hz)", "Dimensionless omega")
    add_plot("Dominant EEG Frequency vs Standardized Omega", fig, "13_dominant_frequency_vs_omega.png")

    line_band("Mean Synchronization Averaged over Amplitude", "Driving frequency parameter b", "Mean synchronization r", "14_mean_sync_vs_b.png", b_values, np.nanmean(sync, axis=0), np.nanstd(sync, axis=0), (0.0, 1.0))
    line_band("Mean Synchronization Averaged over Driving Frequency", "Driving amplitude a", "Mean synchronization r", "15_mean_sync_vs_a.png", a_values, np.nanmean(sync, axis=1), np.nanstd(sync, axis=1), (0.0, 1.0))
    line_band("Drive Locking Averaged over Amplitude", "Driving frequency parameter b", "Drive-locking index", "16_drive_lock_vs_b.png", b_values, np.nanmean(drive_lock, axis=0), np.nanstd(drive_lock, axis=0), (0.0, 1.0))
    line_band("Drive Locking Averaged over Driving Frequency", "Driving amplitude a", "Drive-locking index", "17_drive_lock_vs_a.png", a_values, np.nanmean(drive_lock, axis=1), np.nanstd(drive_lock, axis=1), (0.0, 1.0))
    line_band("Frequency Error Averaged over Amplitude", "Driving frequency parameter b", "Mean frequency error", "18_frequency_error_vs_b.png", b_values, np.nanmean(freq_error, axis=0), np.nanstd(freq_error, axis=0), None)
    line_band("Frequency Error Averaged over Driving Frequency", "Driving amplitude a", "Mean frequency error", "19_frequency_error_vs_a.png", a_values, np.nanmean(freq_error, axis=1), np.nanstd(freq_error, axis=1), None)

    best_drive_idx = np.unravel_index(int(np.nanargmax(drive_lock)), drive_lock.shape)
    best_sync_idx = np.unravel_index(int(np.nanargmax(sync)), sync.shape)
    best_a = float(a_values[best_drive_idx[0]]); best_b = float(b_values[best_drive_idx[1]])

    top_csv = out_dir / "top_arnold_tongue_parameter_pairs.csv"
    grid_df.sort_values(["drive_lock", "mean_sync"], ascending=False).head(50).to_csv(top_csv, index=False)

    summary_txt = out_dir / "arnold_tongues_summary.txt"
    lines = [
        "Arnold Tongues via Phase-Forced Kuramoto Model",
        "=============================================",
        "",
        f"Recording: {root}",
        f"Mode: {mode}",
        f"Speed description: {cfg.get('description', '')}",
        f"Sampling rate used: {fs}",
        f"Channels / oscillators: {len(channels)}",
        f"Coupling K: {K}",
        f"Drive amplitude range: [{float(a_values.min())}, {float(a_values.max())}], count={len(a_values)}",
        f"Drive frequency range: [{float(b_values.min())}, {float(b_values.max())}], count={len(b_values)}",
        f"Simulation span: [0, {float(cfg['t_end'])}], n_t_eval={len(t_eval)}",
        f"Transient fraction discarded: {float(cfg['transient_fraction'])}",
        f"Solver: {solver_method}, rtol={rtol}, atol={atol}",
        f"Frequency band for omega estimation: {band}",
        "External drive model: a * sin(2*pi*b*t - theta_i)",
        "",
        "Diagnostics:",
        f"mean_sync min={float(np.nanmin(sync)):.10f}, max={float(np.nanmax(sync)):.10f}, mean={sync_mean:.10f}, range={sync_range:.10e}",
        f"std_sync min={float(np.nanmin(sync_std)):.10f}, max={float(np.nanmax(sync_std)):.10f}, mean={std_mean:.10f}",
        f"drive_lock min={float(np.nanmin(drive_lock)):.10f}, max={float(np.nanmax(drive_lock)):.10f}, mean={drive_lock_mean:.10f}",
        f"collective_drive_lock min={float(np.nanmin(collective_drive_lock)):.10f}, max={float(np.nanmax(collective_drive_lock)):.10f}, mean={collective_lock_mean:.10f}",
        f"frequency_error min={float(np.nanmin(freq_error)):.10f}, max={float(np.nanmax(freq_error)):.10f}, mean={freq_error_mean:.10f}",
        f"Best drive-lock cell: a={best_a:.6g}, b={best_b:.6g}, drive_lock={float(drive_lock[best_drive_idx]):.6g}",
        f"Best mean-sync cell: a={float(a_values[best_sync_idx[0]]):.6g}, b={float(b_values[best_sync_idx[1]]):.6g}, mean_sync={float(sync[best_sync_idx]):.6g}",
        "",
        "Dominant frequencies and omega values:",
    ]
    for ch, fdom, om in zip(channels, dominant_freqs, omega):
        lines.append(f"  {ch}: dominant_freq={float(fdom):.6f} Hz | omega={float(om):.6f}")
    lines.extend(["", "Plot files:"])
    for item in plot_paths:
        lines.append(Path(item["path"]).name)
    lines.extend([
        "",
        "Interpretation note:",
        "Mean synchronization r measures oscillator-to-oscillator coherence. Drive locking and frequency error measure entrainment to the external drive. For Arnold tongues, the drive-locking and frequency-error plots are usually more meaningful than mean synchronization alone.",
    ])
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = {
        "npz": str(npz_path), "grid_csv": str(grid_csv), "omega_csv": str(omega_csv), "top_pairs_csv": str(top_csv),
        "summary_txt": str(summary_txt), "output_dir": str(out_dir), "plots_dir": str(plots_dir),
    }
    elapsed = time.perf_counter() - start_clock
    summary = {
        "mode": mode, "description": cfg.get("description"), "n_channels": int(len(channels)), "analysis_samples": int(signal.shape[0]), "sampling_rate_hz": fs,
        "K": K, "a_count": int(len(a_values)), "b_count": int(len(b_values)), "grid_points": int(len(a_values) * len(b_values)),
        "t_end": float(cfg["t_end"]), "n_t_eval": int(len(t_eval)), "transient_fraction": float(cfg["transient_fraction"]),
        "freq_band_min": float(band[0]), "freq_band_max": float(band[1]),
        "mean_sync": _safe_float(sync_mean), "max_sync": _safe_float(np.nanmax(sync)), "sync_range": _safe_float(sync_range),
        "mean_drive_lock": _safe_float(drive_lock_mean), "max_drive_lock": _safe_float(np.nanmax(drive_lock)),
        "mean_collective_drive_lock": _safe_float(collective_lock_mean), "max_collective_drive_lock": _safe_float(np.nanmax(collective_drive_lock)),
        "mean_frequency_error": _safe_float(freq_error_mean), "min_frequency_error": _safe_float(np.nanmin(freq_error)),
        "best_a": best_a, "best_b": best_b, "best_drive_lock": _safe_float(drive_lock[best_drive_idx]),
        "best_sync_a": _safe_float(a_values[best_sync_idx[0]]), "best_sync_b": _safe_float(b_values[best_sync_idx[1]]), "best_sync": _safe_float(sync[best_sync_idx]),
        "mean_dominant_frequency_hz": _safe_float(np.nanmean(dominant_freqs)), "plot_count": int(len(plot_paths)), "elapsed_sec": _safe_float(elapsed),
    }
    return {"arnold_tongues_kuramoto": {"summary": summary, "grid_rows": _json_safe(grid_df.head(5000).to_dict("records")), "frequency_rows": _json_safe(omega_df.to_dict("records")), "plot_paths": plot_paths, "outputs": outputs}}


CIRCLE_MAP_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {
        "n_omega": 40, "n_K": 40, "omega_min": 0.0, "omega_max": 1.0, "K_min": 0.0, "K_max": 4.0 * math.pi,
        "iterations": 25, "tol": 1e-5, "max_phase_samples": 800, "max_analysis_samples": 60_000, "lock_threshold": 0.5, "dpi": 140,
        "description": "Very fast fixed-point-locking preview.",
    },
    "fast": {
        "n_omega": 75, "n_K": 75, "omega_min": 0.0, "omega_max": 1.0, "K_min": 0.0, "K_max": 4.0 * math.pi,
        "iterations": 40, "tol": 3e-6, "max_phase_samples": 2_000, "max_analysis_samples": 100_000, "lock_threshold": 0.5, "dpi": 170,
        "description": "Fast exploratory fixed-point-locking grid.",
    },
    "balanced": {
        "n_omega": 110, "n_K": 110, "omega_min": 0.0, "omega_max": 1.0, "K_min": 0.0, "K_max": 4.0 * math.pi,
        "iterations": 60, "tol": 1e-6, "max_phase_samples": 5_000, "max_analysis_samples": 160_000, "lock_threshold": 0.5, "dpi": 200,
        "description": "Notebook-like circle-map run with fewer grid points than full.",
    },
    "full": {
        "n_omega": 150, "n_K": 150, "omega_min": 0.0, "omega_max": 1.0, "K_min": 0.0, "K_max": 4.0 * math.pi,
        "iterations": 60, "tol": 1e-6, "max_phase_samples": 5_000, "max_analysis_samples": 240_000, "lock_threshold": 0.5, "dpi": 220,
        "description": "Full notebook grid: 150×150, 60 iterations, 5000 phase samples.",
    },
}


def _circle_to_cycle_phase(rad_phase: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(rad_phase, dtype=float) / (2.0 * math.pi), 1.0)


def _circle_mean_phase(phases_rad: np.ndarray, axis: int = 1) -> np.ndarray:
    z = np.mean(np.exp(1j * phases_rad), axis=axis)
    return np.angle(z)


def _circle_uniform_subsample(x: np.ndarray, max_n: int) -> tuple[np.ndarray, int]:
    x = np.asarray(x)
    if max_n <= 0 or len(x) <= max_n:
        return x, 1
    idx = np.linspace(0, len(x) - 1, max_n).astype(int)
    step = max(1, int(round(len(x) / max_n)))
    return x[idx], step


def _circle_map(theta: np.ndarray, omega: np.ndarray, K: float) -> np.ndarray:
    return np.mod(theta + omega - (float(K) / (2.0 * math.pi)) * np.sin(2.0 * math.pi * theta), 1.0)


def _circle_wrapped_cycle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((a - b + 0.5) % 1.0) - 0.5)


def _circle_fixed_point_locked_proportion_row(K: float, omegas: np.ndarray, phases: np.ndarray, iterations: int, tol: float) -> np.ndarray:
    omegas = np.asarray(omegas, dtype=float)
    phases = np.asarray(phases, dtype=float)
    theta = np.broadcast_to(phases[None, :], (len(omegas), len(phases))).copy()
    omega_col = omegas[:, None]
    for _ in range(int(iterations)):
        theta = _circle_map(theta, omega_col, K)
    theta_next = _circle_map(theta, omega_col, K)
    final_delta = _circle_wrapped_cycle_distance(theta_next, theta)
    return np.mean(final_delta < float(tol), axis=1)



CIRCLE_DENSITY_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "ultra": {
        "Omega": 1.0 / 3.0,
        "n_K": 60,
        "K_min": 0.0,
        "K_max": 4.0 * math.pi,
        "iterations": 25,
        "n_bins": 32,
        "max_samples_per_channel": 800,
        "max_analysis_samples": 60_000,
        "use_bandpass": True,
        "phase_min_hz": 1.0,
        "phase_max_hz": 45.0,
        "filter_order": 4,
        "dpi": 140,
        "description": "Very fast final-state density preview.",
    },
    "fast": {
        "Omega": 1.0 / 3.0,
        "n_K": 120,
        "K_min": 0.0,
        "K_max": 4.0 * math.pi,
        "iterations": 35,
        "n_bins": 40,
        "max_samples_per_channel": 2_000,
        "max_analysis_samples": 100_000,
        "use_bandpass": True,
        "phase_min_hz": 1.0,
        "phase_max_hz": 45.0,
        "filter_order": 4,
        "dpi": 170,
        "description": "Fast exploratory converged-density run.",
    },
    "balanced": {
        "Omega": 1.0 / 3.0,
        "n_K": 200,
        "K_min": 0.0,
        "K_max": 4.0 * math.pi,
        "iterations": 50,
        "n_bins": 50,
        "max_samples_per_channel": 5_000,
        "max_analysis_samples": 160_000,
        "use_bandpass": True,
        "phase_min_hz": 1.0,
        "phase_max_hz": 45.0,
        "filter_order": 4,
        "dpi": 200,
        "description": "Notebook-like final-state density run with fewer K points than full.",
    },
    "full": {
        "Omega": 1.0 / 3.0,
        "n_K": 300,
        "K_min": 0.0,
        "K_max": 4.0 * math.pi,
        "iterations": 50,
        "n_bins": 50,
        "max_samples_per_channel": 5_000,
        "max_analysis_samples": 240_000,
        "use_bandpass": True,
        "phase_min_hz": 1.0,
        "phase_max_hz": 45.0,
        "filter_order": 4,
        "dpi": 220,
        "description": "Full notebook settings: 300 K values, 50 iterations, 50 histogram bins, 5000 samples/channel.",
    },
}


def _circle_bool(value: Any, default: bool = True) -> bool:
    if value in (None, ""):
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _circle_bandpass_filter_eeg(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    fs = float(fs or 1.0)
    nyq = 0.5 * fs
    if nyq <= 0:
        return x
    low = max(0.0, float(low_hz))
    high = min(float(high_hz), nyq * 0.999)
    if high <= 0 or (low <= 0 and high >= nyq * 0.998):
        return x
    if low <= 0:
        sos = butter(int(order), high / nyq, btype="lowpass", output="sos")
    elif high >= nyq:
        sos = butter(int(order), low / nyq, btype="highpass", output="sos")
    else:
        if high <= low:
            raise ValueError("phase_max_hz must be greater than phase_min_hz and below Nyquist.")
        sos = butter(int(order), [low / nyq, high / nyq], btype="bandpass", output="sos")
    # filtfilt needs enough samples. If the segment is too short, return de-meaned raw data.
    if x.shape[0] < max(16, int(order) * 6):
        return x
    return sosfiltfilt(sos, x, axis=0)


def _circle_random_or_uniform_subsample(x: np.ndarray, max_n: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, int]:
    x = np.asarray(x)
    if max_n <= 0 or len(x) <= max_n:
        return x, 1
    if rng is None:
        idx = np.linspace(0, len(x) - 1, max_n).astype(int)
    else:
        idx = np.sort(rng.choice(len(x), size=max_n, replace=False))
    step = max(1, int(round(len(x) / max_n)))
    return x[idx], step


def _circle_circular_mean_cycles(values: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    angles = 2.0 * math.pi * values
    if weights is None:
        weights_arr = np.ones_like(values, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / (np.sum(weights_arr) + 1e-12)
    z = np.sum(weights_arr * np.exp(1j * angles))
    mean_cycles = np.mod(np.angle(z) / (2.0 * math.pi), 1.0)
    resultant = np.abs(z)
    return float(mean_cycles), float(resultant)


def _circle_normalized_concentration_from_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    raw = np.sum(prob ** 2, axis=-1)
    n_bins = max(1, int(prob.shape[-1]))
    uniform_val = 1.0 / n_bins
    norm = (raw - uniform_val) / (1.0 - uniform_val + 1e-12)
    return np.clip(norm, 0.0, 1.0)


def _circle_map_arnold_tongues(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    import time
    from scipy.signal import hilbert
    start_clock = time.perf_counter()
    mode = str(params.get("mode") or "ultra").lower()
    if mode not in CIRCLE_MAP_MODE_CONFIGS:
        mode = "ultra"
    cfg = dict(CIRCLE_MAP_MODE_CONFIGS[mode])

    n_omega = _safe_int(params.get("n_omega"), int(cfg["n_omega"]))
    n_K = _safe_int(params.get("n_K"), int(cfg["n_K"]))
    omega_min = float(_safe_float(params.get("omega_min")) if _safe_float(params.get("omega_min")) is not None else cfg["omega_min"])
    omega_max = float(_safe_float(params.get("omega_max")) if _safe_float(params.get("omega_max")) is not None else cfg["omega_max"])
    K_min = float(_safe_float(params.get("K_min")) if _safe_float(params.get("K_min")) is not None else cfg["K_min"])
    K_max = float(_safe_float(params.get("K_max")) if _safe_float(params.get("K_max")) is not None else cfg["K_max"])
    if omega_max <= omega_min:
        raise ValueError("omega_max must be greater than omega_min")
    if K_max <= K_min:
        raise ValueError("K_max must be greater than K_min")
    iterations = _safe_int(params.get("iterations"), int(cfg["iterations"]))
    tol = _positive_float(params.get("tol"), float(cfg["tol"]))
    max_phase_samples = _safe_int(params.get("max_phase_samples"), int(cfg["max_phase_samples"]))
    lock_threshold = float(_safe_float(params.get("lock_threshold")) if _safe_float(params.get("lock_threshold")) is not None else cfg["lock_threshold"])
    lock_threshold = float(min(max(lock_threshold, 0.0), 1.0))
    dpi = _safe_int(params.get("dpi"), int(cfg["dpi"]))

    load_cfg = {"max_samples": _safe_int(params.get("max_analysis_samples"), int(cfg["max_analysis_samples"])), "decimate": 1}
    rec = _load_hfd_segment(recording_dir, params, load_cfg)
    root = Path(rec["root"])
    signal = np.asarray(rec["signal"], dtype=float)
    channels = list(rec["channels"])
    max_channels = _safe_int(params.get("max_channels"), len(channels)) if params.get("max_channels") not in (None, "") else len(channels)
    max_channels = max(1, min(max_channels, signal.shape[1]))
    signal = signal[:, :max_channels]
    channels = channels[:max_channels]
    if signal.shape[0] < 8:
        raise ValueError("Need at least 8 samples for Hilbert phase extraction.")
    signal = np.nan_to_num(signal - np.mean(signal, axis=0, keepdims=True), nan=0.0, posinf=0.0, neginf=0.0)
    fs = float(rec.get("sampling_rate_hz") or 1.0)

    out_dir = root / "advanced_methods" / "circle_map_arnold_tongues"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    analytic_signal = hilbert(signal, axis=0)
    inst_phase_rad = np.angle(analytic_signal)
    avg_phase_rad = _circle_mean_phase(inst_phase_rad, axis=1)
    avg_phase_cycles = _circle_to_cycle_phase(avg_phase_rad)
    avg_phase_cycles_sub, phase_subsample_step = _circle_uniform_subsample(avg_phase_cycles, max_phase_samples)
    if avg_phase_cycles_sub.size == 0:
        raise ValueError("No phase samples available after subsampling.")

    omegas = np.linspace(omega_min, omega_max, n_omega)
    K_values = np.linspace(K_min, K_max, n_K)
    locked = np.zeros((len(K_values), len(omegas)), dtype=float)
    for i, K in enumerate(K_values):
        locked[i, :] = _circle_fixed_point_locked_proportion_row(K, omegas, avg_phase_cycles_sub, iterations, tol)

    locked_min = float(np.nanmin(locked)); locked_max = float(np.nanmax(locked)); locked_mean = float(np.nanmean(locked))
    phase_min = float(np.nanmin(avg_phase_cycles_sub)); phase_max = float(np.nanmax(avg_phase_cycles_sub)); phase_mean = float(np.nanmean(avg_phase_cycles_sub)); phase_std = float(np.nanstd(avg_phase_cycles_sub))
    locked_binary = locked >= lock_threshold

    npz_path = out_dir / "circle_map_fixed_point_locking_results.npz"
    np.savez_compressed(
        npz_path,
        locked=locked,
        omegas=omegas,
        K_values=K_values,
        avg_phase_cycles_sub=avg_phase_cycles_sub,
        avg_phase_cycles_full=avg_phase_cycles,
        iterations=iterations,
        tol=tol,
        lock_threshold=lock_threshold,
        max_phase_samples=max_phase_samples,
        phase_subsample_step=phase_subsample_step,
        eeg_channels=np.array(channels, dtype=object),
        sampling_rate=fs,
        segment_start_sec=rec.get("segment_start_sec"),
        segment_end_sec=rec.get("segment_end_sec"),
        mode=mode,
    )

    grid_rows = []
    for i, K in enumerate(K_values):
        for j, omega in enumerate(omegas):
            grid_rows.append({"mode": mode, "Omega": float(omega), "K": float(K), "proportion_fixed_point_locked": float(locked[i, j]), "locked_binary": bool(locked_binary[i, j])})
    grid_df = pd.DataFrame(grid_rows)
    grid_csv = out_dir / "circle_map_fixed_point_locking_grid.csv"
    grid_df.to_csv(grid_csv, index=False)

    phase_df = pd.DataFrame({"sample_index": np.arange(len(avg_phase_cycles_sub)), "phase_cycles": avg_phase_cycles_sub})
    phase_csv = out_dir / "eeg_derived_mean_phase_samples.csv"
    phase_df.to_csv(phase_csv, index=False)

    plt, _ = _mfdfa_import_plotting()
    _mfdfa_apply_style(plt)
    plot_paths: list[dict[str, str]] = []

    def add_plot(title: str, fig: Any, filename: str) -> None:
        path = _mfdfa_save_fig(fig, plots_dir / filename, dpi=dpi)
        plot_paths.append({"title": title, "path": path})

    def style_cbar(cbar: Any, label: str) -> None:
        _arnold_apply_colorbar_style(cbar, label)

    def heatmap(title: str, label: str, filename: str, data: np.ndarray, *, vmin: float = 0.0, vmax: float = 1.0, overlay_boundaries: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(9, 7), facecolor=MFDFA_BLACK)
        _mfdfa_style_ax(ax, title, "Circle-map frequency parameter Ω", "Nonlinearity / coupling parameter K")
        im = ax.imshow(data, extent=[omegas.min(), omegas.max(), K_values.min(), K_values.max()], origin="lower", aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
        if overlay_boundaries:
            omega_line = np.linspace(float(omegas.min()), float(omegas.max()), 1000)
            K_boundary_m0 = 2.0 * math.pi * np.abs(omega_line - 0.0)
            K_boundary_m1 = 2.0 * math.pi * np.abs(omega_line - 1.0)
            ax.plot(omega_line, K_boundary_m0, color=MFDFA_ACCENT, linestyle="--", linewidth=1.5, alpha=0.85, label="Fixed-point boundary, m=0")
            ax.plot(omega_line, K_boundary_m1, color=MFDFA_ACCENT, linestyle=":", linewidth=1.8, alpha=0.85, label="Fixed-point boundary, m=1")
            ax.set_ylim(K_values.min(), K_values.max())
            leg = ax.legend(loc="best", frameon=True)
            for text in leg.get_texts():
                text.set_color(MFDFA_ACCENT)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        style_cbar(cbar, label)
        add_plot(title, fig, filename)

    heatmap("Circle Map Fixed-Point Locking Probability", "Proportion fixed-point locked", "01_circle_map_fixed_point_locking_heatmap.png", locked, vmin=0.0, vmax=1.0)
    heatmap("Fixed-Point Locking with Theoretical Boundaries", "Proportion fixed-point locked", "02_circle_map_fixed_point_locking_with_boundaries.png", locked, vmin=0.0, vmax=1.0, overlay_boundaries=True)

    mean_locked_over_K = np.nanmean(locked, axis=0); std_locked_over_K = np.nanstd(locked, axis=0)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=MFDFA_BLACK)
    ax.plot(omegas, mean_locked_over_K, color=MFDFA_ACCENT, linewidth=2.0)
    ax.fill_between(omegas, mean_locked_over_K - std_locked_over_K, mean_locked_over_K + std_locked_over_K, color=MFDFA_ACCENT, alpha=0.15)
    _mfdfa_style_ax(ax, "Mean Fixed-Point Locking Across K", "Circle-map frequency parameter Ω", "Mean proportion fixed-point locked")
    add_plot("Mean Fixed-Point Locking Across K", fig, "03_mean_fixed_point_locking_across_K.png")

    mean_locked_over_omega = np.nanmean(locked, axis=1); std_locked_over_omega = np.nanstd(locked, axis=1)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=MFDFA_BLACK)
    ax.plot(K_values, mean_locked_over_omega, color=MFDFA_ACCENT, linewidth=2.0)
    ax.fill_between(K_values, mean_locked_over_omega - std_locked_over_omega, mean_locked_over_omega + std_locked_over_omega, color=MFDFA_ACCENT, alpha=0.15)
    _mfdfa_style_ax(ax, "Mean Fixed-Point Locking Across Ω", "Nonlinearity / coupling parameter K", "Mean proportion fixed-point locked")
    add_plot("Mean Fixed-Point Locking Across Ω", fig, "04_mean_fixed_point_locking_across_omega.png")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=MFDFA_BLACK)
    ax.hist(avg_phase_cycles_sub, bins=min(60, max(10, int(math.sqrt(len(avg_phase_cycles_sub))))), color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.85)
    _mfdfa_style_ax(ax, "Histogram of EEG-Derived Circular Mean Phase Samples", "Circular mean phase (cycles)", "Count")
    add_plot("EEG-Derived Mean Phase Histogram", fig, "05_eeg_derived_mean_phase_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=MFDFA_BLACK)
    ax.plot(np.arange(len(avg_phase_cycles_sub)), avg_phase_cycles_sub, color=MFDFA_ACCENT, linewidth=1.2)
    _mfdfa_style_ax(ax, "EEG-Derived Circular Mean Phase Samples", "Subsample index", "Phase (cycles)")
    add_plot("EEG-Derived Circular Mean Phase Samples", fig, "06_eeg_derived_mean_phase_samples.png")

    heatmap(f"Thresholded Fixed-Point Locking Map ≥ {lock_threshold:g}", "Locked region indicator", "07_thresholded_fixed_point_locking_map.png", locked_binary.astype(float), vmin=0.0, vmax=1.0, overlay_boundaries=True)

    summary_txt = out_dir / "circle_map_fixed_point_locking_summary.txt"
    lines = [
        "Circle Map Fixed-Point Locking Using EEG-Derived Mean Phase",
        "==========================================================",
        "",
        "Interpretation:",
        "  This computes fixed-point locking probability for the circle map.",
        "  It does not compute full p:q Arnold tongues via rotation numbers.",
        "",
        f"Recording: {root}",
        f"Mode: {mode}",
        f"Description: {cfg.get('description', '')}",
        f"Sampling rate used: {fs}",
        f"Segment: [{rec.get('segment_start_sec')}, {rec.get('segment_end_sec')}] sec",
        f"Channels used: {len(channels)}",
        f"Omega range: [{float(omegas.min())}, {float(omegas.max())}] with {len(omegas)} points",
        f"K range: [{float(K_values.min())}, {float(K_values.max())}] with {len(K_values)} points",
        f"Iterations: {iterations}",
        f"Tolerance: {tol}",
        f"Lock threshold: {lock_threshold}",
        f"Original phase samples: {len(avg_phase_cycles)}",
        f"Used phase samples: {len(avg_phase_cycles_sub)}",
        f"Phase subsample step: {phase_subsample_step}",
        "",
        "Diagnostics:",
        f"locked min  = {locked_min:.10f}",
        f"locked max  = {locked_max:.10f}",
        f"locked mean = {locked_mean:.10f}",
        f"phase min   = {phase_min:.10f} cycles",
        f"phase max   = {phase_max:.10f} cycles",
        f"phase mean  = {phase_mean:.10f} cycles",
        f"phase std   = {phase_std:.10f} cycles",
        "",
        "Plot files:",
    ]
    for item in plot_paths:
        lines.append(Path(item["path"]).name)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = {"npz": str(npz_path), "grid_csv": str(grid_csv), "phase_csv": str(phase_csv), "summary_txt": str(summary_txt), "output_dir": str(out_dir), "plots_dir": str(plots_dir)}
    elapsed = time.perf_counter() - start_clock
    summary = {
        "mode": mode,
        "description": cfg.get("description"),
        "n_channels": int(len(channels)),
        "analysis_samples": int(signal.shape[0]),
        "sampling_rate_hz": _safe_float(fs),
        "omega_count": int(len(omegas)),
        "K_count": int(len(K_values)),
        "grid_points": int(len(omegas) * len(K_values)),
        "iterations": int(iterations),
        "tol": float(tol),
        "lock_threshold": float(lock_threshold),
        "phase_samples_full": int(len(avg_phase_cycles)),
        "phase_samples_used": int(len(avg_phase_cycles_sub)),
        "locked_min": _safe_float(locked_min),
        "locked_max": _safe_float(locked_max),
        "locked_mean": _safe_float(locked_mean),
        "phase_mean_cycles": _safe_float(phase_mean),
        "phase_std_cycles": _safe_float(phase_std),
        "plot_count": int(len(plot_paths)),
        "elapsed_sec": _safe_float(elapsed),
    }
    top_rows = grid_df.sort_values("proportion_fixed_point_locked", ascending=False).head(50)
    return {"circle_map_arnold_tongues": {"summary": summary, "grid_rows": _json_safe(grid_df.head(5000).to_dict("records")), "phase_rows": _json_safe(phase_df.head(5000).to_dict("records")), "top_rows": _json_safe(top_rows.to_dict("records")), "plot_paths": plot_paths, "outputs": outputs}}


def _circle_map_converged_density(recording_dir: str | Path, params: dict[str, Any]) -> dict[str, Any]:
    import time
    from scipy.signal import hilbert

    start_clock = time.perf_counter()
    mode = str(params.get("mode") or "ultra").lower()
    if mode not in CIRCLE_DENSITY_MODE_CONFIGS:
        mode = "ultra"
    cfg = dict(CIRCLE_DENSITY_MODE_CONFIGS[mode])

    Omega = float(_safe_float(params.get("Omega")) if _safe_float(params.get("Omega")) is not None else cfg["Omega"])
    K_min = float(_safe_float(params.get("K_min")) if _safe_float(params.get("K_min")) is not None else cfg["K_min"])
    K_max = float(_safe_float(params.get("K_max")) if _safe_float(params.get("K_max")) is not None else cfg["K_max"])
    if K_max <= K_min:
        raise ValueError("K_max must be greater than K_min")
    n_K = max(2, _safe_int(params.get("n_K"), int(cfg["n_K"])))
    iterations = max(1, _safe_int(params.get("iterations"), int(cfg["iterations"])))
    n_bins = max(4, _safe_int(params.get("n_bins"), int(cfg["n_bins"])))
    max_samples_per_channel = max(1, _safe_int(params.get("max_samples_per_channel"), int(cfg["max_samples_per_channel"])))
    max_analysis_samples = max(8, _safe_int(params.get("max_analysis_samples"), int(cfg["max_analysis_samples"])))
    use_bandpass = _circle_bool(params.get("use_bandpass"), bool(cfg["use_bandpass"]))
    phase_min_hz = float(_safe_float(params.get("phase_min_hz")) if _safe_float(params.get("phase_min_hz")) is not None else cfg["phase_min_hz"])
    phase_max_hz = float(_safe_float(params.get("phase_max_hz")) if _safe_float(params.get("phase_max_hz")) is not None else cfg["phase_max_hz"])
    filter_order = max(1, _safe_int(params.get("filter_order"), int(cfg["filter_order"])))
    random_seed = _safe_int(params.get("random_seed"), 42)
    dpi = _safe_int(params.get("dpi"), int(cfg["dpi"]))

    load_cfg = {"max_samples": max_analysis_samples, "decimate": 1}
    rec = _load_hfd_segment(recording_dir, params, load_cfg)
    root = Path(rec["root"])
    signal = np.asarray(rec["signal"], dtype=float)
    channels = list(rec["channels"])
    max_channels = _safe_int(params.get("max_channels"), len(channels)) if params.get("max_channels") not in (None, "") else len(channels)
    max_channels = max(1, min(max_channels, signal.shape[1]))
    signal = signal[:, :max_channels]
    channels = channels[:max_channels]
    fs = float(rec.get("sampling_rate_hz") or 1.0)
    if signal.ndim != 2 or signal.shape[1] == 0:
        raise ValueError("No channels available for Circle Map Converged Density analysis.")
    if signal.shape[0] < 16:
        raise ValueError("Need at least 16 samples for Hilbert phase extraction.")
    signal = np.nan_to_num(signal - np.mean(signal, axis=0, keepdims=True), nan=0.0, posinf=0.0, neginf=0.0)

    out_dir = root / "advanced_methods" / "circle_map_converged_density"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if use_bandpass:
        phase_signal = _circle_bandpass_filter_eeg(signal, fs, phase_min_hz, phase_max_hz, order=filter_order)
    else:
        phase_signal = signal.copy()
    phase_signal = np.nan_to_num(phase_signal, nan=0.0, posinf=0.0, neginf=0.0)

    analytic_signal = hilbert(phase_signal, axis=0)
    inst_phase_cycles = _circle_to_cycle_phase(np.angle(analytic_signal))

    K_values = np.linspace(K_min, K_max, n_K)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rng = np.random.default_rng(random_seed)

    n_channels = len(channels)
    all_hist_prob = np.zeros((n_channels, len(K_values), n_bins), dtype=float)
    initial_phase_hist_prob = np.zeros((n_channels, n_bins), dtype=float)
    channel_summary_rows: list[dict[str, Any]] = []
    per_channel_rows: list[dict[str, Any]] = []

    for ch_idx, channel_name in enumerate(channels):
        theta_init_full = inst_phase_cycles[:, ch_idx]
        counts0, _ = np.histogram(theta_init_full, bins=bin_edges, density=False)
        initial_phase_hist_prob[ch_idx] = counts0.astype(float) / (np.sum(counts0) + 1e-12)
        theta_init, phase_step = _circle_random_or_uniform_subsample(theta_init_full, max_samples_per_channel, rng=rng)
        channel_hist_prob = np.zeros((len(K_values), n_bins), dtype=float)
        channel_circ_mean = np.zeros(len(K_values), dtype=float)
        channel_resultant = np.zeros(len(K_values), dtype=float)
        channel_concentration = np.zeros(len(K_values), dtype=float)
        for i, K in enumerate(K_values):
            theta = theta_init.copy()
            for _ in range(iterations):
                theta = _circle_map(theta, Omega, float(K))
            counts, _ = np.histogram(theta, bins=bin_edges, density=False)
            prob = counts.astype(float) / (np.sum(counts) + 1e-12)
            channel_hist_prob[i, :] = prob
            cm, R = _circle_circular_mean_cycles(theta)
            channel_circ_mean[i] = cm
            channel_resultant[i] = R
            channel_concentration[i] = float(_circle_normalized_concentration_from_prob(prob))
        all_hist_prob[ch_idx] = channel_hist_prob
        mean_cm, mean_R = _circle_circular_mean_cycles(channel_circ_mean)
        row = {
            "channel": channel_name,
            "channel_index": ch_idx,
            "phase_samples_full": int(theta_init_full.size),
            "phase_samples_used": int(theta_init.size),
            "phase_subsample_step": int(phase_step),
            "mean_normalized_concentration": _safe_float(np.mean(channel_concentration)),
            "max_normalized_concentration": _safe_float(np.max(channel_concentration)),
            "mean_resultant_length": _safe_float(np.mean(channel_resultant)),
            "max_resultant_length": _safe_float(np.max(channel_resultant)),
            "mean_circular_final_state": _safe_float(mean_cm),
            "mean_final_state_resultant": _safe_float(mean_R),
        }
        channel_summary_rows.append(row)
        per_channel_rows.extend({"channel": channel_name, "K": float(K), "circular_mean_final_state": float(channel_circ_mean[i]), "resultant_length": float(channel_resultant[i]), "normalized_concentration": float(channel_concentration[i])} for i, K in enumerate(K_values))

    avg_hist_prob = np.mean(all_hist_prob, axis=0)
    agg_concentration = _circle_normalized_concentration_from_prob(avg_hist_prob)
    agg_circ_mean = np.zeros(len(K_values), dtype=float)
    agg_resultant = np.zeros(len(K_values), dtype=float)
    for i in range(len(K_values)):
        cm, R = _circle_circular_mean_cycles(bin_centers, weights=avg_hist_prob[i])
        agg_circ_mean[i] = cm
        agg_resultant[i] = R

    summary_df = pd.DataFrame(channel_summary_rows).sort_values("mean_normalized_concentration").reset_index(drop=True)
    per_channel_df = pd.DataFrame(per_channel_rows)
    density_rows = [{"K": float(K), "final_state_bin_center": float(bin_centers[b]), "avg_probability": float(avg_hist_prob[i, b])} for i, K in enumerate(K_values) for b in range(n_bins)]
    density_df = pd.DataFrame(density_rows)
    agg_df = pd.DataFrame({"K": K_values, "circular_mean_final_state": agg_circ_mean, "resultant_length": agg_resultant, "normalized_concentration": agg_concentration})
    initial_rows = [{"channel": ch, "phase_bin_center": float(bin_centers[b]), "initial_probability": float(initial_phase_hist_prob[i, b])} for i, ch in enumerate(channels) for b in range(n_bins)]
    initial_df = pd.DataFrame(initial_rows)

    npz_path = out_dir / "converged_phase_probability_results.npz"
    np.savez_compressed(
        npz_path,
        all_hist_prob=all_hist_prob,
        avg_hist_prob=avg_hist_prob,
        initial_phase_hist_prob=initial_phase_hist_prob,
        K_values=K_values,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        Omega=Omega,
        iterations=iterations,
        max_samples_per_channel=max_samples_per_channel,
        eeg_channels=np.array(channels, dtype=object),
        sampling_rate=fs,
        segment_start_sec=rec.get("segment_start_sec"),
        segment_end_sec=rec.get("segment_end_sec"),
        use_bandpass=use_bandpass,
        phase_band=np.array([phase_min_hz, phase_max_hz], dtype=float),
        filter_order=filter_order,
        agg_concentration=agg_concentration,
        agg_circ_mean=agg_circ_mean,
        agg_resultant=agg_resultant,
        mode=mode,
    )
    density_csv = out_dir / "avg_final_state_probability_grid.csv"
    summary_csv = out_dir / "converged_phase_channel_summary.csv"
    aggregate_csv = out_dir / "converged_phase_aggregate_by_K.csv"
    per_channel_csv = out_dir / "converged_phase_channel_by_K.csv"
    initial_phase_csv = out_dir / "initial_phase_distribution_by_channel.csv"
    density_df.to_csv(density_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    agg_df.to_csv(aggregate_csv, index=False)
    per_channel_df.to_csv(per_channel_csv, index=False)
    initial_df.to_csv(initial_phase_csv, index=False)

    plt, _ = _mfdfa_import_plotting()
    _mfdfa_apply_style(plt)
    plot_paths: list[dict[str, str]] = []

    def add_plot(title: str, fig: Any, filename: str) -> None:
        path = _mfdfa_save_fig(fig, plots_dir / filename, dpi=dpi)
        plot_paths.append({"title": title, "path": path})

    def style_cbar(cbar: Any, label: str) -> None:
        _arnold_apply_colorbar_style(cbar, label)

    # Plot 1: aggregate final-state probability heatmap.
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=MFDFA_BLACK)
    _mfdfa_style_ax(ax, f"Circle-Map Final-State Probability from EEG Phase Initial Conditions\nΩ = {Omega:.4f}", "Circle-map nonlinearity K", "Final state θ after iteration (cycles)")
    im = ax.imshow(avg_hist_prob.T, extent=(K_values.min(), K_values.max(), 0.0, 1.0), origin="lower", aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); style_cbar(cbar, "Mean probability per bin")
    add_plot("Aggregate Final-State Probability Heatmap", fig, "01_aggregate_final_state_probability_heatmap.png")

    # Plot 2: circular mean final state vs K.
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=MFDFA_BLACK)
    ax.plot(K_values, agg_circ_mean, color=MFDFA_ACCENT, linewidth=1.8)
    _mfdfa_style_ax(ax, "Circular Mean Final State vs K", "Circle-map nonlinearity K", "Circular mean final state θ (cycles)")
    add_plot("Circular Mean Final State vs K", fig, "02_circular_mean_final_state_vs_K.png")

    # Plot 3: normalized concentration vs K.
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=MFDFA_BLACK)
    ax.plot(K_values, agg_concentration, color=MFDFA_ACCENT, linewidth=1.8)
    _mfdfa_style_ax(ax, "Final-State Concentration vs K", "Circle-map nonlinearity K", "Normalized concentration")
    add_plot("Normalized Concentration vs K", fig, "03_normalized_concentration_vs_K.png")

    # Plot 4: circular resultant length vs K.
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=MFDFA_BLACK)
    ax.plot(K_values, agg_resultant, color=MFDFA_ACCENT, linewidth=1.8)
    _mfdfa_style_ax(ax, "Circular Resultant Length vs K", "Circle-map nonlinearity K", "Resultant length R")
    add_plot("Circular Resultant Length vs K", fig, "04_circular_resultant_length_vs_K.png")

    # Plot 5: channel-wise concentration summary.
    plot_df = summary_df.sort_values("mean_normalized_concentration", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.26 * len(plot_df))), facecolor=MFDFA_BLACK)
    ax.barh(plot_df["channel"].values, plot_df["mean_normalized_concentration"].values, color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.86)
    _mfdfa_style_ax(ax, "Mean Final-State Concentration by Channel", "Mean normalized concentration", "Channel")
    add_plot("Mean Final-State Concentration by Channel", fig, "05_mean_final_state_concentration_by_channel.png")

    # Plot 6: channel-wise resultant length summary.
    plot_df = summary_df.sort_values("mean_resultant_length", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.26 * len(plot_df))), facecolor=MFDFA_BLACK)
    ax.barh(plot_df["channel"].values, plot_df["mean_resultant_length"].values, color=MFDFA_ACCENT, edgecolor=MFDFA_ACCENT_SOFT, alpha=0.86)
    _mfdfa_style_ax(ax, "Mean Circular Resultant Length by Channel", "Mean resultant length R", "Channel")
    add_plot("Mean Circular Resultant Length by Channel", fig, "06_mean_resultant_length_by_channel.png")

    # Plot 7: initial EEG phase distribution by channel.
    fig, ax = plt.subplots(figsize=(10, max(6, 0.26 * len(channels))), facecolor=MFDFA_BLACK)
    _mfdfa_style_ax(ax, "Initial EEG Hilbert-Phase Distribution by Channel", "Initial phase θ (cycles)", "Channel", grid=False)
    im = ax.imshow(initial_phase_hist_prob, extent=(0.0, 1.0, -0.5, n_channels - 0.5), origin="lower", aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(n_channels)); ax.set_yticklabels(channels, color=MFDFA_ACCENT, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); style_cbar(cbar, "Probability per bin")
    add_plot("Initial EEG Phase Distribution by Channel", fig, "07_initial_eeg_phase_distribution_by_channel.png")

    # Plot 8: selected final-state distributions.
    selected_indices = sorted(set([0, len(K_values)//4, len(K_values)//2, (3*len(K_values))//4, len(K_values)-1]))
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=MFDFA_BLACK)
    line_styles = ["-", "--", "-.", ":", "-"]; alphas = [1.0, 0.85, 0.70, 0.55, 0.42]
    for idx, k_idx in enumerate(selected_indices):
        ax.plot(bin_centers, avg_hist_prob[k_idx], color=MFDFA_ACCENT, linestyle=line_styles[idx % len(line_styles)], alpha=alphas[idx % len(alphas)], linewidth=1.8, label=f"K = {K_values[k_idx]:.2f}")
    _mfdfa_style_ax(ax, "Selected Final-State Probability Distributions", "Final state θ (cycles)", "Probability per bin")
    leg = ax.legend(loc="best", frameon=True)
    for text in leg.get_texts(): text.set_color(MFDFA_ACCENT)
    add_plot("Selected Final-State Probability Distributions", fig, "08_selected_final_state_distributions.png")

    summary_txt = out_dir / "converged_phase_density_summary.txt"
    lines = [
        "Circle-Map Final-State Probability from EEG Hilbert-Phase Initial Conditions",
        "==========================================================================",
        "",
        "Interpretation:",
        "  EEG channel Hilbert phase is used as the initial theta distribution.",
        "  The circle map is iterated for each K at fixed Omega.",
        "  Final states are summarized as probability distributions on [0, 1).",
        "  Concentration is normalized: 0 = uniform distribution, 1 = one-bin peak.",
        "",
        f"Recording: {root}",
        f"Mode: {mode}",
        f"Description: {cfg.get('description', '')}",
        f"Sampling rate used: {fs}",
        f"Segment: [{rec.get('segment_start_sec')}, {rec.get('segment_end_sec')}] sec",
        f"Channels used: {n_channels}",
        f"Omega: {Omega}",
        f"K range: [{float(K_values.min())}, {float(K_values.max())}] with {len(K_values)} points",
        f"Iterations: {iterations}",
        f"Histogram bins: {n_bins}",
        f"Max samples per channel: {max_samples_per_channel}",
        f"Use bandpass: {use_bandpass}",
        f"Phase band: ({phase_min_hz}, {phase_max_hz}) Hz",
        f"Filter order: {filter_order}",
        "",
        "Aggregate diagnostics:",
        f"  mean normalized concentration: {np.mean(agg_concentration):.8f}",
        f"  max normalized concentration:  {np.max(agg_concentration):.8f}",
        f"  mean resultant length:         {np.mean(agg_resultant):.8f}",
        f"  max resultant length:          {np.max(agg_resultant):.8f}",
        "",
        "Per-channel summary:",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"  {row['channel']}: mean_norm_concentration={row['mean_normalized_concentration']:.6f}, max_norm_concentration={row['max_normalized_concentration']:.6f}, mean_resultant_length={row['mean_resultant_length']:.6f}, max_resultant_length={row['max_resultant_length']:.6f}")
    lines.append("")
    lines.append("Plot files:")
    for item in plot_paths:
        lines.append(Path(item["path"]).name)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = {"npz": str(npz_path), "density_csv": str(density_csv), "summary_csv": str(summary_csv), "aggregate_csv": str(aggregate_csv), "per_channel_csv": str(per_channel_csv), "initial_phase_csv": str(initial_phase_csv), "summary_txt": str(summary_txt), "output_dir": str(out_dir), "plots_dir": str(plots_dir)}
    elapsed = time.perf_counter() - start_clock
    summary = {
        "mode": mode,
        "description": cfg.get("description"),
        "n_channels": int(n_channels),
        "analysis_samples": int(signal.shape[0]),
        "sampling_rate_hz": _safe_float(fs),
        "Omega": _safe_float(Omega),
        "K_count": int(len(K_values)),
        "iterations": int(iterations),
        "n_bins": int(n_bins),
        "max_samples_per_channel": int(max_samples_per_channel),
        "use_bandpass": bool(use_bandpass),
        "phase_min_hz": _safe_float(phase_min_hz),
        "phase_max_hz": _safe_float(phase_max_hz),
        "mean_normalized_concentration": _safe_float(np.mean(agg_concentration)),
        "max_normalized_concentration": _safe_float(np.max(agg_concentration)),
        "mean_resultant_length": _safe_float(np.mean(agg_resultant)),
        "max_resultant_length": _safe_float(np.max(agg_resultant)),
        "plot_count": int(len(plot_paths)),
        "elapsed_sec": _safe_float(elapsed),
    }
    top_channels = summary_df.sort_values("mean_normalized_concentration", ascending=False).head(50)
    return {"circle_map_converged_density": {"summary": summary, "channel_rows": _json_safe(summary_df.to_dict("records")), "top_channels": _json_safe(top_channels.to_dict("records")), "density_rows": _json_safe(density_df.head(5000).to_dict("records")), "aggregate_rows": _json_safe(agg_df.head(5000).to_dict("records")), "initial_phase_rows": _json_safe(initial_df.head(5000).to_dict("records")), "plot_paths": plot_paths, "outputs": outputs}}


METHODS: dict[str, Callable[[str | Path, dict[str, Any]], dict[str, Any]]] = {
    "higuchi_fractal_dimension": _higuchi_fractal_dimension,
    "embedded_fractal_dimension": _embedded_fractal_dimension,
    "dimension_saturation_profiling": _dimension_saturation_profiling,
    "katz_fractal_dimension": _katz_fractal_dimension,
    "wavelet_hurst_exponent": _wavelet_hurst_exponent,
    "mfdfa_plot_viewer": _mfdfa_plot_viewer,
    "manual_expert_mfdfa": _manual_expert_mfdfa,
    "manual_mfdfa_spectrum": _manual_mfdfa_spectrum,
    "manual_mfdfa_shuffle_surrogate": _manual_mfdfa_shuffle_surrogate,
    "manual_mfdfa_iaaft_surrogate": _manual_mfdfa_iaaft_surrogate,
    "wavelet_leader_multifractal": _wavelet_leader_multifractal,
    "lyapunov_spectrum_custom": _lyapunov_spectrum_custom,
    "arnold_tongues_kuramoto": _arnold_tongues_kuramoto,
    "circle_map_arnold_tongues": _circle_map_arnold_tongues,
    "circle_map_converged_density": _circle_map_converged_density,
    "neuromouse_advanced_plots": _neuromouse_advanced_plots,
    "band_power_summary": _band_power_summary,
    "spike_detect": _spike_detect,
    "network_burst": _network_burst,
    "electrode_connectivity": _electrode_connectivity,
}


def list_advanced_methods() -> list[dict[str, Any]]:
    return _method_specs()


def get_method_spec(method_id: str) -> dict[str, Any]:
    for spec in _method_specs():
        if spec["id"] == method_id:
            return spec
    raise KeyError(method_id)


def run_advanced_method(method_id: str, recording_dir: str | Path, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if method_id not in METHODS:
        raise ValueError(f"Unknown advanced method: {method_id}")
    spec = get_method_spec(method_id)
    defaults = {param["name"]: param.get("default") for param in spec.get("parameters", [])}
    merged = {**defaults, **(params or {})}
    # When a preset band is chosen, keep blank min/max from overriding it.
    if method_id == "band_power_summary" and merged.get("band") not in (None, "", "custom"):
        if params is None or "min_hz" not in params:
            merged["min_hz"] = ""
        if params is None or "max_hz" not in params:
            merged["max_hz"] = ""
    result = METHODS[method_id](recording_dir, merged)
    payload = {
        "ok": True,
        "method": spec,
        "method_id": method_id,
        "recording_dir": str(Path(recording_dir).expanduser()),
        "params": merged,
        "result": result,
    }
    return _json_safe(payload)
