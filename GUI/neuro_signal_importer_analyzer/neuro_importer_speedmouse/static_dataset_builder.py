from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt, welch

try:  # optional but available in the normal scipy extra set
    from scipy.sparse.csgraph import minimum_spanning_tree
except Exception:  # pragma: no cover - fallback below handles this
    minimum_spanning_tree = None


# Keep these names for backwards-compatible imports. The user-facing product is
# NeuroMouse, but older backend code still imports build_speedmouse_dataset /
# write_speedmouse_dataset.


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_signal_path(recording_dir: Path) -> Path:
    candidates = [
        recording_dir / "signal.npy",
        recording_dir / "processed" / "signal.npy",
        recording_dir / "raw" / "signal.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No signal.npy found in {recording_dir}")


def _load_channels(recording_dir: Path, n_channels: int) -> tuple[list[str], list[str], dict[str, list[str]]]:
    candidates = [recording_dir / "channels.csv", recording_dir / "processed" / "channels.csv", recording_dir / "raw" / "channels.csv"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            name_col = next((c for c in ("name", "channel_name", "label", "channel") if c in df.columns), None)
            if name_col:
                names = [str(x) for x in df[name_col].tolist()]
                if len(names) == n_channels:
                    type_col = next((c for c in ("type", "channel_type", "kind", "modality") if c in df.columns), None)
                    types = [str(x) for x in df[type_col].tolist()] if type_col else ["NEURAL"] * n_channels
                    groups: dict[str, list[str]] = {}
                    for group_col in ("group", "region", "organoid", "well", "row", "hemisphere"):
                        if group_col in df.columns:
                            for val, name in zip(df[group_col].fillna("").astype(str), names):
                                if val and val.lower() != "nan":
                                    groups.setdefault(f"{group_col}:{val}", []).append(name)
                    return names, types, groups
        except Exception:
            pass
    names = [f"ch_{i:03d}" for i in range(n_channels)]
    return names, ["NEURAL"] * n_channels, {}


def _load_metadata(recording_dir: Path) -> dict[str, Any]:
    for p in (recording_dir / "metadata.json", recording_dir / "processed" / "metadata.json", recording_dir / "raw" / "metadata.json"):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def _sampling_rate(metadata: dict[str, Any], override: float | None = None) -> float | None:
    if override:
        return float(override)
    for key in ("sampling_rate", "sampling_rate_hz", "sample_rate_hz", "fs", "sfreq"):
        val = metadata.get(key)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
    return None


def _safe_float(x: Any) -> float | None:
    try:
        f = float(x)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return None


def _round_array(values: np.ndarray, digits: int = 6) -> list[float]:
    arr = np.asarray(values, dtype=float).ravel()
    out: list[float] = []
    for val in arr:
        out.append(round(float(val), digits) if np.isfinite(val) else 0.0)
    return out


def _round_matrix(values: np.ndarray, digits: int = 6) -> list[list[float]]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return [[round(float(v), digits) for v in row] for row in arr]


def _downsample_time_axis(x: np.ndarray, max_samples: int) -> tuple[np.ndarray, int]:
    if x.shape[0] <= max_samples:
        return x, 1
    step = int(np.ceil(x.shape[0] / max_samples))
    return x[::step], step


def _normalize_channels(samples_by_channels: np.ndarray) -> np.ndarray:
    x = np.asarray(samples_by_channels, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.nanmedian(x, axis=0, keepdims=True)
    scale = np.nanstd(x, axis=0, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return x / scale


def _nperseg_for_welch(sample_count: int, fs: float | None) -> int:
    if sample_count <= 16:
        return max(8, sample_count)
    if fs and fs > 0:
        target = int(max(64, min(4096, round(fs * 4))))
    else:
        target = max(64, sample_count // 8)
    return int(min(sample_count, max(8, target)))


def _sliding_geometry(
    signal: np.ndarray,
    fs: float | None,
    *,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    max_windows: int = 600,
) -> dict[str, Any]:
    n_samples, n_channels = signal.shape
    if not fs or fs <= 0:
        fs = 1.0
    win = max(8, int(window_sec * fs))
    step = max(1, int(step_sec * fs))
    if n_samples < win:
        win = max(8, n_samples)
        step = max(1, win)
    starts = list(range(0, max(1, n_samples - win + 1), step))
    if len(starts) > max_windows:
        stride = int(np.ceil(len(starts) / max_windows))
        starts = starts[::stride]

    times = []
    centroid = np.zeros((n_channels, len(starts)), dtype=float)
    spread = np.zeros_like(centroid)
    entropy = np.zeros_like(centroid)
    flatness = np.zeros_like(centroid)
    edge95 = np.zeros_like(centroid)
    alpha_relative_power = np.zeros_like(centroid)

    for j, start in enumerate(starts):
        seg = np.asarray(signal[start:start + win], dtype=float)
        seg = np.nan_to_num(seg - np.nanmean(seg, axis=0, keepdims=True))
        nperseg = min(seg.shape[0], max(8, int(min(fs, seg.shape[0]))))
        freqs, psd = welch(seg, fs=fs, nperseg=nperseg, axis=0)
        positive = freqs > 0
        p = psd[positive]
        f = freqs[positive]
        if p.size == 0:
            continue
        denom = np.sum(p, axis=0) + 1e-20
        c = np.sum(f[:, None] * p, axis=0) / denom
        centroid[:, j] = c
        spread[:, j] = np.sqrt(np.sum(((f[:, None] - c[None, :]) ** 2) * p, axis=0) / denom)
        prob = p / denom[None, :]
        entropy[:, j] = -np.sum(prob * np.log2(prob + 1e-20), axis=0)
        flatness[:, j] = np.exp(np.mean(np.log(p + 1e-20), axis=0)) / (np.mean(p + 1e-20, axis=0) + 1e-20)
        cumulative = np.cumsum(p, axis=0) / denom[None, :]
        for ch in range(n_channels):
            idx = int(np.searchsorted(cumulative[:, ch], 0.95))
            edge95[ch, j] = float(f[min(idx, len(f)-1)])
        alpha_mask = (f >= 8) & (f < 13)
        total_mask = (f >= 1) & (f < min(60, fs/2))
        alpha_power = np.sum(p[alpha_mask], axis=0) if np.any(alpha_mask) else np.zeros(n_channels)
        total_power = np.sum(p[total_mask], axis=0) if np.any(total_mask) else denom
        alpha_relative_power[:, j] = alpha_power / (total_power + 1e-20)
        times.append(float(start / fs))

    return {
        "time": times,
        "centroid": centroid,
        "spread": spread,
        "entropy": entropy,
        "flatness": flatness,
        "edge95": edge95,
        "alpha_relative_power": alpha_relative_power,
    }


def _centered_start(center_sec: float, win_sec: float, fs: float, sample_count: int) -> int:
    win = max(1, int(round(win_sec * fs)))
    start = int(round((center_sec - win_sec / 2) * fs))
    return max(0, min(max(0, sample_count - win), start))


def _safe_alpha_phase(signal: np.ndarray, fs: float | None) -> tuple[np.ndarray, str]:
    """Return channel-major phase values aligned with signal samples.

    Prefer 8-13 Hz alpha when the sampling rate allows it. For low-rate or
    non-EEG data, fall back to the analytic phase of the standardized signal so
    Kuramoto/PLV views still render a useful synchrony proxy instead of hiding.
    """
    x = _normalize_channels(signal)
    n_samples, n_channels = x.shape
    if n_samples < 8:
        return np.zeros((n_channels, max(1, n_samples)), dtype=float), "too_short_zero_phase"
    fs = float(fs or 1.0)
    nyq = fs / 2.0
    try:
        if nyq > 13.5:
            low, high = 8.0 / nyq, 13.0 / nyq
            sos = butter(4, [low, high], btype="band", output="sos")
            filtered = sosfiltfilt(sos, x, axis=0)
            return np.angle(hilbert(filtered, axis=0)).T, "alpha_8_13_hz"
    except Exception:
        pass
    # Fallback: analytic phase of the standardized broadband trace.
    try:
        return np.angle(hilbert(x, axis=0)).T, "broadband_analytic_phase_fallback"
    except Exception:
        return np.zeros((n_channels, n_samples), dtype=float), "phase_fallback_zero"


def _circular_mean(values: np.ndarray, axis: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(values.shape[0] if axis == 1 else values.shape[1], dtype=float)
    return np.angle(np.mean(np.exp(1j * values), axis=axis))


def _plv(phase_window: np.ndarray) -> np.ndarray:
    phase_window = np.asarray(phase_window, dtype=float)
    if phase_window.ndim != 2 or phase_window.shape[1] == 0:
        n = phase_window.shape[0] if phase_window.ndim else 1
        return np.eye(n)
    unit = np.exp(1j * phase_window)
    matrix = np.abs(np.einsum("it,jt->ij", unit, unit.conj(), optimize=True) / max(1, unit.shape[1]))
    matrix = np.clip(matrix, 0.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _build_kuramoto_and_plv(signal: np.ndarray, fs: float | None, geometry_time: list[float], channels: list[str]) -> tuple[dict[str, Any], dict[str, Any], str]:
    fs = float(fs or 1.0)
    phase, phase_method = _safe_alpha_phase(signal, fs)
    n_channels, n_phase_samples = phase.shape
    times = np.asarray(geometry_time, dtype=float)
    if times.size == 0:
        times = np.asarray([0.0], dtype=float)
    step_win = max(1, int(round(0.25 * fs)))
    channel_phases = np.zeros((n_channels, len(times)), dtype=float)
    order_r = np.zeros(len(times), dtype=float)
    mean_psi = np.zeros(len(times), dtype=float)

    for index, time_sec in enumerate(times):
        start = _centered_start(float(time_sec), 0.25, fs, n_phase_samples)
        window = phase[:, start:start + step_win]
        phases = _circular_mean(window, axis=1)
        channel_phases[:, index] = phases
        z = np.mean(np.exp(1j * phases)) if phases.size else 0.0
        order_r[index] = float(np.abs(z))
        mean_psi[index] = float(np.angle(z))

    sync_samples = min(n_phase_samples, max(1, int(round(30.0 * fs))))
    sync_phase = phase[:, :sync_samples]
    phase_synchrony: dict[str, Any] = {
        "channels": channels,
        "plv_static": _round_matrix(_plv(sync_phase)),
        "phase_method": phase_method,
    }
    win = max(1, int(round(2.0 * fs)))
    usable_times = [float(t) for t in times if t <= max(0.0, (sync_samples / fs) - 1.0)]
    if len(usable_times) > 120:
        stride = int(np.ceil(len(usable_times) / 120))
        usable_times = usable_times[::stride]
    sliding = []
    for time_sec in usable_times:
        start = _centered_start(time_sec, 2.0, fs, sync_samples)
        sliding.append(_round_matrix(_plv(sync_phase[:, start:start + win])))
    if sliding:
        phase_synchrony["plv_sliding_time"] = [round(float(t), 6) for t in usable_times]
        phase_synchrony["plv_sliding"] = sliding

    kuramoto = {
        "time": _round_array(times),
        "order_parameter_r": _round_array(order_r),
        "mean_phase_psi": _round_array(mean_psi),
        "channel_phases": _round_matrix(channel_phases),
        "channels": channels,
        "phase_method": phase_method,
    }
    return kuramoto, phase_synchrony, phase_method


def _higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < max(16, kmax * 2):
        return 0.0
    lengths = []
    ks = np.arange(1, kmax + 1, dtype=float)
    for k in range(1, kmax + 1):
        lk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            diffs = np.abs(np.diff(x[idx])).sum()
            norm = (n - 1) / ((idx.size - 1) * k)
            lk.append((diffs * norm) / k)
        lengths.append(float(np.mean(lk)) if lk else np.nan)
    lengths = np.asarray(lengths, dtype=float)
    mask = np.isfinite(lengths) & (lengths > 0)
    if mask.sum() < 2:
        return 0.0
    slope = np.polyfit(np.log(ks[mask]), np.log(lengths[mask]), 1)[0]
    return float(max(0.0, -slope))


def _add_higuchi_fd(signal: np.ndarray, fs: float | None, geometry: dict[str, Any], *, max_eval_windows: int = 120) -> None:
    fs = float(fs or 1.0)
    times = np.asarray(geometry.get("time") or [0.0], dtype=float)
    n_samples, n_channels = signal.shape
    x = _normalize_channels(signal)
    win = max(16, int(round(2.0 * fs)))
    eval_idx = np.arange(len(times))
    if len(eval_idx) > max_eval_windows:
        eval_idx = np.unique(np.linspace(0, len(times) - 1, max_eval_windows).round().astype(int))
    h_eval = np.zeros((n_channels, len(eval_idx)), dtype=float)
    for out_i, time_index in enumerate(eval_idx):
        start = _centered_start(float(times[time_index]), 2.0, fs, n_samples)
        seg = x[start:start + win]
        if seg.shape[0] > 512:
            dec = int(np.ceil(seg.shape[0] / 512))
            seg = seg[::dec]
        for ch in range(n_channels):
            h_eval[ch, out_i] = _higuchi_fd(seg[:, ch], kmax=8)
    if len(eval_idx) == len(times):
        hfd = h_eval
    else:
        hfd = np.zeros((n_channels, len(times)), dtype=float)
        for ch in range(n_channels):
            hfd[ch] = np.interp(np.arange(len(times)), eval_idx, h_eval[ch])
    geometry["higuchi_fd"] = hfd


def _similarity_from_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    if n == 0:
        return np.empty((0, 0))
    finite = np.isfinite(values)
    if not finite.any():
        return np.eye(n)
    fill = float(np.nanmedian(values[finite]))
    values = np.where(np.isfinite(values), values, fill)
    spread = float(np.nanmax(values) - np.nanmin(values)) or 1.0
    sim = 1.0 - np.abs(values[:, None] - values[None, :]) / spread
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def _safe_corrcoef(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape[1] == 1:
        return np.ones((1, 1), dtype=float)
    sample_limit = min(x.shape[0], 120_000)
    if x.shape[0] > sample_limit:
        stride = int(np.ceil(x.shape[0] / sample_limit))
        x = x[::stride]
    x = _normalize_channels(x)
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _build_channel_network(signal: np.ndarray, geometry: dict[str, Any], channels: list[str]) -> dict[str, Any]:
    n_channels = len(channels)
    corr = _safe_corrcoef(signal)
    centroid_mean = np.nanmean(np.asarray(geometry["centroid"], dtype=float), axis=1)
    alpha_mean = np.nanmean(np.asarray(geometry["alpha_relative_power"], dtype=float), axis=1)
    entropy_mean = np.nanmean(np.asarray(geometry["entropy"], dtype=float), axis=1)
    centroid_sim = _similarity_from_values(centroid_mean)
    alpha_sim = _similarity_from_values(alpha_mean)
    entropy_sim = _similarity_from_values(entropy_mean)
    composite = (np.abs(corr) + centroid_sim + alpha_sim + entropy_sim) / 4.0
    np.fill_diagonal(composite, 1.0)
    # Adaptive threshold keeps the graph informative for non-demo datasets.
    tri = composite[np.triu_indices(n_channels, k=1)] if n_channels > 1 else np.asarray([1.0])
    threshold = float(np.nanpercentile(tri, 85)) if tri.size else 0.7
    threshold = float(np.clip(threshold, 0.45, 0.9))
    return {
        "channels": channels,
        "composite_correlation": _round_matrix(composite),
        "per_metric": {
            "signal_correlation": _round_matrix(corr),
            "spectral_centroid_similarity": _round_matrix(centroid_sim),
            "alpha_relative_power_similarity": _round_matrix(alpha_sim),
            "spectral_entropy_similarity": _round_matrix(entropy_sim),
        },
        "threshold_strong": round(threshold, 6),
        "threshold_moderate": round(max(0.3, threshold * 0.72), 6),
    }


def _channel_role_indices(channels: list[str]) -> tuple[list[int], list[int]]:
    frontal_prefixes = ("FP", "AF", "F")
    posterior_prefixes = ("P", "PO", "O")
    frontal: list[int] = []
    posterior: list[int] = []
    for i, raw in enumerate(channels):
        name = str(raw).upper().replace(" ", "")
        if name.startswith(frontal_prefixes):
            frontal.append(i)
        if name.startswith(posterior_prefixes):
            posterior.append(i)
    n = len(channels)
    if not frontal:
        frontal = list(range(0, max(1, n // 3)))
    if not posterior:
        posterior = list(range(max(0, n - max(1, n // 3)), n))
    return frontal, posterior


def _build_polar_chronomap(geometry: dict[str, Any], channels: list[str]) -> dict[str, Any]:
    alpha = np.asarray(geometry["alpha_relative_power"], dtype=float)
    frontal, posterior = _channel_role_indices(channels)
    frontal_alpha = np.nanmean(alpha[frontal, :], axis=0) if frontal else np.nanmean(alpha, axis=0)
    posterior_alpha = np.nanmean(alpha[posterior, :], axis=0) if posterior else np.nanmean(alpha, axis=0)
    balance = posterior_alpha - frontal_alpha
    return {
        "time": [round(float(t), 6) for t in geometry.get("time", [])],
        "posterior_alpha": _round_array(posterior_alpha),
        "frontal_alpha": _round_array(frontal_alpha),
        "balance": _round_array(balance),
        "posterior_channels": [channels[i] for i in posterior],
        "frontal_channels": [channels[i] for i in frontal],
    }


def _trapezoid_area(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz")
    return integrate(y, x, axis=axis)


def _add_area_normalized_psd(geometry: dict[str, Any], freqs: np.ndarray, psd_ch_major: np.ndarray) -> None:
    p = np.asarray(psd_ch_major, dtype=float)
    denom = _trapezoid_area(np.maximum(p, 0.0), freqs, axis=1)
    denom[~np.isfinite(denom) | (denom <= 1e-20)] = 1.0
    normalized = p / denom[:, None]
    geometry["area_normalized_psd"] = {
        "frequencies": _round_array(freqs),
        "psd": _round_matrix(normalized),
    }


def _approx_lyapunov(series: np.ndarray) -> float:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 32:
        return 0.0
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    dx = np.diff(x)
    val = np.log((np.std(dx) + 1e-12) / (np.std(x) + 1e-12))
    return float(np.clip(val, -5.0, 5.0))


def _build_tda(geometry: dict[str, Any], channels: list[str]) -> dict[str, Any]:
    keys = ["centroid", "spread", "entropy", "flatness", "edge95", "alpha_relative_power"]
    features = []
    for key in keys:
        arr = np.asarray(geometry.get(key), dtype=float)
        features.append(np.nanmean(arr, axis=1))
    points = np.vstack(features).T
    points = np.nan_to_num(points, nan=np.nanmedian(points) if np.isfinite(points).any() else 0.0)
    mean = points.mean(axis=0, keepdims=True)
    std = points.std(axis=0, keepdims=True)
    std[std <= 1e-12] = 1.0
    z = (points - mean) / std
    n = z.shape[0]
    if n < 2:
        return {"status": "computed", "h0": [], "h1": [], "point_cloud": _round_matrix(z), "channels": channels, "features": keys}
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    if minimum_spanning_tree is not None:
        mst = minimum_spanning_tree(d).toarray()
        edges = mst[mst > 0]
    else:
        # Greedy nearest-neighbor fallback: enough for a barcode/scatter view.
        edges = np.sort(d[np.triu_indices(n, k=1)])[: max(1, n - 1)]
    h0 = np.column_stack([np.zeros(edges.size), np.sort(edges)]) if edges.size else np.empty((0, 2))
    h1 = np.empty((0, 2))
    if n >= 4:
        tri = np.sort(d[np.triu_indices(n, k=1)])
        if tri.size >= 4:
            q = np.quantile(tri, [0.45, 0.62, 0.78, 0.9])
            h1 = np.asarray([
                [q[0], q[2]],
                [q[1], q[3]],
            ], dtype=float)
            h1 = h1[h1[:, 1] > h1[:, 0]]
    return {
        "status": "computed",
        "h0": _round_matrix(h0),
        "h1": _round_matrix(h1),
        "point_cloud": _round_matrix(z),
        "channels": channels,
        "features": keys,
        "method": "lightweight_feature_distance_persistence",
    }


def build_speedmouse_dataset(
    recording_dir: str | Path,
    *,
    dataset_id: str | None = None,
    sampling_rate: float | None = None,
    max_analysis_samples: int = 240_000,
    max_windows: int = 600,
) -> dict[str, Any]:
    """Build a full NeuroMouse-compatible static data.json object.

    The output is variable-electrode safe. Matrix convention is channel-major:
    - welch_psd.psd: channels x frequencies
    - geometry metrics: channels x time_windows

    v0.10.3 restores the advanced NeuroMouse analysis objects that the original
    workbench plotted: polar chronomap, Kuramoto phase animation, channel
    network, PLV synchrony, Higuchi FD, area-normalized PSD, and TDA.
    """
    recording_dir = Path(recording_dir).expanduser().resolve()
    signal_path = _load_signal_path(recording_dir)
    signal = np.load(signal_path, mmap_mode="r")
    if signal.ndim != 2:
        raise ValueError(f"Expected signal.npy samples x channels, got {signal.shape}")
    n_samples, n_channels = signal.shape
    metadata = _load_metadata(signal_path.parent) or _load_metadata(recording_dir)
    fs = _sampling_rate(metadata, sampling_rate) or 1.0
    channel_names, channel_types, groups = _load_channels(signal_path.parent, n_channels)
    if channel_names == [f"ch_{i:03d}" for i in range(n_channels)]:
        channel_names, channel_types, groups = _load_channels(recording_dir, n_channels)

    analysis_signal, ds_step = _downsample_time_axis(np.asarray(signal), max_analysis_samples)
    fs_analysis = fs / ds_step if fs and fs > 0 else fs
    analysis_signal = np.asarray(analysis_signal, dtype=np.float32)
    analysis_signal = np.nan_to_num(analysis_signal)

    nperseg = _nperseg_for_welch(analysis_signal.shape[0], fs_analysis)
    freqs, psd = welch(analysis_signal, fs=fs_analysis, nperseg=nperseg, axis=0)
    psd_ch_major = np.nan_to_num(psd.T, nan=0.0, posinf=0.0, neginf=0.0)

    geometry = _sliding_geometry(analysis_signal, fs_analysis, max_windows=max_windows)
    _add_area_normalized_psd(geometry, freqs, psd_ch_major)
    _add_higuchi_fd(analysis_signal, fs_analysis, geometry)

    kuramoto, phase_synchrony, phase_method = _build_kuramoto_and_plv(analysis_signal, fs_analysis, geometry["time"], channel_names)
    channel_network = _build_channel_network(analysis_signal, geometry, channel_names)
    polar_chronomap = _build_polar_chronomap(geometry, channel_names)
    tda = _build_tda(geometry, channel_names)

    channel_summary = []
    for i, name in enumerate(channel_names):
        ch = np.asarray(analysis_signal[:, i], dtype=float)
        psd_i = psd_ch_major[i]
        alpha_mask = (freqs >= 8) & (freqs < 13)
        peak_idx = int(np.nanargmax(psd_i)) if psd_i.size else 0
        alpha_rel = _safe_float(np.nanmean(geometry["alpha_relative_power"][i]))
        centroid_hz = _safe_float(np.nanmean(geometry["centroid"][i]))
        spread_hz = _safe_float(np.nanmean(geometry["spread"][i]))
        entropy_val = _safe_float(np.nanmean(geometry["entropy"][i]))
        flatness_val = _safe_float(np.nanmean(geometry["flatness"][i]))
        edge95_hz = _safe_float(np.nanmean(geometry["edge95"][i]))
        hfd_val = _safe_float(np.nanmean(geometry.get("higuchi_fd", np.zeros((n_channels, 1)))[i]))
        channel_summary.append({
            "channel_index": i,
            "channel": name,
            "name": name,
            "type": channel_types[i] if i < len(channel_types) else "NEURAL",
            "region": "generic",
            "hemisphere": "",
            "mean": _safe_float(np.nanmean(ch)),
            "std": _safe_float(np.nanstd(ch)),
            "rms": _safe_float(np.sqrt(np.nanmean(ch * ch))),
            "variance": _safe_float(np.nanvar(ch)),
            "peak_frequency_hz": _safe_float(freqs[peak_idx]) if psd_i.size else None,
            "alpha_peak_frequency_hz": _safe_float(freqs[peak_idx]) if psd_i.size and np.any(alpha_mask) else None,
            "alpha_power": _safe_float(np.nanmean(psd_i[alpha_mask])) if np.any(alpha_mask) else None,
            "alpha_relative_power": alpha_rel,
            "spectral_centroid_hz": centroid_hz,
            "spectral_spread_hz": spread_hz,
            "spectral_entropy": entropy_val,
            "spectral_flatness": flatness_val,
            "edge95_hz": edge95_hz,
            "higuchi_fd": hfd_val,
            "lyapunov_exponent": _safe_float(_approx_lyapunov(ch[: min(ch.size, 50_000)])),
            "sliding_alpha_relative_mean": alpha_rel,
            "mean_centroid_hz": centroid_hz,
            "mean_alpha_relative_power": alpha_rel,
            "has_clear_alpha_peak": bool(alpha_rel is not None and alpha_rel > 0.05),
            "variability": {
                "alpha_range": _safe_float(np.nanmax(geometry["alpha_relative_power"][i]) - np.nanmin(geometry["alpha_relative_power"][i])),
            },
        })

    manifest = {
        "n_channels": n_channels,
        "channel_names": channel_names,
        "channel_types": channel_types,
        "channel_groups": groups,
        "namespace": metadata.get("channel_namespace") or metadata.get("layout") or "variable_neural",
        "modality": metadata.get("modality") or metadata.get("signal_type") or "continuous_neural",
    }
    data = {
        "meta": {
            "schema": "neuromouse.data.v1.variable.full_analysis",
            "dataset_id": dataset_id or recording_dir.name,
            "source_recording_dir": str(recording_dir),
            "source_signal_path": str(signal_path),
            "channels": channel_names,
            "n_channels": n_channels,
            "channel_types": channel_types,
            "channel_manifest": manifest,
            "sampling_rate_hz": fs,
            "sampling_rate_analysis_hz": fs_analysis,
            "duration_sec": float(n_samples / fs) if fs else None,
            "n_samples": int(n_samples),
            "analysis_samples": int(analysis_signal.shape[0]),
            "analysis_downsample_step": int(ds_step),
            "segment_duration_sec": float(analysis_signal.shape[0] / fs_analysis) if fs_analysis else None,
            "welch_window_sec": float(nperseg / fs_analysis) if fs_analysis else None,
            "welch_overlap_fraction": 0.5,
            "sliding_window_sec": 2.0,
            "sliding_step_sec": 0.5,
            "continuous_signal_only": True,
            "generated_by": "Neuro Signal Importer NeuroMouse adapter",
            "analysis_by": "Neuro Signal Importer + NeuroMouse integration",
            "advanced_plots_generated": True,
            "advanced_analysis_keys": [
                "polar_chronomap",
                "kuramoto",
                "channel_network",
                "phase_synchrony",
                "tda",
                "geometry.higuchi_fd",
                "geometry.area_normalized_psd",
            ],
            "phase_method": phase_method,
        },
        "welch_psd": {
            "frequencies": _round_array(freqs),
            "psd": _round_matrix(psd_ch_major),
        },
        "centroid": {
            "time_relative": geometry["time"],
            "values": _round_matrix(geometry["centroid"]),
        },
        "geometry": {
            "time": geometry["time"],
            "centroid": _round_matrix(geometry["centroid"]),
            "spread": _round_matrix(geometry["spread"]),
            "entropy": _round_matrix(geometry["entropy"]),
            "flatness": _round_matrix(geometry["flatness"]),
            "edge95": _round_matrix(geometry["edge95"]),
            "alpha_relative_power": _round_matrix(geometry["alpha_relative_power"]),
            "higuchi_fd": _round_matrix(geometry["higuchi_fd"]),
            "area_normalized_psd": geometry["area_normalized_psd"],
        },
        "channel_summary": channel_summary,
        "polar_chronomap": polar_chronomap,
        "channel_network": channel_network,
        "kuramoto": kuramoto,
        "phase_synchrony": phase_synchrony,
        "tda": tda,
        "phase3_meta": {
            "source_file": signal_path.name,
            "sampling_rate_hz": _safe_float(fs),
            "analysis_sample_rate_hz": _safe_float(fs_analysis),
            "analysis_segment_sec": _safe_float(analysis_signal.shape[0] / fs_analysis) if fs_analysis else None,
            "higuchi_kmax": 8,
            "phase_method": phase_method,
            "notes": [
                "Generated from the uploaded/converted canonical signal.npy, not the bundled demo data.",
                "Advanced NeuroMouse objects are generated for the original workbench plots.",
                "TDA uses a lightweight feature-distance persistence approximation when optional ripser is unavailable.",
            ],
        },
    }
    return _json_safe(data)


def write_speedmouse_dataset(
    recording_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str | None = None,
    sampling_rate: float | None = None,
    max_analysis_samples: int = 240_000,
    max_windows: int = 600,
) -> dict[str, str]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_speedmouse_dataset(
        recording_dir,
        dataset_id=dataset_id,
        sampling_rate=sampling_rate,
        max_analysis_samples=max_analysis_samples,
        max_windows=max_windows,
    )
    data_path = output_dir / "data.json"
    manifest_path = output_dir / "neuromouse_manifest.json"
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    manifest = {
        "data_json": str(data_path),
        "dataset_id": data["meta"].get("dataset_id"),
        "n_channels": data["meta"].get("n_channels"),
        "sampling_rate_hz": data["meta"].get("sampling_rate_hz"),
        "source_recording_dir": data["meta"].get("source_recording_dir"),
        "advanced_plots_generated": True,
        "advanced_analysis_keys": data["meta"].get("advanced_analysis_keys", []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"data_json": str(data_path), "manifest": str(manifest_path)}
