from __future__ import annotations

import math
from typing import Any

import numpy as np


def _trapezoid_area(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz")
    return integrate(y, x, axis=axis)


def welch_psd_numpy(x: np.ndarray, fs_hz: float, nperseg: int = 500, overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency vector and PSD matrix with shape (freqs, channels)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D samples x channels array, got shape {x.shape}")
    n_samples, n_channels = x.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute PSD")
    nperseg = int(max(8, min(nperseg, n_samples)))
    step = max(1, int(nperseg * (1.0 - overlap)))
    starts = list(range(0, n_samples - nperseg + 1, step)) or [0]
    window = np.hanning(nperseg).astype(np.float32)
    win_power = float(np.sum(window * window)) or 1.0
    acc = None
    for start in starts:
        seg = x[start : start + nperseg]
        if seg.shape[0] < nperseg:
            pad = np.zeros((nperseg, n_channels), dtype=np.float32)
            pad[: seg.shape[0], :] = seg
            seg = pad
        seg = seg - np.mean(seg, axis=0, keepdims=True)
        fft = np.fft.rfft(seg * window[:, None], axis=0)
        psd = (np.abs(fft) ** 2) / (fs_hz * win_power)
        acc = psd if acc is None else acc + psd
    psd_mean = (acc / max(1, len(starts))).astype(np.float32)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs_hz).astype(np.float32)
    return freqs, psd_mean


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.zeros(psd.shape[1], dtype=np.float32)
    return _trapezoid_area(psd[mask, :], freqs[mask], axis=0).astype(np.float32)


def compute_channel_metrics(
    x: np.ndarray,
    fs_hz: float,
    channel_names: list[str],
    groups: dict[str, list[str]] | None = None,
    *,
    nperseg: int = 500,
    centroid_band: tuple[float, float] = (2.0, 45.0),
    display_band: tuple[float, float] = (1.0, 60.0),
) -> dict[str, Any]:
    freqs, psd = welch_psd_numpy(x, fs_hz=fs_hz, nperseg=nperseg)
    n_channels = psd.shape[1]
    if len(channel_names) != n_channels:
        channel_names = [f"ch_{i:03d}" for i in range(n_channels)]

    c_lo, c_hi = centroid_band
    c_mask = (freqs >= c_lo) & (freqs <= c_hi)
    if not np.any(c_mask):
        c_mask = np.ones_like(freqs, dtype=bool)
    band_freqs = freqs[c_mask]
    band_psd = psd[c_mask, :]
    denom = np.sum(band_psd, axis=0) + 1e-20
    centroid = (np.sum(band_psd * band_freqs[:, None], axis=0) / denom).astype(np.float32)
    spread = np.sqrt(np.sum(band_psd * (band_freqs[:, None] - centroid[None, :]) ** 2, axis=0) / denom).astype(np.float32)
    prob = band_psd / denom[None, :]
    entropy = (-np.sum(prob * np.log2(prob + 1e-20), axis=0) / math.log2(max(2, band_psd.shape[0]))).astype(np.float32)
    flatness = (np.exp(np.mean(np.log(band_psd + 1e-20), axis=0)) / (np.mean(band_psd, axis=0) + 1e-20)).astype(np.float32)

    powers = {
        "delta_power": _band_power(freqs, psd, 1.0, 4.0),
        "theta_power": _band_power(freqs, psd, 4.0, 8.0),
        "alpha_power": _band_power(freqs, psd, 8.0, 13.0),
        "beta_power": _band_power(freqs, psd, 13.0, 30.0),
        "gamma_power": _band_power(freqs, psd, 30.0, 60.0),
    }

    metrics_by_channel: list[dict[str, Any]] = []
    for i, name in enumerate(channel_names):
        row = {
            "index": i,
            "name": name,
            "centroid_hz": float(centroid[i]),
            "spread_hz": float(spread[i]),
            "entropy": float(entropy[i]),
            "flatness": float(flatness[i]),
        }
        row.update({k: float(v[i]) for k, v in powers.items()})
        metrics_by_channel.append(row)

    metrics_by_name = {row["name"]: row for row in metrics_by_channel}
    group_metrics: dict[str, dict[str, float]] = {}
    if groups:
        idx_by_name = {name: i for i, name in enumerate(channel_names)}
        for group_name, names in groups.items():
            idx = [idx_by_name[n] for n in names if n in idx_by_name]
            if not idx:
                continue
            group_metrics[group_name] = {
                "n_channels": float(len(idx)),
                "mean_centroid_hz": float(np.mean(centroid[idx])),
                "mean_alpha_power": float(np.mean(powers["alpha_power"][idx])),
                "mean_entropy": float(np.mean(entropy[idx])),
            }

    display_mask = (freqs >= display_band[0]) & (freqs <= display_band[1])
    return {
        "frequency_hz": freqs[display_mask].astype(np.float32),
        "psd": psd[display_mask, :].astype(np.float32),
        "metrics_by_channel": metrics_by_channel,
        "metrics_by_name": metrics_by_name,
        "groups": group_metrics,
        "summary": {
            "n_channels": n_channels,
            "mean_centroid_hz": float(np.mean(centroid)),
            "mean_alpha_power": float(np.mean(powers["alpha_power"])),
            "mean_entropy": float(np.mean(entropy)),
        },
    }
