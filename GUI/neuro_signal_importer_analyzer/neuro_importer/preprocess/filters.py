from __future__ import annotations

import numpy as np


def demean(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    return x - np.nanmean(x, axis=0, keepdims=True)


def detrend(signal: np.ndarray) -> np.ndarray:
    try:
        from scipy import signal as sp_signal  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install scipy to use detrend: pip install scipy") from exc
    return sp_signal.detrend(np.asarray(signal, dtype=float), axis=0, type="linear")


def notch_filter(signal: np.ndarray, sampling_rate: float, notch_hz: float, q: float = 30.0) -> np.ndarray:
    try:
        from scipy import signal as sp_signal  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install scipy to use notch filtering: pip install scipy") from exc
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive for notch filtering")
    b, a = sp_signal.iirnotch(w0=float(notch_hz), Q=float(q), fs=float(sampling_rate))
    return sp_signal.filtfilt(b, a, np.asarray(signal, dtype=float), axis=0)


def bandpass_filter(signal: np.ndarray, sampling_rate: float, low_hz: float | None, high_hz: float | None, order: int = 4) -> np.ndarray:
    try:
        from scipy import signal as sp_signal  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install scipy to use bandpass filtering: pip install scipy") from exc
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive for filtering")
    nyq = sampling_rate / 2.0
    low = None if low_hz is None else float(low_hz) / nyq
    high = None if high_hz is None else float(high_hz) / nyq
    if low is not None and high is not None:
        btype = "bandpass"
        wn = [max(low, 1e-9), min(high, 0.999999)]
    elif low is not None:
        btype = "highpass"
        wn = max(low, 1e-9)
    elif high is not None:
        btype = "lowpass"
        wn = min(high, 0.999999)
    else:
        return np.asarray(signal)
    b, a = sp_signal.butter(order, wn, btype=btype)
    return sp_signal.filtfilt(b, a, np.asarray(signal, dtype=float), axis=0)


def downsample(signal: np.ndarray, sampling_rate: float, target_rate: float) -> tuple[np.ndarray, float, int]:
    if target_rate is None or target_rate <= 0 or sampling_rate is None or sampling_rate <= 0:
        return np.asarray(signal), sampling_rate, 1
    if target_rate >= sampling_rate:
        return np.asarray(signal), sampling_rate, 1
    factor = int(round(float(sampling_rate) / float(target_rate)))
    factor = max(factor, 1)
    new_rate = float(sampling_rate) / factor
    return np.asarray(signal)[::factor, :], new_rate, factor
