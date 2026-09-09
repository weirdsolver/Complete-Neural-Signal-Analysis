from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.common import apply_aux_filter, labels_from_any, make_channel_table, normalize_signal_orientation, numeric_vector, synthesize_or_validate_time
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording


class NumpyArrayAdapter(BaseAdapter):
    """Adapter for .npy/.npz arrays.

    For .npy, the array itself is treated as the signal.
    For .npz, common keys are recognized: signal/data/eeg/X, time/t, labels/channels, fs/srate.
    """

    name = "numpy_array"
    signal_keys = ("signal", "data", "eeg", "raw", "x", "X")
    time_keys = ("time", "times", "t")
    label_keys = ("labels", "label", "channels", "ch_names")
    fs_keys = ("fs", "srate", "sfreq", "sampling_rate", "sample_rate")

    def score(self, raw: Any) -> AdapterScore:
        sig_key, sig = self._find_signal(raw)
        if sig is None:
            return AdapterScore(self.name, 0.0, ["no 2D NumPy signal array found"])
        reasons = [f"2D NumPy signal array found at {sig_key}"]
        confidence = 0.70
        if self._find_fs(raw)[1] is not None:
            confidence += 0.10
            reasons.append("sampling-rate key found")
        if self._find_labels(raw)[1]:
            confidence += 0.10
            reasons.append("channel-label key found")
        if self._find_time(raw)[1] is not None:
            confidence += 0.05
            reasons.append("time key found")
        return AdapterScore(self.name, min(confidence, 0.90), reasons)

    def convert(self, raw: Any, *, source_path: str | Path | None = None, subject=None, session=None, include_aux: bool = False, sampling_rate: float | None = None, **_: Any) -> Recording:
        q = QualityReport(adapter=self.name)
        score = self.score(raw)
        q.confidence = score.confidence
        for r in score.reasons:
            q.add_info(r)
        sig_key, sig_raw = self._find_signal(raw)
        if sig_raw is None:
            q.add_error("No 2D NumPy signal array found.")
            raise ValueError(q.errors[-1])
        labels_key, labels = self._find_labels(raw)
        time_key, time = self._find_time(raw)
        fs_key, fs = self._find_fs(raw)
        if sampling_rate is not None:
            fs = sampling_rate
        signal, note = normalize_signal_orientation(np.asarray(sig_raw), labels=labels, time=time, quality=q)
        q.add_assumption(note)
        channels = make_channel_table(labels, signal.shape[1], quality=q)
        signal, channels = apply_aux_filter(signal, channels, include_aux=include_aux, quality=q)
        time = synthesize_or_validate_time(time, signal.shape[0], fs, q)
        metadata = {
            "format": "NumPy",
            "source_path": str(source_path) if source_path is not None else None,
            "signal_key": sig_key,
            "labels_key": labels_key,
            "time_key": time_key,
            "sampling_rate_key": fs_key,
            "sampling_rate": fs,
            "subject": subject,
            "session": session,
            "canonical_signal_shape": tuple(int(x) for x in signal.shape),
        }
        return Recording(signal=signal, sampling_rate=fs, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None)

    def _find_signal(self, raw: Any) -> tuple[str | None, Any | None]:
        if isinstance(raw, np.ndarray):
            arr = np.squeeze(raw)
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return "array", arr
        if isinstance(raw, dict):
            for key in self.signal_keys:
                if key in raw:
                    arr = np.squeeze(np.asarray(raw[key]))
                    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                        return key, arr
            for key, value in raw.items():
                try:
                    arr = np.squeeze(np.asarray(value))
                    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                        return str(key), arr
                except Exception:
                    pass
        return None, None

    def _find_time(self, raw: Any) -> tuple[str | None, np.ndarray | None]:
        if not isinstance(raw, dict):
            return None, None
        for key in self.time_keys:
            if key in raw:
                t = numeric_vector(raw[key])
                if t is not None:
                    return key, t
        return None, None

    def _find_labels(self, raw: Any) -> tuple[str | None, list[str]]:
        if not isinstance(raw, dict):
            return None, []
        for key in self.label_keys:
            if key in raw:
                return key, labels_from_any(raw[key])
        return None, []

    def _find_fs(self, raw: Any) -> tuple[str | None, float | None]:
        if not isinstance(raw, dict):
            return None, None
        for key in self.fs_keys:
            if key in raw:
                try:
                    return key, float(np.asarray(raw[key]).reshape(-1)[0])
                except Exception:
                    pass
        return None, None
