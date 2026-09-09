from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.common import (
    apply_aux_filter,
    as_numeric_2d,
    labels_from_any,
    make_channel_table,
    normalize_signal_orientation,
    numeric_vector,
    synthesize_or_validate_time,
)
from neuro_importer.adapters.utils import get_field, safe_float
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording


class GenericMatAdapter(BaseAdapter):
    """Heuristic adapter for simple unknown MATLAB/HDF5 dictionaries.

    This adapter is intentionally conservative. It converts only when it can find
    one clear 2D numeric signal candidate and reasonable metadata candidates.
    For ambiguous files, inspect output should guide writing a specific adapter.
    """

    name = "generic_mat"
    signal_name_hints = ("eeg", "data", "signal", "signals", "raw", "voltage", "values", "x")
    fs_name_hints = ("fs", "srate", "sfreq", "sample_rate", "sampling_rate", "fsample", "rate")
    label_name_hints = ("label", "labels", "clab", "chan", "ch_names", "channels", "electrodes", "names")
    time_name_hints = ("time", "times", "t", "timestamp", "timestamps")

    def score(self, raw: Any) -> AdapterScore:
        candidates = self._find_signal_candidates(raw)
        fs_path, fs = self._find_scalar_by_hints(raw, self.fs_name_hints)
        label_path, labels = self._find_labels(raw)
        time_path, time = self._find_time(raw)
        reasons: list[str] = []
        confidence = 0.0
        if len(candidates) == 1:
            confidence += 0.45
            reasons.append(f"one clear 2D numeric signal candidate found at {candidates[0][0]}")
        elif len(candidates) > 1:
            confidence += 0.20
            reasons.append(f"multiple 2D numeric signal candidates found: {[c[0] for c in candidates[:5]]}")
        else:
            return AdapterScore(self.name, 0.0, ["no usable 2D numeric signal candidate found"])
        if fs is not None:
            confidence += 0.20
            reasons.append(f"sampling-rate candidate found at {fs_path}")
        if labels:
            confidence += 0.20
            reasons.append(f"channel-label candidate found at {label_path}")
        if time is not None:
            confidence += 0.10
            reasons.append(f"time candidate found at {time_path}")
        # Keep generic below specific adapters so DSamp/EEGLAB/FieldTrip win.
        return AdapterScore(self.name, min(confidence, 0.78), reasons)

    def convert(
        self,
        raw: Any,
        *,
        source_path: str | Path | None = None,
        subject: int | float | str | None = None,
        session: int | float | str | None = None,
        include_aux: bool = False,
        signal_path: str | None = None,
        sampling_rate: float | None = None,
        **_: Any,
    ) -> Recording:
        q = QualityReport(adapter=self.name)
        score = self.score(raw)
        q.confidence = score.confidence
        for reason in score.reasons:
            q.add_info(reason)

        candidates = self._find_signal_candidates(raw)
        if signal_path:
            candidate = next((x for x in candidates if x[0] == signal_path), None)
            if candidate is None:
                raise ValueError(f"Requested signal_path={signal_path!r} was not found among 2D numeric candidates.")
        elif len(candidates) == 1:
            candidate = candidates[0]
        else:
            raise ValueError(
                "GenericMatAdapter found ambiguous signal candidates. "
                f"Candidates: {[c[0] for c in candidates]}. Pass signal_path=... or write a specific adapter."
            )
        sig_path, sig_raw = candidate
        labels_path, labels = self._find_labels(raw)
        time_path, time = self._find_time(raw)
        fs_path, fs_found = self._find_scalar_by_hints(raw, self.fs_name_hints)
        fs = sampling_rate if sampling_rate is not None else fs_found

        arr = as_numeric_2d(sig_raw, name=sig_path)
        signal, note = normalize_signal_orientation(arr, labels=labels, time=time, quality=q)
        q.add_assumption(note)
        channels = make_channel_table(labels, signal.shape[1], quality=q)
        signal, channels = apply_aux_filter(signal, channels, include_aux=include_aux, quality=q)
        time = synthesize_or_validate_time(time, signal.shape[0], fs, q)

        metadata = {
            "format": "Generic MATLAB/HDF5 dictionary",
            "source_path": str(source_path) if source_path is not None else None,
            "signal_path": sig_path,
            "sampling_rate_path": fs_path,
            "labels_path": labels_path,
            "time_path": time_path,
            "sampling_rate": fs,
            "subject": subject,
            "session": session,
            "raw_signal_shape": tuple(int(x) for x in arr.shape),
            "canonical_signal_shape": tuple(int(x) for x in signal.shape),
        }
        return Recording(signal=signal, sampling_rate=fs, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None)

    def _walk(self, obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).startswith("__"):
                    continue
                path = f"{prefix}.{k}" if prefix else str(k)
                out.append((path, v))
                out.extend(self._walk(v, path))
        return out

    def _find_signal_candidates(self, raw: Any) -> list[tuple[str, Any]]:
        candidates: list[tuple[str, Any, float]] = []
        for path, value in self._walk(raw):
            try:
                arr = np.asarray(value)
                arr = np.squeeze(arr)
            except Exception:
                continue
            if arr.ndim != 2 or arr.size < 20 or not np.issubdtype(arr.dtype, np.number):
                continue
            rows, cols = arr.shape
            if min(rows, cols) < 2:
                continue
            score = 0.0
            lower = path.lower()
            if any(h in lower for h in self.signal_name_hints):
                score += 1.0
            if max(rows, cols) / max(min(rows, cols), 1) >= 5:
                score += 0.5
            if max(rows, cols) >= 100:
                score += 0.3
            candidates.append((path, value, score))
        candidates.sort(key=lambda x: (x[2], np.asarray(x[1]).size), reverse=True)
        # If the best has a much higher name/shape score, keep only that one.
        if len(candidates) > 1 and candidates[0][2] >= candidates[1][2] + 0.9:
            return [(candidates[0][0], candidates[0][1])]
        return [(p, v) for p, v, _ in candidates[:8]]

    def _find_scalar_by_hints(self, raw: Any, hints: tuple[str, ...]) -> tuple[str | None, float | None]:
        for path, value in self._walk(raw):
            name = path.lower().split(".")[-1]
            if name not in hints and not any(name == h for h in hints):
                continue
            val = safe_float(value, default=None)
            if val is not None and val > 0:
                return path, val
        return None, None

    def _find_labels(self, raw: Any) -> tuple[str | None, list[str]]:
        for path, value in self._walk(raw):
            lower = path.lower().split(".")[-1]
            if not any(h in lower for h in self.label_name_hints):
                continue
            labels = labels_from_any(value)
            if len(labels) >= 2 and not all(self._looks_numeric(x) for x in labels):
                return path, labels
        return None, []

    def _find_time(self, raw: Any) -> tuple[str | None, np.ndarray | None]:
        for path, value in self._walk(raw):
            lower = path.lower().split(".")[-1]
            if not any(h == lower or h in lower for h in self.time_name_hints):
                continue
            t = numeric_vector(value)
            if t is not None and len(t) >= 2:
                return path, t
        return None, None

    def _looks_numeric(self, value: str) -> bool:
        try:
            float(value)
            return True
        except Exception:
            return False
