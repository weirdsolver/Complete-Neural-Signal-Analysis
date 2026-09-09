from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.common import (
    apply_aux_filter,
    labels_from_any,
    make_channel_table,
    normalize_signal_orientation,
    numeric_vector,
    synthesize_or_validate_time,
)
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording
from neuro_importer.detect.continuous_signal_detector import (
    LABEL_HINTS,
    TIME_HINTS,
    find_continuous_signal_candidates,
    find_vector_by_hints,
)
from neuro_importer.detect.hdf5_signature_detector import collect_attrs, get_by_slash_path
from neuro_importer.core.signal_units import infer_sampling_rate, infer_voltage_scale


class GenericHDF5ContinuousAdapter(BaseAdapter):
    """Heuristic continuous-signal adapter for unknown HDF5/NWB/MEA-like files."""

    name = "generic_hdf5_continuous"

    def score(self, raw: Any) -> AdapterScore:
        if not isinstance(raw, dict) or raw.get("__file_type__") != "hdf5":
            return AdapterScore(self.name, 0.0, ["not an HDF5 dictionary produced by HDF5Reader"])
        candidates = find_continuous_signal_candidates(raw)
        if not candidates:
            return AdapterScore(self.name, 0.0, ["no continuous 2D numeric signal candidate found"])
        best = candidates[0]
        confidence = 0.35 + min(max(best[3], 0.0), 2.5) / 5.0
        attrs = collect_attrs(raw)
        fs_key, fs = infer_sampling_rate(attrs)
        if fs is not None:
            confidence += 0.15
        reasons = [f"best continuous signal candidate: {best[0]} shape={best[2]}"]
        if len(candidates) > 1:
            reasons.append(f"additional candidates: {[c[0] for c in candidates[1:6]]}")
        if fs is not None:
            reasons.append(f"sampling-rate attribute found: {fs_key}={fs}")
        return AdapterScore(self.name, min(confidence, 0.82), reasons)

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
        apply_scaling: bool = True,
        **_: Any,
    ) -> Recording:
        q = QualityReport(adapter=self.name)
        score = self.score(raw)
        q.confidence = score.confidence
        for reason in score.reasons:
            q.add_info(reason)

        candidates = find_continuous_signal_candidates(raw)
        if signal_path:
            sig_raw = get_by_slash_path(raw, signal_path)
            sig_path = signal_path
        elif len(candidates) == 1 or (len(candidates) > 1 and candidates[0][3] > candidates[1][3] + 0.75):
            sig_path, sig_raw, _, _ = candidates[0]
        else:
            raise ValueError(
                "GenericHDF5ContinuousAdapter found ambiguous signal candidates. "
                f"Candidates: {[c[0] for c in candidates[:8]]}. Pass signal_path=... or write a specific adapter."
            )

        attrs = collect_attrs(raw)
        fs_key, fs_found = infer_sampling_rate(attrs)
        fs = sampling_rate if sampling_rate is not None else fs_found

        label_path, labels_raw = find_vector_by_hints(raw, LABEL_HINTS)
        labels = labels_from_any(labels_raw)
        time_path, time_raw = find_vector_by_hints(raw, TIME_HINTS)
        time = numeric_vector(time_raw)

        signal_raw = np.asarray(sig_raw)
        signal_raw = np.squeeze(signal_raw)
        scale_key, scale, unit = infer_voltage_scale(attrs)
        if apply_scaling and scale is not None:
            signal_raw = signal_raw.astype(float) * float(scale)
            q.add_info(f"Applied voltage scale {scale_key}={scale}; output unit={unit}.")
        signal, note = normalize_signal_orientation(signal_raw, labels=labels, time=time, quality=q)
        q.add_assumption(note)
        channels = make_channel_table(labels, signal.shape[1], quality=q)
        signal, channels = apply_aux_filter(signal, channels, include_aux=include_aux, quality=q)
        time = synthesize_or_validate_time(time, signal.shape[0], fs, q)

        metadata = {
            "format": "Generic HDF5 continuous signal",
            "source_path": str(source_path) if source_path is not None else None,
            "signal_path": sig_path,
            "sampling_rate_path": fs_key,
            "labels_path": label_path,
            "time_path": time_path,
            "sampling_rate": fs,
            "unit": unit if apply_scaling and scale is not None else attrs.get("unit"),
            "scale_attribute": scale_key,
            "subject": subject,
            "session": session,
            "hdf5_attrs": {str(k): str(v) for k, v in attrs.items()},
            "raw_signal_shape": tuple(int(x) for x in np.asarray(sig_raw).shape),
            "canonical_signal_shape": tuple(int(x) for x in signal.shape),
        }
        return Recording(signal=signal, sampling_rate=fs, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None)
