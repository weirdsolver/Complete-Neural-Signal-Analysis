from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.common import make_channel_table, normalize_signal_orientation, synthesize_or_validate_time
from neuro_importer.core.electrode_table import make_electrode_table
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording
from neuro_importer.core.signal_units import infer_sampling_rate, infer_voltage_scale
from neuro_importer.detect.hdf5_signature_detector import collect_attrs, get_by_slash_path, looks_like_cortical_labs_cl1, walk_dict


class CorticalLabsCL1Adapter(BaseAdapter):
    """Continuous raw-sample adapter for Cortical Labs CL1-style HDF5 recordings.

    This adapter reads the raw sample matrix and metadata. It ignores spikes,
    stimulation logs, and analysis tables by design.
    """

    name = "cortical_labs_cl1"

    def score(self, raw: Any) -> AdapterScore:
        if not isinstance(raw, dict) or raw.get("__file_type__") != "hdf5":
            return AdapterScore(self.name, 0.0, ["not an HDF5 dictionary produced by HDF5Reader"])
        if not looks_like_cortical_labs_cl1(raw):
            return AdapterScore(self.name, 0.0, ["does not match CL1 HDF5 sample/attribute signature"])
        sample_path = self._find_samples_path(raw)
        attrs = collect_attrs(raw)
        reasons = [f"CL1-like /samples dataset found at {sample_path}"]
        if attrs:
            reasons.append(f"metadata attributes found: {sorted(str(k) for k in attrs.keys())[:12]}")
        return AdapterScore(self.name, 0.95, reasons)

    def convert(
        self,
        raw: Any,
        *,
        source_path: str | Path | None = None,
        subject: int | float | str | None = None,
        session: int | float | str | None = None,
        sampling_rate: float | None = None,
        include_aux: bool = False,
        apply_scaling: bool = True,
        **_: Any,
    ) -> Recording:
        q = QualityReport(adapter=self.name, confidence=self.score(raw).confidence)
        sample_path = self._find_samples_path(raw)
        if sample_path is None:
            raise ValueError("Could not find CL1 samples dataset.")
        samples = np.asarray(get_by_slash_path(raw, sample_path)).squeeze()
        if samples.ndim != 2:
            raise ValueError(f"CL1 samples dataset must be 2D, got {samples.shape}.")

        attrs = collect_attrs(raw)
        fs_key, fs_found = infer_sampling_rate(attrs)
        fs = sampling_rate if sampling_rate is not None else fs_found
        scale_key, scale, unit = infer_voltage_scale(attrs)
        if apply_scaling and scale is not None:
            samples = samples.astype(float) * float(scale)
            q.add_info(f"Applied CL1 scale {scale_key}={scale}; output unit={unit}.")

        channel_count = self._safe_int(attrs.get("channel_count"), default=None)
        labels = [f"ch_{i:03d}" for i in range(channel_count or samples.shape[1])]
        signal, note = normalize_signal_orientation(samples, labels=labels, n_channels=channel_count, quality=q)
        q.add_assumption(note)
        channels = make_channel_table(labels, signal.shape[1], quality=q)
        time = synthesize_or_validate_time(None, signal.shape[0], fs, q)
        electrodes = make_electrode_table(labels, device="Cortical Labs CL1")
        q.add_info(f"Converted CL1 continuous samples from {sample_path} to shape {signal.shape}.")

        metadata = {
            "format": "Cortical Labs CL1 continuous HDF5",
            "source_path": str(source_path) if source_path is not None else None,
            "signal_path": sample_path,
            "sampling_rate": fs,
            "sampling_rate_path": fs_key,
            "channel_count": channel_count,
            "unit": unit if apply_scaling and scale is not None else attrs.get("unit"),
            "scale_attribute": scale_key,
            "subject": subject,
            "session": session,
            "hdf5_attrs": {str(k): str(v) for k, v in attrs.items()},
            "raw_signal_shape": tuple(int(x) for x in samples.shape),
            "canonical_signal_shape": tuple(int(x) for x in signal.shape),
        }
        return Recording(signal=signal, sampling_rate=fs, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None, electrodes=electrodes)

    def _find_samples_path(self, raw: dict[str, Any]) -> str | None:
        for path, value in walk_dict(raw):
            if path.lower().endswith("/samples"):
                try:
                    if np.asarray(value).ndim == 2:
                        return path
                except Exception:
                    pass
        return None

    def _safe_int(self, value: Any, default: int | None = None) -> int | None:
        try:
            return int(value)
        except Exception:
            return default
