from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.common import apply_aux_filter, labels_from_any, make_channel_table, normalize_signal_orientation, numeric_vector, synthesize_or_validate_time
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording
from neuro_importer.detect.hdf5_signature_detector import get_by_slash_path
from neuro_importer.mapping import MappingSpec
from neuro_importer.units.calibration import UnitCalibration, calibrate_recording


class MappingAdapter(BaseAdapter):
    """User-supplied mapping adapter for unknown but readable files.

    This is the escape hatch that gets the system closest to "any neural signal file":
    inspect the file, write a YAML mapping once, then convert deterministically.
    """

    name = 'manual_mapping'

    def __init__(self, spec: MappingSpec) -> None:
        self.spec = spec

    def score(self, raw: Any) -> AdapterScore:
        reasons = ['manual mapping supplied']
        if self.spec.signal_path or self.spec.signal_columns:
            return AdapterScore(self.name, 1.0, reasons)
        return AdapterScore(self.name, 0.0, ['manual mapping has no signal_path or signal_columns'])

    def convert(self, raw: Any, *, source_path: str | Path | None = None, subject=None, session=None, include_aux: bool = False, **_: Any) -> Recording:
        q = QualityReport(adapter=self.name)
        q.confidence = 1.0
        q.add_info('Using user-supplied mapping YAML.')
        spec = self.spec

        sig_raw, sig_label_names = self._extract_signal(raw, spec)
        labels = self._extract_labels(raw, spec) or sig_label_names
        fs = self._extract_sampling_rate(raw, spec)
        time = self._extract_time(raw, spec)

        arr = np.asarray(sig_raw)
        if spec.orientation == 'samples_by_channels':
            signal = np.squeeze(arr)
            note = 'orientation forced by mapping: samples × channels'
        elif spec.orientation == 'channels_by_samples':
            signal = np.squeeze(arr).T
            note = 'orientation forced by mapping: channels × samples converted to samples × channels'
        else:
            signal, note = normalize_signal_orientation(arr, labels=labels, time=time, quality=q)
        q.add_assumption(note)

        channels = make_channel_table(labels, signal.shape[1], quality=q)
        signal, channels = apply_aux_filter(signal, channels, include_aux=include_aux, quality=q)
        time = synthesize_or_validate_time(time, signal.shape[0], fs, q)

        metadata = {
            'format': 'Manual mapping',
            'source_path': str(source_path) if source_path is not None else None,
            'mapping': spec.to_dict(),
            'sampling_rate': fs,
            'subject': subject,
            'session': session,
            'canonical_signal_shape': tuple(int(x) for x in signal.shape),
        }
        metadata.update(spec.metadata or {})
        rec = Recording(signal=signal, sampling_rate=fs, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None)
        calibration = UnitCalibration.from_values(original_units=spec.original_units, target_units=spec.target_units, scale_factor=spec.scale_factor, offset=spec.offset)
        return calibrate_recording(rec, calibration)

    def _extract_signal(self, raw: Any, spec: MappingSpec) -> tuple[Any, list[str]]:
        if isinstance(raw, pd.DataFrame):
            if spec.signal_columns:
                return raw[spec.signal_columns].to_numpy(), [str(c) for c in spec.signal_columns]
            if spec.signal_path:
                cols = [c.strip() for c in spec.signal_path.split(',')]
                return raw[cols].to_numpy(), cols
        if isinstance(raw, dict):
            if spec.signal_path:
                return get_by_slash_path(raw, spec.signal_path), []
            if spec.signal_columns:
                return np.column_stack([np.asarray(raw[c]).reshape(-1) for c in spec.signal_columns]), list(spec.signal_columns)
        if isinstance(raw, np.ndarray):
            if spec.signal_path in (None, '', 'array'):
                return raw, []
        raise ValueError('Could not extract signal using mapping. Provide signal_path or signal_columns.')

    def _extract_labels(self, raw: Any, spec: MappingSpec) -> list[str]:
        if not spec.channel_names_path:
            return []
        try:
            if isinstance(raw, dict):
                return labels_from_any(get_by_slash_path(raw, spec.channel_names_path))
            if isinstance(raw, pd.DataFrame) and spec.channel_names_path in raw.columns:
                return labels_from_any(raw[spec.channel_names_path].tolist())
        except Exception:
            return []
        return []

    def _extract_sampling_rate(self, raw: Any, spec: MappingSpec) -> float | None:
        if spec.sampling_rate is not None:
            return float(spec.sampling_rate)
        if spec.sampling_rate_path and isinstance(raw, dict):
            val = get_by_slash_path(raw, spec.sampling_rate_path)
            return float(np.asarray(val).reshape(-1)[0])
        return None

    def _extract_time(self, raw: Any, spec: MappingSpec) -> np.ndarray | None:
        try:
            if spec.time_path and isinstance(raw, dict):
                return numeric_vector(get_by_slash_path(raw, spec.time_path))
            if spec.time_column and isinstance(raw, pd.DataFrame) and spec.time_column in raw.columns:
                return numeric_vector(raw[spec.time_column].to_numpy())
        except Exception:
            return None
        return None
