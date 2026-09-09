from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neuro_importer.core.recording import Recording

_CANON = {
    'uv': 'microvolts', 'µv': 'microvolts', 'microvolt': 'microvolts', 'microvolts': 'microvolts',
    'mv': 'millivolts', 'millivolt': 'millivolts', 'millivolts': 'millivolts',
    'v': 'volts', 'volt': 'volts', 'volts': 'volts',
    'adc': 'adc_counts', 'counts': 'adc_counts', 'adc_counts': 'adc_counts',
    'a.u.': 'arbitrary', 'au': 'arbitrary', 'arbitrary': 'arbitrary', 'arbitrary_units': 'arbitrary',
}
_FACTORS_TO_VOLTS = {'microvolts': 1e-6, 'millivolts': 1e-3, 'volts': 1.0}


def canonical_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return _CANON.get(str(unit).strip().lower(), str(unit).strip().lower())


@dataclass
class UnitCalibration:
    original_units: str | None = None
    target_units: str | None = None
    scale_factor: float | None = None
    offset: float | None = None

    @classmethod
    def from_values(cls, *, original_units: str | None = None, target_units: str | None = None,
                    scale_factor: float | None = None, offset: float | None = None) -> 'UnitCalibration':
        return cls(original_units=canonical_unit(original_units), target_units=canonical_unit(target_units),
                   scale_factor=scale_factor, offset=offset)

    def describe(self) -> dict[str, Any]:
        return {
            'original_units': self.original_units,
            'target_units': self.target_units,
            'scale_factor': self.scale_factor,
            'offset': self.offset,
        }

    def apply_array(self, signal: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        out = np.asarray(signal)
        info = self.describe()
        changed = False
        if self.scale_factor is not None or self.offset is not None:
            out = out.astype(float, copy=False)
            if self.scale_factor is not None:
                out = out * float(self.scale_factor)
                changed = True
            if self.offset is not None:
                out = out + float(self.offset)
                changed = True
        src = canonical_unit(self.original_units)
        dst = canonical_unit(self.target_units)
        if src and dst and src != dst:
            if src in _FACTORS_TO_VOLTS and dst in _FACTORS_TO_VOLTS:
                factor = _FACTORS_TO_VOLTS[src] / _FACTORS_TO_VOLTS[dst]
                out = out.astype(float, copy=False) * factor
                info['unit_conversion_factor'] = factor
                changed = True
            elif src == 'adc_counts' and self.scale_factor is None:
                info['warning'] = 'ADC counts cannot be converted without a scale_factor.'
            elif src == 'arbitrary' and self.scale_factor is None:
                info['warning'] = 'Arbitrary units cannot be converted without a scale_factor.'
        info['applied'] = changed
        info['final_units'] = dst or src
        return out, info


def calibrate_recording(recording: Recording, calibration: UnitCalibration | None) -> Recording:
    if calibration is None:
        return recording
    signal, info = calibration.apply_array(recording.signal)
    recording.signal = signal
    recording.metadata.setdefault('unit_calibration', info)
    if info.get('final_units'):
        recording.metadata['signal_units'] = info['final_units']
    if hasattr(recording, 'quality') and recording.quality is not None:
        if info.get('applied'):
            recording.quality.add_info(f"Applied unit calibration: {info}")
        elif info.get('warning'):
            recording.quality.add_warning(str(info['warning']))
    return recording
