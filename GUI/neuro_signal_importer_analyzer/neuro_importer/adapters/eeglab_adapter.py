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
from neuro_importer.adapters.utils import as_scalar, get_field, safe_float
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording


class EEGLABAdapter(BaseAdapter):
    """Adapter for EEGLAB-style MATLAB structures.

    Expected raw structure after MatReader:
        raw["EEG"].data       -> signal, usually channels × samples or channels × samples × trials
        raw["EEG"].srate      -> sampling rate
        raw["EEG"].times      -> optional time in ms
        raw["EEG"].chanlocs   -> channel metadata; labels often under .labels

    This adapter intentionally exports neural signal only. It ignores EEGLAB
    events/epochs/stimulation until that layer is added explicitly.
    """

    name = "eeglab_mat"

    def score(self, raw: Any) -> AdapterScore:
        eeg = self._extract_eeg(raw)
        if eeg is None:
            return AdapterScore(self.name, 0.0, ["no top-level EEG object found"])

        reasons: list[str] = ["top-level EEG object found"]
        confidence = 0.30
        if get_field(eeg, "data") is not None:
            confidence += 0.30
            reasons.append("EEG.data found")
        if get_field(eeg, "srate") is not None:
            confidence += 0.15
            reasons.append("EEG.srate found")
        if get_field(eeg, "chanlocs") is not None:
            confidence += 0.15
            reasons.append("EEG.chanlocs found")
        if get_field(eeg, "times") is not None:
            confidence += 0.05
            reasons.append("EEG.times found")
        if get_field(eeg, "event") is not None:
            confidence += 0.02
            reasons.append("EEG.event found but ignored for neural-signal-only conversion")
        return AdapterScore(self.name, min(confidence, 1.0), reasons)

    def convert(
        self,
        raw: Any,
        *,
        source_path: str | Path | None = None,
        subject: int | float | str | None = None,
        session: int | float | str | None = None,
        include_aux: bool = False,
        **_: Any,
    ) -> Recording:
        q = QualityReport(adapter=self.name)
        score = self.score(raw)
        q.confidence = score.confidence
        for reason in score.reasons:
            q.add_info(reason)

        eeg = self._extract_eeg(raw)
        if eeg is None:
            q.add_error("No EEGLAB EEG object was found.")
            raise ValueError(q.errors[-1])

        data = get_field(eeg, "data")
        if data is None:
            q.add_error("EEG.data is missing.")
            raise ValueError(q.errors[-1])
        arr = np.asarray(data)
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            # EEGLAB commonly stores channels × points × trials. Concatenate trials along time.
            q.add_warning("EEG.data is 3D; assuming channels × samples × trials and concatenating trials along time.")
            arr = np.concatenate([arr[:, :, i] for i in range(arr.shape[2])], axis=1)
        if arr.ndim != 2:
            raise ValueError(f"EEG.data must resolve to 2D or 3D. Got shape {arr.shape}.")

        fs = safe_float(get_field(eeg, "srate"), default=None)
        labels = self._extract_labels(eeg)
        time = self._extract_time(eeg)
        nbchan = self._safe_int(get_field(eeg, "nbchan"))
        pnts = self._safe_int(get_field(eeg, "pnts"))
        trials = self._safe_int(get_field(eeg, "trials")) or 1
        n_samples_declared = pnts * trials if pnts and trials else pnts

        signal, note = normalize_signal_orientation(arr, labels=labels, time=time, n_channels=nbchan, n_samples=n_samples_declared, quality=q)
        q.add_assumption(note)
        if time is not None and len(time) != signal.shape[0] and trials and trials > 1 and pnts and len(time) == pnts:
            # Repeat epoch-relative time with offsets so dataframe length matches signal.
            if fs and fs > 0:
                epoch_len_s = pnts / fs
                time = np.concatenate([time + i * epoch_len_s for i in range(trials)])
                q.add_warning("Repeated EEGLAB epoch time vector with offsets after concatenating trials.")

        channels = make_channel_table(labels, signal.shape[1], quality=q)
        signal, channels = apply_aux_filter(signal, channels, include_aux=include_aux, quality=q)
        time = synthesize_or_validate_time(time, signal.shape[0], fs, q)

        if get_field(eeg, "event") is not None:
            q.add_info("EEG.event was found but intentionally ignored in this neural-signal-only version.")

        metadata = {
            "format": "EEGLAB MATLAB",
            "source_path": str(source_path) if source_path is not None else None,
            "sampling_rate": fs,
            "subject": subject or as_scalar(get_field(eeg, "subject"), default=None),
            "session": session or as_scalar(get_field(eeg, "session"), default=None),
            "nbchan_declared": nbchan,
            "pnts_declared": pnts,
            "trials_declared": trials,
            "raw_signal_shape": tuple(int(x) for x in np.asarray(data).shape),
            "canonical_signal_shape": tuple(int(x) for x in signal.shape),
            "events_ignored": get_field(eeg, "event") is not None,
        }
        return Recording(signal=signal, sampling_rate=fs, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None)

    def _extract_eeg(self, raw: Any) -> Any | None:
        if isinstance(raw, dict) and "EEG" in raw:
            return raw["EEG"]
        return get_field(raw, "EEG", default=None)

    def _extract_labels(self, eeg: Any) -> list[str]:
        chanlocs = get_field(eeg, "chanlocs")
        if chanlocs is None:
            return []
        labels: list[str] = []
        if isinstance(chanlocs, list):
            for item in chanlocs:
                label = get_field(item, "labels") or get_field(item, "label") or get_field(item, "name")
                if label is not None:
                    labels.extend(labels_from_any(label))
            return labels
        arr = np.asarray(chanlocs, dtype=object)
        if arr.dtype.names and "labels" in arr.dtype.names:
            return labels_from_any(arr["labels"])
        # mat73/simplify_cells often gives dict of arrays: {'labels': [...], 'X': [...]}.
        label_field = get_field(chanlocs, "labels") or get_field(chanlocs, "label") or get_field(chanlocs, "name")
        return labels_from_any(label_field)

    def _extract_time(self, eeg: Any) -> np.ndarray | None:
        times = numeric_vector(get_field(eeg, "times"))
        if times is None:
            return None
        # EEGLAB times are conventionally milliseconds. For short synthetic or
        # epoched files the maximum may be well below 100 ms, so also check
        # whether the median step looks millisecond-scale rather than seconds.
        finite = times[np.isfinite(times)]
        if finite.size:
            diffs = np.diff(finite)
            diffs = diffs[diffs > 0]
            median_step = float(np.median(diffs)) if diffs.size else 0.0
            if np.nanmax(np.abs(finite)) > 10 or median_step > 0.5:
                return times / 1000.0
        return times

    def _safe_int(self, value: Any) -> int | None:
        f = safe_float(value, default=None)
        if f is None or not np.isfinite(f):
            return None
        return int(f)
