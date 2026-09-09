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


class FieldTripAdapter(BaseAdapter):
    """Adapter for FieldTrip-style MATLAB raw structures.

    Expected fields are usually:
        trial   -> list/cell of arrays, each channels × samples
        time    -> list/cell of time vectors, one per trial
        label   -> channel labels
        fsample -> sampling rate

    First version concatenates trials along time and records trial lengths in metadata.
    """

    name = "fieldtrip_mat"

    def score(self, raw: Any) -> AdapterScore:
        ft = self._extract_candidate(raw)
        if ft is None:
            return AdapterScore(self.name, 0.0, ["no FieldTrip-like structure found"])
        reasons = ["FieldTrip-like structure found"]
        confidence = 0.20
        if get_field(ft, "trial") is not None:
            confidence += 0.35
            reasons.append("trial field found")
        if get_field(ft, "label") is not None:
            confidence += 0.20
            reasons.append("label field found")
        if get_field(ft, "fsample") is not None:
            confidence += 0.15
            reasons.append("fsample field found")
        if get_field(ft, "time") is not None:
            confidence += 0.10
            reasons.append("time field found")
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

        ft = self._extract_candidate(raw)
        if ft is None:
            q.add_error("No FieldTrip-like structure was found.")
            raise ValueError(q.errors[-1])

        trial_raw = get_field(ft, "trial")
        if trial_raw is None:
            q.add_error("FieldTrip trial field is missing.")
            raise ValueError(q.errors[-1])

        labels = labels_from_any(get_field(ft, "label"))
        fs = safe_float(get_field(ft, "fsample"), default=None)
        trials = self._trial_list(trial_raw)
        if not trials:
            q.add_error("No numeric FieldTrip trials were found.")
            raise ValueError(q.errors[-1])

        norm_trials: list[np.ndarray] = []
        trial_lengths: list[int] = []
        for i, trial in enumerate(trials):
            arr = as_numeric_2d(trial, name=f"trial[{i}]")
            sig, note = normalize_signal_orientation(arr, labels=labels, quality=q)
            if i == 0:
                q.add_assumption(note)
            norm_trials.append(sig)
            trial_lengths.append(sig.shape[0])
        if len({x.shape[1] for x in norm_trials}) != 1:
            q.add_error("FieldTrip trials have inconsistent channel counts after orientation normalization.")
            raise ValueError(q.errors[-1])
        signal = np.concatenate(norm_trials, axis=0)
        if len(norm_trials) > 1:
            q.add_warning("Multiple FieldTrip trials were concatenated along time for neural-signal-only export.")

        time = self._concat_time(get_field(ft, "time"), trial_lengths, fs, q)
        channels = make_channel_table(labels, signal.shape[1], quality=q)
        signal, channels = apply_aux_filter(signal, channels, include_aux=include_aux, quality=q)
        time = synthesize_or_validate_time(time, signal.shape[0], fs, q)

        metadata = {
            "format": "FieldTrip MATLAB",
            "source_path": str(source_path) if source_path is not None else None,
            "sampling_rate": fs,
            "subject": subject,
            "session": session,
            "n_trials": len(norm_trials),
            "trial_lengths_samples": trial_lengths,
            "canonical_signal_shape": tuple(int(x) for x in signal.shape),
        }
        return Recording(signal=signal, sampling_rate=fs, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None)

    def _extract_candidate(self, raw: Any) -> Any | None:
        if get_field(raw, "trial") is not None and get_field(raw, "label") is not None:
            return raw
        if isinstance(raw, dict):
            for key in ("data", "raw", "ft_data", "fieldtrip", "clean_data"):
                value = raw.get(key)
                if value is not None and get_field(value, "trial") is not None and get_field(value, "label") is not None:
                    return value
            for value in raw.values():
                if get_field(value, "trial") is not None and get_field(value, "label") is not None:
                    return value
        return None

    def _trial_list(self, trial_raw: Any) -> list[Any]:
        if isinstance(trial_raw, list):
            return trial_raw
        arr = np.asarray(trial_raw, dtype=object)
        if arr.ndim == 0:
            return [arr.item()]
        # Object arrays are likely MATLAB cells. Numeric 2D array is a single trial.
        if arr.dtype == object:
            return [x for x in arr.reshape(-1)]
        return [trial_raw]

    def _concat_time(self, time_raw: Any, trial_lengths: list[int], fs: float | None, q: QualityReport) -> np.ndarray | None:
        if time_raw is None:
            return None
        if isinstance(time_raw, list):
            pieces = [numeric_vector(x) for x in time_raw]
        else:
            arr = np.asarray(time_raw, dtype=object)
            if arr.dtype == object and arr.size == len(trial_lengths):
                pieces = [numeric_vector(x) for x in arr.reshape(-1)]
            else:
                pieces = [numeric_vector(time_raw)]
        if not pieces or any(p is None for p in pieces):
            return None
        valid = [p for p in pieces if p is not None]
        if len(valid) == 1 and sum(trial_lengths) != len(valid[0]) and len(trial_lengths) > 1:
            return None
        if len(valid) == 1:
            return valid[0]
        if len(valid) != len(trial_lengths):
            return None
        out: list[np.ndarray] = []
        offset = 0.0
        for t, n in zip(valid, trial_lengths):
            if len(t) != n:
                return None
            tt = np.asarray(t, dtype=float)
            if out:
                if fs and fs > 0:
                    offset = out[-1][-1] + 1.0 / fs - tt[0]
                else:
                    offset = out[-1][-1] + 1.0 - tt[0]
            out.append(tt + offset)
        if len(out) > 1:
            q.add_warning("FieldTrip trial time vectors were concatenated with offsets.")
        return np.concatenate(out)
