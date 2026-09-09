from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.common import apply_aux_filter, make_channel_table
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording


class MNERawAdapter(BaseAdapter):
    """Adapter for MNE Raw objects, commonly produced by EDF/BDF/FIF readers."""

    name = "mne_raw"

    def score(self, raw: Any) -> AdapterScore:
        if hasattr(raw, "get_data") and hasattr(raw, "info") and hasattr(raw, "ch_names"):
            return AdapterScore(self.name, 0.95, ["MNE Raw-like object found"])
        return AdapterScore(self.name, 0.0, ["not an MNE Raw-like object"])

    def convert(self, raw: Any, *, source_path: str | Path | None = None, subject=None, session=None, include_aux: bool = False, **_: Any) -> Recording:
        q = QualityReport(adapter=self.name, confidence=self.score(raw).confidence)
        q.add_info("MNE Raw-like object found")
        data = raw.get_data()  # MNE returns channels × samples
        signal = np.asarray(data, dtype=float).T
        fs = float(raw.info.get("sfreq")) if raw.info.get("sfreq") else None
        time = np.asarray(raw.times, dtype=float) if hasattr(raw, "times") else None
        labels = [str(x) for x in raw.ch_names]
        channels = make_channel_table(labels, signal.shape[1], quality=q)
        # Preserve MNE channel types when available.
        try:
            types = raw.get_channel_types()
            if len(types) == len(channels):
                channels["mne_type"] = types
                channels["type"] = ["neural" if t.lower() in {"eeg", "ecog", "seeg", "dbs", "meg"} else "auxiliary" for t in types]
                channels["include_by_default"] = [t.lower() in {"eeg", "ecog", "seeg", "dbs", "meg"} for t in types]
        except Exception:
            pass
        signal, channels = apply_aux_filter(signal, channels, include_aux=include_aux, quality=q)
        metadata = {
            "format": "MNE Raw",
            "source_path": str(source_path) if source_path is not None else None,
            "sampling_rate": fs,
            "subject": subject,
            "session": session,
            "canonical_signal_shape": tuple(int(x) for x in signal.shape),
        }
        return Recording(signal=signal, sampling_rate=fs, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None)
