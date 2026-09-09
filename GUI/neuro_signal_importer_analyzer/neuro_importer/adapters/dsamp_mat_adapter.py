from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.utils import as_scalar, as_vector, flatten, get_field, safe_float, stringify_label
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording


class DSampMatAdapter(BaseAdapter):
    """Complete adapter for DSamp-style MATLAB files.

    This adapter is based on the uploaded notebook, but the first production
    version intentionally focuses only on neural signal data.

    It extracts:
    - DSamp.EEGdata -> Recording.signal
    - DSamp.fs -> Recording.sampling_rate
    - DSamp.time -> Recording.time
    - DSamp.label -> Recording.channels
    - DSamp.Subj and basic DSamp fields -> Recording.metadata

    It ignores DSamp.triggers/stimulation for now because the current milestone
    is clean neural signal conversion only.
    """

    name = "dsamp_mat"

    def __init__(self, *, exclude_channels: list[str] | None = None, include_aux: bool = False) -> None:
        self.exclude_channels = exclude_channels or ["BIP1", "BIP2", "RESP1"]
        self.include_aux = include_aux

    def score(self, raw: Any) -> AdapterScore:
        reasons: list[str] = []
        confidence = 0.0

        dsamp = self._extract_dsamp(raw)
        if dsamp is None:
            return AdapterScore(self.name, 0.0, ["no DSamp object found"])

        confidence += 0.35
        reasons.append("top-level DSamp object found")

        if get_field(dsamp, "EEGdata") is not None:
            confidence += 0.30
            reasons.append("DSamp.EEGdata found")
        if get_field(dsamp, "fs") is not None:
            confidence += 0.15
            reasons.append("DSamp.fs found")
        if get_field(dsamp, "label") is not None:
            confidence += 0.15
            reasons.append("DSamp.label found")
        if get_field(dsamp, "time") is not None:
            confidence += 0.05
            reasons.append("DSamp.time found")

        return AdapterScore(self.name, min(confidence, 1.0), reasons)

    def convert(
        self,
        raw: Any,
        *,
        source_path: str | Path | None = None,
        subject: int | float | str | None = None,
        session: int | float | str | None = None,
        include_aux: bool | None = None,
        **_: Any,
    ) -> Recording:
        quality = QualityReport(adapter=self.name)
        score = self.score(raw)
        quality.confidence = score.confidence
        for reason in score.reasons:
            quality.add_info(reason)

        dsamp = self._extract_dsamp(raw)
        if dsamp is None:
            quality.add_error("No DSamp object was found.")
            raise ValueError("No DSamp object was found.")

        eeg_raw = get_field(dsamp, "EEGdata")
        if eeg_raw is None:
            quality.add_error("DSamp.EEGdata is missing.")
            raise ValueError("DSamp.EEGdata is missing.")

        signal_raw = np.asarray(eeg_raw)
        if signal_raw.ndim != 2:
            signal_raw = signal_raw.squeeze()
        if signal_raw.ndim != 2:
            raise ValueError(f"DSamp.EEGdata must resolve to a 2D array. Got shape {signal_raw.shape}")
        if not np.issubdtype(signal_raw.dtype, np.number):
            signal_raw = signal_raw.astype(float)

        fs = safe_float(get_field(dsamp, "fs"), default=None)
        fs_old = safe_float(get_field(dsamp, "fsOld"), default=None)
        rate = safe_float(get_field(dsamp, "rate"), default=None)
        nchan = safe_float(get_field(dsamp, "nchan"), default=None)
        npt = safe_float(get_field(dsamp, "npt"), default=None)

        time = self._extract_time(dsamp)
        labels = self._extract_labels(dsamp)

        signal, orientation_note = self._normalize_signal_orientation(
            signal_raw,
            labels=labels,
            time=time,
            nchan=nchan,
            npt=npt,
            quality=quality,
        )
        quality.add_assumption(orientation_note)

        channels_all = self._build_channels(labels, signal.shape[1], quality)
        keep_aux = self.include_aux if include_aux is None else include_aux

        if not keep_aux:
            include_mask = channels_all["include_by_default"].to_numpy(dtype=bool)
            if len(include_mask) == signal.shape[1] and not include_mask.all():
                excluded = channels_all.loc[~include_mask, "name"].tolist()
                signal = signal[:, include_mask]
                channels = channels_all.loc[include_mask].copy().reset_index(drop=True)
                channels["index"] = range(len(channels))
                quality.add_info(f"Excluded auxiliary/non-neural channels from signal export: {excluded}")
            else:
                channels = channels_all
        else:
            channels = channels_all.copy()
            quality.add_info("Kept auxiliary channels because include_aux=True.")

        if time is None or len(time) != signal.shape[0]:
            if fs is not None and fs > 0:
                time = np.arange(signal.shape[0], dtype=float) / float(fs)
                quality.add_warning("Time vector missing/mismatched; synthesized seconds from sampling rate.")
            else:
                time = np.arange(signal.shape[0], dtype=float)
                quality.add_warning("Time vector and sampling rate missing/mismatched; synthesized sample index as Time.")

        if subject is None:
            subj_raw = as_scalar(get_field(dsamp, "Subj"), default=None)
            subject = str(subj_raw).strip() if subj_raw not in (None, "") else None
        if session is None:
            session = None

        if get_field(dsamp, "triggers") is not None:
            quality.add_info("DSamp.triggers was found but intentionally ignored in this neural-signal-only version.")

        metadata = {
            "format": "DSamp MATLAB",
            "source_path": str(source_path) if source_path is not None else None,
            "sampling_rate": fs,
            "fsOld": fs_old,
            "rate": rate,
            "nchan_declared": nchan,
            "npt_declared": npt,
            "subject": subject,
            "session": session,
            "auxiliary_channels_excluded": not keep_aux,
            "excluded_channels_default": self.exclude_channels,
            "raw_signal_shape": tuple(int(x) for x in signal_raw.shape),
            "canonical_signal_shape": tuple(int(x) for x in signal.shape),
            "triggers_ignored": get_field(dsamp, "triggers") is not None,
        }

        return Recording(
            signal=signal,
            sampling_rate=fs,
            time=time,
            channels=channels,
            metadata=metadata,
            quality=quality,
            source_path=str(source_path) if source_path is not None else None,
        )

    def _extract_dsamp(self, raw: Any) -> Any | None:
        if isinstance(raw, dict) and "DSamp" in raw:
            return raw["DSamp"]
        return get_field(raw, "DSamp", default=None)

    def _extract_time(self, dsamp: Any) -> np.ndarray | None:
        value = get_field(dsamp, "time")
        if value is None:
            return None
        arr = as_vector(value)
        if arr.size == 0:
            return None
        numeric = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
        if np.isnan(numeric).all():
            return None
        return numeric

    def _extract_labels(self, dsamp: Any) -> list[str]:
        value = get_field(dsamp, "label")
        if value is None:
            return []
        labels = [stringify_label(v) for v in flatten(value)]
        return [x for x in labels if x not in {"", "None", "nan"}]

    def _normalize_signal_orientation(
        self,
        signal_raw: np.ndarray,
        *,
        labels: list[str],
        time: np.ndarray | None,
        nchan: float | None,
        npt: float | None,
        quality: QualityReport,
    ) -> tuple[np.ndarray, str]:
        rows, cols = signal_raw.shape
        label_count = len(labels) if labels else None
        time_count = len(time) if time is not None else None
        nchan_i = int(nchan) if nchan is not None and np.isfinite(nchan) else None
        npt_i = int(npt) if npt is not None and np.isfinite(npt) else None

        channel_counts = {x for x in [label_count, nchan_i] if x is not None and x > 0}
        sample_counts = {x for x in [time_count, npt_i] if x is not None and x > 0}

        if rows in channel_counts and cols in sample_counts:
            return signal_raw.T, "Original signal appeared to be channels × samples; transposed to samples × channels."
        if cols in channel_counts and rows in sample_counts:
            return signal_raw, "Original signal appeared to already be samples × channels."

        if label_count is not None:
            if rows == label_count and cols != label_count:
                return signal_raw.T, "Original signal first dimension matched channel labels; transposed to samples × channels."
            if cols == label_count:
                return signal_raw, "Original signal second dimension matched channel labels; kept as samples × channels."

        if rows < cols:
            quality.add_warning("Could not prove signal orientation from metadata; assumed rows are channels because rows < columns.")
            return signal_raw.T, "Heuristic orientation assumption: rows < columns, so transposed to samples × channels."

        quality.add_warning("Could not prove signal orientation from metadata; assumed rows are samples.")
        return signal_raw, "Heuristic orientation assumption: rows are samples and columns are channels."

    def _build_channels(self, labels: list[str], n_channels: int, quality: QualityReport) -> pd.DataFrame:
        if not labels:
            labels = [f"ch_{i:03d}" for i in range(n_channels)]
            quality.add_warning("No labels found; generated generic channel names.")
        if len(labels) != n_channels:
            quality.add_warning(f"Label count ({len(labels)}) did not match signal channels ({n_channels}); adjusting labels.")
            if len(labels) < n_channels:
                labels = labels + [f"ch_{i:03d}" for i in range(len(labels), n_channels)]
            else:
                labels = labels[:n_channels]

        exclude_upper = {x.upper() for x in self.exclude_channels}
        rows = []
        for idx, name in enumerate(labels):
            upper = name.upper()
            is_aux = upper in exclude_upper or upper.startswith(("RESP", "AUX", "BIP"))
            if upper.startswith("RESP"):
                ch_type = "respiratory"
            elif upper.startswith("BIP"):
                ch_type = "bipolar_aux"
            elif upper.startswith("AUX"):
                ch_type = "auxiliary"
            else:
                ch_type = "neural"
            rows.append({"index": idx, "name": name, "type": ch_type, "include_by_default": not is_aux})
        return pd.DataFrame(rows)
