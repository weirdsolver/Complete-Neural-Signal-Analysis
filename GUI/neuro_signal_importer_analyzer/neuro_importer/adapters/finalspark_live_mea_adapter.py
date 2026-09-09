from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.common import make_channel_table, synthesize_or_validate_time
from neuro_importer.core.electrode_table import make_electrode_table
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording
from neuro_importer.detect.hdf5_signature_detector import collect_attrs, looks_like_finalspark_live_mea
from neuro_importer.core.signal_units import infer_sampling_rate


class FinalSparkLiveMEAAdapter(BaseAdapter):
    """Continuous voltage adapter for FinalSpark LiveMEA-style HDF5 chunks.

    This adapter handles files where each time/chunk group contains multiple
    electrode/channel datasets. It intentionally ignores triggers, stim metadata,
    and spike-derived tables.
    """

    name = "finalspark_live_mea"

    def score(self, raw: Any) -> AdapterScore:
        if not isinstance(raw, dict) or raw.get("__file_type__") != "hdf5":
            return AdapterScore(self.name, 0.0, ["not an HDF5 dictionary produced by HDF5Reader"])
        if not looks_like_finalspark_live_mea(raw):
            return AdapterScore(self.name, 0.0, ["does not look like timestamp/chunk groups with electrode datasets"])
        groups = self._timestamp_groups(raw)
        max_e = max((len(self._electrode_keys(g)) for _, g in groups), default=0)
        return AdapterScore(self.name, 0.93, [f"FinalSpark LiveMEA-like timestamp groups found: {len(groups)}", f"max electrode datasets per group: {max_e}"])

    def convert(
        self,
        raw: Any,
        *,
        source_path: str | Path | None = None,
        subject: int | float | str | None = None,
        session: int | float | str | None = None,
        sampling_rate: float | None = None,
        include_aux: bool = False,
        **_: Any,
    ) -> Recording:
        q = QualityReport(adapter=self.name, confidence=self.score(raw).confidence)
        groups = self._timestamp_groups(raw)
        if not groups:
            raise ValueError("No FinalSpark LiveMEA timestamp/chunk groups found.")

        first_keys = self._electrode_keys(groups[0][1])
        if not first_keys:
            raise ValueError("No electrode datasets found in the first timestamp/chunk group.")
        chunks: list[np.ndarray] = []
        skipped: list[str] = []
        for group_name, group in groups:
            keys = [k for k in self._electrode_keys(group) if k in first_keys]
            if len(keys) != len(first_keys):
                skipped.append(group_name)
                continue
            columns = []
            length = None
            ok = True
            for key in first_keys:
                try:
                    vec = np.asarray(group[key]).squeeze().astype(float).reshape(-1)
                except Exception:
                    ok = False
                    break
                if length is None:
                    length = len(vec)
                elif len(vec) != length:
                    ok = False
                    break
                columns.append(vec)
            if ok and columns:
                chunks.append(np.column_stack(columns))
            else:
                skipped.append(group_name)
        if not chunks:
            raise ValueError("Could not assemble any consistent FinalSpark LiveMEA chunks.")
        signal = np.concatenate(chunks, axis=0)
        labels = [str(k) for k in first_keys]
        channels = make_channel_table(labels, signal.shape[1], quality=q)

        attrs = collect_attrs(raw)
        fs_key, fs_found = infer_sampling_rate(attrs)
        fs = sampling_rate if sampling_rate is not None else fs_found
        if fs is None:
            # Public LiveMEA examples commonly show source data downsampled 30 kHz / 8.
            # Keep it explicit in the quality report because real exports should override it.
            fs = 30000.0 / 8.0
            q.add_assumption("No sampling-rate metadata found; used FinalSpark LiveMEA default assumption 30000/8 = 3750 Hz. Pass sampling_rate=... to override.")
        time = synthesize_or_validate_time(None, signal.shape[0], fs, q)
        if skipped:
            q.add_warning(f"Skipped inconsistent timestamp/chunk groups: {skipped[:20]}")
        q.add_info(f"Assembled {len(chunks)} chunks into continuous signal shape {signal.shape}.")

        electrodes = make_electrode_table(labels, device="FinalSpark LiveMEA", extra={"source_group_count": len(chunks)})
        metadata = {
            "format": "FinalSpark LiveMEA continuous HDF5",
            "source_path": str(source_path) if source_path is not None else None,
            "sampling_rate": fs,
            "sampling_rate_path": fs_key,
            "n_chunks": len(chunks),
            "samples_per_first_chunk": int(chunks[0].shape[0]),
            "n_electrodes": int(signal.shape[1]),
            "subject": subject,
            "session": session,
            "hdf5_attrs": {str(k): str(v) for k, v in attrs.items()},
        }
        return Recording(signal=signal, sampling_rate=fs, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None, electrodes=electrodes)

    def _timestamp_groups(self, raw: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        groups = []
        for key, value in raw.items():
            if str(key).startswith("__") or not isinstance(value, dict):
                continue
            if self._looks_time_key(str(key)) and len(self._electrode_keys(value)) >= 2:
                groups.append((str(key), value))
        groups.sort(key=lambda item: self._numeric_suffix(item[0]))
        return groups

    def _looks_time_key(self, key: str) -> bool:
        return bool(re.search(r"(timestamp|time|chunk|frame)[_\-]?\d+", key, flags=re.I) or key.replace('.', '', 1).isdigit())

    def _electrode_keys(self, group: dict[str, Any]) -> list[str]:
        keys = [str(k) for k in group.keys() if re.search(r"(electrode|channel|ch)[_\-]?\d+", str(k), flags=re.I)]
        keys.sort(key=self._numeric_suffix)
        return keys

    def _numeric_suffix(self, text: str) -> float:
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        return float(nums[-1]) if nums else 0.0
