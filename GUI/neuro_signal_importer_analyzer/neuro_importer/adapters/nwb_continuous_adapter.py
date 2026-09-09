from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.generic_hdf5_continuous_adapter import GenericHDF5ContinuousAdapter
from neuro_importer.detect.hdf5_signature_detector import looks_like_nwb, walk_dict


class NWBContinuousAdapter(BaseAdapter):
    """Continuous-signal adapter for simple NWB HDF5 trees.

    This lightweight v0.3 adapter prefers acquisition/*/data datasets. It does
    not yet use PyNWB semantic objects; it stays within the existing HDF5 reader.
    """

    name = "nwb_continuous"

    def __init__(self) -> None:
        self.generic = GenericHDF5ContinuousAdapter()

    def score(self, raw: Any) -> AdapterScore:
        if not isinstance(raw, dict) or raw.get("__file_type__") != "hdf5" or not looks_like_nwb(raw):
            return AdapterScore(self.name, 0.0, ["not an NWB-like HDF5 tree"])
        data_paths = self._acquisition_data_paths(raw)
        if not data_paths:
            return AdapterScore(self.name, 0.25, ["NWB-like tree found but no acquisition/*/data 2D candidate found"])
        return AdapterScore(self.name, 0.88, [f"NWB acquisition data candidates: {data_paths[:8]}"])

    def convert(self, raw: Any, *, source_path: str | Path | None = None, **kwargs: Any):
        data_paths = self._acquisition_data_paths(raw)
        if data_paths and "signal_path" not in kwargs:
            kwargs["signal_path"] = data_paths[0]
        rec = self.generic.convert(raw, source_path=source_path, **kwargs)
        rec.quality.adapter = self.name
        rec.metadata["format"] = "NWB-like continuous HDF5"
        rec.metadata["nwb_signal_path"] = kwargs.get("signal_path", data_paths[0] if data_paths else None)
        return rec

    def _acquisition_data_paths(self, raw: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for path, value in walk_dict(raw):
            lower = path.lower()
            if "/acquisition/" in lower and lower.endswith("/data"):
                try:
                    arr = np.asarray(value).squeeze()
                    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                        out.append(path)
                except Exception:
                    continue
        return out
