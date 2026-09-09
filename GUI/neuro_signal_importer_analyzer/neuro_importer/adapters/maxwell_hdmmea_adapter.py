from __future__ import annotations

from pathlib import Path
from typing import Any

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.generic_hdf5_continuous_adapter import GenericHDF5ContinuousAdapter
from neuro_importer.detect.hdf5_signature_detector import collect_attrs, walk_dict


class MaxwellHDMEAAdapter(BaseAdapter):
    """Conservative continuous adapter for MaxWell/HD-MEA-like HDF5 exports.

    Without a guaranteed public export schema, this adapter detects common
    MaxWell/HD-MEA names and delegates signal extraction to the generic HDF5
    continuous adapter. Real lab samples should be used to harden this later.
    """

    name = "maxwell_hdmmea"

    def __init__(self) -> None:
        self.generic = GenericHDF5ContinuousAdapter()

    def score(self, raw: Any) -> AdapterScore:
        if not isinstance(raw, dict) or raw.get("__file_type__") != "hdf5":
            return AdapterScore(self.name, 0.0, ["not an HDF5 dictionary produced by HDF5Reader"])
        all_text = " ".join([p.lower() for p, _ in walk_dict(raw)] + [str(k).lower()+str(v).lower() for k, v in collect_attrs(raw).items()])
        if not any(token in all_text for token in ("maxwell", "maxone", "hd-mea", "hdmea", "mxw")):
            return AdapterScore(self.name, 0.0, ["no MaxWell/HD-MEA signature found"])
        base = self.generic.score(raw)
        if base.confidence <= 0:
            return AdapterScore(self.name, 0.2, ["MaxWell/HD-MEA signature found, but no continuous signal candidate found"])
        return AdapterScore(self.name, min(0.86, base.confidence + 0.12), ["MaxWell/HD-MEA signature found"] + base.reasons)

    def convert(self, raw: Any, *, source_path: str | Path | None = None, **kwargs: Any):
        rec = self.generic.convert(raw, source_path=source_path, **kwargs)
        rec.quality.adapter = self.name
        rec.metadata["format"] = "MaxWell/HD-MEA-like continuous HDF5"
        return rec
