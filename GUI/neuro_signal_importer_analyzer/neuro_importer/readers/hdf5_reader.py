from __future__ import annotations

from pathlib import Path
from typing import Any


class HDF5Reader:
    """Read HDF5/NWB/MEA-like files into nested dictionaries.

    This lightweight reader does not interpret NWB semantics yet. It exposes a
    nested dictionary so GenericMatAdapter can find simple signal arrays. A later
    NWB-specific adapter should use pynwb for richer metadata.
    """

    def read(self, path: str | Path) -> dict[str, Any]:
        try:
            import h5py  # type: ignore
            import numpy as np
        except Exception as exc:
            raise ImportError("Install h5py to read HDF5/NWB files: pip install h5py") from exc

        p = Path(path)

        def convert(obj: Any) -> Any:
            if isinstance(obj, h5py.Dataset):
                arr = obj[()]
                if isinstance(arr, bytes):
                    try:
                        return arr.decode()
                    except Exception:
                        return arr
                if isinstance(arr, np.ndarray) and arr.dtype.kind == "S":
                    try:
                        return arr.astype(str)
                    except Exception:
                        return arr
                return arr
            if isinstance(obj, h5py.Group):
                out: dict[str, Any] = {key: convert(obj[key]) for key in obj.keys()}
                if obj.attrs:
                    out["__attrs__"] = {str(k): v for k, v in obj.attrs.items()}
                return out
            return obj

        with h5py.File(p, "r") as f:
            out = {key: convert(f[key]) for key in f.keys()}
            if f.attrs:
                out["__attrs__"] = {str(k): v for k, v in f.attrs.items()}
            out["__source_path__"] = str(p)
            out["__file_type__"] = "hdf5"
            return out
