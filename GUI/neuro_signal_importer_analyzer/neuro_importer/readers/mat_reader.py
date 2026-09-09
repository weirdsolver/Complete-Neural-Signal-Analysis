from __future__ import annotations

from pathlib import Path
from typing import Any


class MatReader:
    """Read MATLAB files with multiple fallbacks.

    Order:
    1. mat73, good for MATLAB v7.3/HDF5 files like the uploaded notebook used.
    2. scipy.io.loadmat, good for older MATLAB files.
    3. h5py fallback, useful for simple HDF5-backed MAT files.
    """

    def read(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        errors: list[str] = []

        try:
            import mat73  # type: ignore

            return mat73.loadmat(str(p))
        except Exception as exc:
            errors.append(f"mat73 failed: {exc}")

        try:
            from scipy.io import loadmat

            return loadmat(str(p), simplify_cells=True)
        except Exception as exc:
            errors.append(f"scipy.io.loadmat failed: {exc}")

        try:
            import h5py  # type: ignore

            return self._read_h5_basic(p, errors)
        except Exception as exc:
            errors.append(f"h5py fallback failed: {exc}")

        raise ValueError("Could not read MATLAB file. " + " | ".join(errors))

    def _read_h5_basic(self, path: Path, previous_errors: list[str]) -> dict[str, Any]:
        import h5py  # type: ignore
        import numpy as np

        def convert(obj: Any) -> Any:
            if isinstance(obj, h5py.Dataset):
                arr = obj[()]
                # h5py often returns byte strings for char data.
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
                return {key: convert(obj[key]) for key in obj.keys()}
            return obj

        with h5py.File(path, "r") as f:
            return {key: convert(f[key]) for key in f.keys()}
