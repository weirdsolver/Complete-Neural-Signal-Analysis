from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def array_size_mb(arr: np.ndarray) -> float:
    return float(np.asarray(arr).nbytes) / (1024.0 * 1024.0)


def save_signal_array(signal: np.ndarray, output_dir: str | Path, *, export_format: str = 'npy', compression: str | None = 'gzip') -> tuple[str, dict[str, Any]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(signal)
    info = {'export_format': export_format, 'shape': list(arr.shape), 'dtype': str(arr.dtype), 'size_mb': array_size_mb(arr)}
    if export_format == 'npy':
        path = out / 'signal.npy'
        np.save(path, arr)
        return str(path), info
    if export_format == 'memmap':
        path = out / 'signal_memmap.npy'
        mm = np.lib.format.open_memmap(path, mode='w+', dtype=arr.dtype, shape=arr.shape)
        mm[:] = arr
        del mm
        return str(path), info
    if export_format == 'hdf5':
        try:
            import h5py  # type: ignore
        except Exception as exc:
            raise ImportError('Install h5py to export HDF5: pip install h5py') from exc
        path = out / 'signal.h5'
        with h5py.File(path, 'w') as f:
            f.create_dataset('signal', data=arr, compression=compression)
        return str(path), info
    if export_format == 'zarr':
        try:
            import zarr  # type: ignore
        except Exception as exc:
            raise ImportError('Install zarr to export Zarr: pip install zarr') from exc
        path = out / 'signal.zarr'
        zarr.save(str(path), arr)
        return str(path), info
    raise ValueError(f'Unsupported export_format: {export_format}')
