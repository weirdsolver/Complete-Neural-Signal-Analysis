from __future__ import annotations

from pathlib import Path
from typing import Any


class EDFReader:
    """Read EDF/BDF files using MNE-Python when installed."""

    def read(self, path: str | Path) -> Any:
        try:
            import mne  # type: ignore
        except Exception as exc:
            raise ImportError("Install MNE to read EDF/BDF files: pip install mne") from exc
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".edf":
            return mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
        if suffix == ".bdf":
            return mne.io.read_raw_bdf(str(p), preload=True, verbose="ERROR")
        raise ValueError(f"Unsupported EDF/BDF suffix: {suffix}")
