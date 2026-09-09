from __future__ import annotations

from pathlib import Path


def detect_file_type(path: str | Path) -> str:
    """Detect broad container/file type from suffix.

    This is intentionally conservative. A later version can add magic-byte
    detection, BIDS-folder detection, and vendor-specific sniffing.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".mat":
        return "mat"
    if suffix in {".edf", ".bdf"}:
        return "edf"
    if suffix in {".set", ".vhdr", ".fif"} or p.name.lower().endswith(".fif.gz"):
        return "mne"
    if suffix in {".h5", ".hdf5", ".nwb"}:
        return "hdf5"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    if suffix in {".npy", ".npz"}:
        return "numpy"
    return "unknown"
