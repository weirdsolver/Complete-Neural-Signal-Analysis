from __future__ import annotations

from pathlib import Path
from typing import Any


class MNEReader:
    """Read common MNE-supported continuous neural signal files.

    This reader is intentionally for primary/header files only.  EEGLAB .set
    files may require the matching .fdt sidecar next to the .set file.
    BrainVision .vhdr files may require .eeg/.vmrk sidecars next to the .vhdr.
    """

    def read(self, path: str | Path) -> Any:
        try:
            import mne  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "Install MNE to read EDF/BDF/EEGLAB/BrainVision/FIF files: pip install mne"
            ) from exc

        p = Path(path)
        name = p.name.lower()
        suffix = p.suffix.lower()

        try:
            if suffix == ".edf":
                return mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
            if suffix == ".bdf":
                return mne.io.read_raw_bdf(str(p), preload=True, verbose="ERROR")
            if suffix == ".set":
                # MNE will load embedded-data .set files and .set + .fdt pairs.
                # When the .fdt is missing, wrap the error with an actionable hint.
                return mne.io.read_raw_eeglab(str(p), preload=True, verbose="ERROR")
            if suffix == ".vhdr":
                return mne.io.read_raw_brainvision(str(p), preload=True, verbose="ERROR")
            if suffix == ".fif" or name.endswith(".fif.gz"):
                return mne.io.read_raw_fif(str(p), preload=True, verbose="ERROR")
        except FileNotFoundError as exc:
            missing = str(exc)
            if suffix == ".set" or ".fdt" in missing.lower():
                raise FileNotFoundError(
                    "EEGLAB .set loading failed because the matching .fdt signal file was not found. "
                    "Upload/download the .set and .fdt together in the same folder, then select/drop the .set file. "
                    f"Original MNE error: {missing}"
                ) from exc
            if suffix == ".vhdr" or any(x in missing.lower() for x in (".eeg", ".vmrk")):
                raise FileNotFoundError(
                    "BrainVision .vhdr loading failed because a required .eeg or .vmrk sidecar was not found. "
                    "Upload/download the .vhdr, .eeg, and .vmrk files together in the same folder, then select/drop the .vhdr file. "
                    f"Original MNE error: {missing}"
                ) from exc
            raise

        raise ValueError(f"Unsupported MNE file suffix: {suffix}")
