from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class NumpyReader:
    """Read .npy and .npz files."""

    def read(self, path: str | Path) -> Any:
        p = Path(path)
        if p.suffix.lower() == ".npy":
            return np.load(p, allow_pickle=True)
        if p.suffix.lower() == ".npz":
            data = np.load(p, allow_pickle=True)
            return {key: data[key] for key in data.files}
        raise ValueError(f"Unsupported NumPy file suffix: {p.suffix}")
