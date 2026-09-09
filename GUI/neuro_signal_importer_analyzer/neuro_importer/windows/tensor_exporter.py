from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neuro_importer.core.recording import Recording
from neuro_importer.windows.window_builder import build_windows


class TensorExporter:
    """Export fixed windows/tensors from continuous signal recordings."""

    def export(self, recording: Recording, output_dir: str | Path, config: dict[str, Any]) -> dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        windowed = build_windows(recording, config)
        x_path = out / "X_windows.npy"
        index_path = out / "window_index.csv"
        np.save(x_path, windowed.X)
        windowed.index.to_csv(index_path, index=False)
        return {"X_windows": str(x_path), "window_index": str(index_path)}
