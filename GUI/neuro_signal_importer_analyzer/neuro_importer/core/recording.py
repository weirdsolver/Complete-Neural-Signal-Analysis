from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from neuro_importer.core.quality import QualityReport


@dataclass
class Recording:
    """Canonical in-memory continuous neural-signal representation.

    The central invariant is:
        signal.shape == (n_samples, n_channels)

    v0.3 remains continuous-signal focused. It intentionally does not store
    spike tables, stimulation events, or behavioral events as first-class data.
    """

    signal: np.ndarray
    sampling_rate: float | None
    time: np.ndarray | None
    channels: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    quality: QualityReport = field(default_factory=QualityReport)
    source_path: str | None = None
    electrodes: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal)
        if self.signal.ndim != 2:
            raise ValueError(f"Recording.signal must be 2D samples × channels, got shape {self.signal.shape}")
        if not np.issubdtype(self.signal.dtype, np.number):
            self.signal = self.signal.astype(float)
        if self.time is not None:
            self.time = np.asarray(self.time).reshape(-1)
        if self.electrodes is not None and not isinstance(self.electrodes, pd.DataFrame):
            self.electrodes = pd.DataFrame(self.electrodes)

    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.signal.shape[1])

    def channel_names(self) -> list[str]:
        if "name" not in self.channels.columns:
            return [f"ch_{i:03d}" for i in range(self.n_channels)]
        return [str(x) for x in self.channels["name"].tolist()]

    def effective_time(self) -> np.ndarray:
        """Return stored time vector or synthesize one from sampling rate/sample index."""
        if self.time is not None and len(self.time) == self.n_samples:
            return self.time.astype(float)
        if self.sampling_rate is not None and self.sampling_rate > 0:
            return np.arange(self.n_samples, dtype=float) / float(self.sampling_rate)
        return np.arange(self.n_samples, dtype=float)

    def to_dataframe(self, *, time_column: str = "Time", time_last: bool = True) -> pd.DataFrame:
        """Return signal as a DataFrame, with Time first or last."""
        df = pd.DataFrame(self.signal, columns=self.channel_names())
        if time_last:
            df[time_column] = self.effective_time()
        else:
            df.insert(0, time_column, self.effective_time())
        return df

    def provenance(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "sampling_rate": self.sampling_rate,
            "metadata": self.metadata,
            "quality": self.quality.to_dict(),
        }
