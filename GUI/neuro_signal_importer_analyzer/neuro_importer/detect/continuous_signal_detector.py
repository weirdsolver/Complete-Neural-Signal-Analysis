from __future__ import annotations

from typing import Any

import numpy as np

from neuro_importer.detect.hdf5_signature_detector import numeric_2d_datasets, walk_dict


SIGNAL_HINTS = (
    "samples",
    "sample",
    "signal",
    "signals",
    "raw",
    "voltage",
    "voltages",
    "data",
    "recording",
    "frames",
    "eeg",
    "ecog",
    "lfp",
    "mea",
)

LABEL_HINTS = ("label", "labels", "channel", "channels", "electrode", "electrodes", "ch_names", "names")
TIME_HINTS = ("time", "times", "timestamp", "timestamps", "t")
FS_HINTS = ("sampling_rate", "sample_rate", "sampling_frequency", "fs", "srate", "sfreq", "rate", "frames_per_second")


def score_signal_path(path: str, shape: tuple[int, int]) -> float:
    lower = path.lower()
    rows, cols = shape
    score = 0.0
    if any(h in lower for h in SIGNAL_HINTS):
        score += 1.5
    if lower.endswith("/samples"):
        score += 1.0
    if max(rows, cols) >= 100:
        score += 0.4
    if max(rows, cols) / max(min(rows, cols), 1) >= 5:
        score += 0.3
    # Penalize likely non-signal summary matrices.
    if any(bad in lower for bad in ("spike", "stim", "event", "trigger", "impedance", "image")):
        score -= 2.0
    return score


def find_continuous_signal_candidates(raw: dict[str, Any]) -> list[tuple[str, Any, tuple[int, int], float]]:
    candidates = []
    for path, value, shape in numeric_2d_datasets(raw):
        score = score_signal_path(path, shape)
        candidates.append((path, value, shape, score))
    candidates.sort(key=lambda item: (item[3], np.asarray(item[1]).size), reverse=True)
    return candidates


def find_vector_by_hints(raw: dict[str, Any], hints: tuple[str, ...]) -> tuple[str | None, Any | None]:
    for path, value in walk_dict(raw):
        name = path.strip("/").split("/")[-1].lower()
        if any(h == name or h in name for h in hints):
            try:
                arr = np.asarray(value)
                if arr.ndim <= 2 and arr.size >= 2:
                    return path, value
            except Exception:
                continue
    return None, None
