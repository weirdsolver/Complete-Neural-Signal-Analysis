from __future__ import annotations

import numpy as np

from neuro_importer.core.recording import Recording


def validate_recording(recording: Recording) -> Recording:
    """Validate canonical neural-signal invariants and annotate quality report."""
    q = recording.quality

    if recording.signal.ndim != 2:
        q.add_error(f"Signal is not 2D: {recording.signal.shape}")
        raise ValueError(q.errors[-1])

    if recording.n_samples == 0:
        q.add_error("Signal has zero samples.")
        raise ValueError(q.errors[-1])

    if recording.n_channels == 0:
        q.add_error("Signal has zero channels.")
        raise ValueError(q.errors[-1])

    if len(recording.channels) != recording.n_channels:
        q.add_error(f"Channel table length ({len(recording.channels)}) does not match signal channels ({recording.n_channels}).")
        raise ValueError(q.errors[-1])

    if recording.time is not None:
        if len(recording.time) != recording.n_samples:
            q.add_error(f"Time length ({len(recording.time)}) does not match samples ({recording.n_samples}).")
            raise ValueError(q.errors[-1])
        finite = np.isfinite(recording.time)
        if not finite.all():
            q.add_warning("Time vector contains non-finite values.")
        if len(recording.time) > 1 and np.any(np.diff(recording.time[finite]) < 0):
            q.add_warning("Time vector is not monotonically increasing.")

    if not np.isfinite(recording.signal).all():
        q.add_warning("Signal contains NaN or infinite values.")

    if recording.sampling_rate is not None:
        if recording.sampling_rate <= 0:
            q.add_warning(f"Sampling rate is non-positive: {recording.sampling_rate}")
        elif recording.sampling_rate > 1_000_000:
            q.add_warning(f"Sampling rate seems unusually high: {recording.sampling_rate}")

    if "name" in recording.channels.columns:
        names = recording.channels["name"].astype(str).tolist()
        if len(names) != len(set(names)):
            q.add_warning("Duplicate channel names detected.")

    q.add_info("Recording validation completed.")
    return recording
