from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_channel_names(channels_csv: str | Path | None, n_channels: int) -> list[str]:
    if channels_csv:
        p = Path(channels_csv).expanduser()
        if p.exists():
            try:
                df = pd.read_csv(p)
                for col in ("name", "channel_name", "label", "channel"):
                    if col in df.columns:
                        names = [str(x) for x in df[col].tolist()]
                        if len(names) == n_channels:
                            return names
            except Exception:
                pass
    return [f"ch_{i:03d}" for i in range(n_channels)]


def load_fs(metadata_json: str | Path | None, fs: float | None) -> float:
    if fs:
        return float(fs)
    if metadata_json:
        p = Path(metadata_json).expanduser()
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                for k in ("sampling_rate", "sampling_rate_hz", "sample_rate_hz", "fs", "sfreq"):
                    if meta.get(k) is not None:
                        return float(meta[k])
            except Exception:
                pass
    return 1000.0


async def stream_speedmouse_samples(
    websocket: Any,
    *,
    source: str | Path,
    channels_csv: str | Path | None = None,
    metadata_json: str | Path | None = None,
    fs: float | None = None,
    chunk_samples: int = 256,
    speed: float = 1.0,
    loop: bool = False,
) -> None:
    signal = np.load(Path(source).expanduser(), mmap_mode="r")
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal.npy, got shape {signal.shape}")
    n_samples, n_channels = signal.shape
    sample_rate = load_fs(metadata_json, fs)
    channel_names = load_channel_names(channels_csv, n_channels)
    metadata = {
        "type": "metadata",
        "n_channels": n_channels,
        "channel_names": channel_names,
        "channels": channel_names,
        "sampling_rate_hz": sample_rate,
        "sample_rate_hz": sample_rate,
        "shape": [int(n_samples), int(n_channels)],
        "source": str(source),
    }
    await websocket.send_json(metadata)
    pos = 0
    seq = 0
    sleep_time = max(0.0, chunk_samples / sample_rate / max(speed, 1e-9)) if sample_rate else 0.1
    while True:
        if pos >= n_samples:
            if not loop:
                await websocket.send_json({"type": "end", "sequence_number": seq, "message": "Replay complete."})
                return
            pos = 0
        chunk = np.asarray(signal[pos:min(n_samples, pos + chunk_samples)], dtype=np.float32)
        frame = {
            "type": "samples",
            "sequence_number": seq,
            "start_sample": int(pos),
            "start_time_sec": float(pos / sample_rate) if sample_rate else None,
            "n_channels": n_channels,
            "channel_names": channel_names,
            "sampling_rate_hz": sample_rate,
            "samples": chunk.tolist(),
        }
        await websocket.send_json(frame)
        seq += 1
        pos += chunk.shape[0]
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
