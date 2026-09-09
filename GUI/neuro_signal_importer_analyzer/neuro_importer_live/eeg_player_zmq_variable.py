from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import zmq

from .channel_manifest import build_manifest


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


class VariableChannelPlayer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.source = Path(args.source)
        self.eeg = np.load(self.source, mmap_mode="r")
        if self.eeg.ndim != 2:
            raise ValueError(f"Expected 2D signal.npy array, got shape {self.eeg.shape}")
        if args.orientation == "channels_by_samples":
            self.n_samples = int(self.eeg.shape[1])
            self.n_channels = int(self.eeg.shape[0])
        else:
            self.n_samples = int(self.eeg.shape[0])
            self.n_channels = int(self.eeg.shape[1])
        self.fs_hz = float(args.fs)
        self.chunk_samples = max(1, int(round(self.fs_hz * float(args.chunk_sec))))
        self.stream_id = args.stream_id or f"neural_npy_replay_{uuid.uuid4().hex[:8]}"
        self.manifest = build_manifest(
            self.n_channels,
            channels_csv=Path(args.channels_csv) if args.channels_csv else self.source.parent / "channels.csv",
            metadata_json=Path(args.metadata_json) if args.metadata_json else self.source.parent / "metadata.json",
            profile=args.channel_profile,
            sample_rate_hz=self.fs_hz,
            units=args.units,
        )

    def _chunk(self, start: int, stop: int) -> np.ndarray:
        if self.args.orientation == "channels_by_samples":
            data = np.asarray(self.eeg[:, start:stop].T, dtype=np.float32)
        else:
            data = np.asarray(self.eeg[start:stop, :], dtype=np.float32)
        return np.ascontiguousarray(data)

    def run(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.SNDHWM, int(self.args.sndhwm))
        sock.bind(self.args.address) if self.args.bind else sock.connect(self.args.address)

        print("Starting variable-channel neural player")
        print(f"  address: {self.args.address}")
        print(f"  source: {self.source}")
        print(f"  signal shape on disk: {tuple(self.eeg.shape)}")
        print(f"  interpreted samples x channels: ({self.n_samples}, {self.n_channels})")
        print(f"  fs: {self.fs_hz:g} Hz")
        print(f"  chunk: {self.chunk_samples} samples = {self.chunk_samples / self.fs_hz:.3f} sec")
        print(f"  channel namespace: {self.manifest.channel_namespace}")

        metadata = {
            "type": "stream_metadata",
            "stream_id": self.stream_id,
            "source": str(self.source),
            "array_shape": list(map(int, self.eeg.shape)),
            "orientation": self.args.orientation,
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "sample_rate_hz": self.fs_hz,
            "chunk_samples": self.chunk_samples,
            "channel_manifest": self.manifest.to_dict(),
        }
        sock.send_json(metadata)

        seq = 0
        start_sample = max(0, int(self.args.start_sample))
        next_wall = time.perf_counter()
        while start_sample < self.n_samples:
            stop = min(self.n_samples, start_sample + self.chunk_samples)
            chunk = self._chunk(start_sample, stop)
            header = {
                "type": "neural_chunk",
                "stream_id": self.stream_id,
                "sequence_number": seq,
                "start_sample": start_sample,
                "start_time_sec": start_sample / self.fs_hz,
                "sample_rate_hz": self.fs_hz,
                "n_channels": self.n_channels,
                "shape": list(map(int, chunk.shape)),
                "dtype": "float32",
                "channel_manifest": self.manifest.to_dict(),
            }
            sock.send_multipart([json.dumps(header).encode("utf-8"), chunk.tobytes(order="C")])
            seq += 1
            start_sample = stop
            if seq % int(self.args.status_every) == 0:
                print(f"status seq={seq} sample={start_sample} t={start_sample / self.fs_hz:.3f}s channels={self.n_channels}")
            if _str2bool(self.args.realtime):
                next_wall += self.chunk_samples / self.fs_hz
                delay = next_wall - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            if start_sample >= self.n_samples and _str2bool(self.args.loop):
                start_sample = 0
        print("player complete")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Variable-channel neural signal ZeroMQ player")
    p.add_argument("--source", required=True, help="Path to signal.npy, shape samples x channels by default")
    p.add_argument("--address", default="tcp://127.0.0.1:5555")
    p.add_argument("--bind", action="store_true", default=True, help="Bind PUSH socket instead of connect")
    p.add_argument("--fs", type=float, default=1000.0)
    p.add_argument("--chunk-sec", type=float, default=0.25)
    p.add_argument("--channels-csv", default=None)
    p.add_argument("--metadata-json", default=None)
    p.add_argument("--channel-profile", default="auto", help="auto, eeg_10_10_32, eeg_10_20_19, finalspark_32, generated_numeric")
    p.add_argument("--units", default=None)
    p.add_argument("--orientation", choices=["samples_by_channels", "channels_by_samples"], default="samples_by_channels")
    p.add_argument("--realtime", default="1")
    p.add_argument("--loop", default="0")
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument("--sndhwm", type=int, default=8)
    p.add_argument("--status-every", type=int, default=20)
    p.add_argument("--stream-id", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    VariableChannelPlayer(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
