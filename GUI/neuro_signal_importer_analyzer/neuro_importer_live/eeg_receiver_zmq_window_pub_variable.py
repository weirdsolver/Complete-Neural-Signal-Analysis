from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import zmq


@dataclass
class RingBuffer:
    window_samples: int
    n_channels: int
    buffer: np.ndarray
    write_pos: int = 0
    total_samples: int = 0

    @classmethod
    def create(cls, window_samples: int, n_channels: int) -> "RingBuffer":
        return cls(window_samples, n_channels, np.zeros((window_samples, n_channels), dtype=np.float32))

    def add(self, chunk: np.ndarray) -> None:
        if chunk.ndim != 2 or chunk.shape[1] != self.n_channels:
            raise ValueError(f"Chunk shape {chunk.shape} incompatible with n_channels={self.n_channels}")
        n = chunk.shape[0]
        if n >= self.window_samples:
            self.buffer[:, :] = chunk[-self.window_samples :, :]
            self.write_pos = 0
            self.total_samples += n
            return
        end = self.write_pos + n
        if end <= self.window_samples:
            self.buffer[self.write_pos:end, :] = chunk
        else:
            first = self.window_samples - self.write_pos
            self.buffer[self.write_pos:, :] = chunk[:first, :]
            self.buffer[: end % self.window_samples, :] = chunk[first:, :]
        self.write_pos = end % self.window_samples
        self.total_samples += n

    def latest(self) -> np.ndarray:
        if self.total_samples < self.window_samples:
            raise RuntimeError("Window not filled yet")
        return np.vstack([self.buffer[self.write_pos :, :], self.buffer[: self.write_pos, :]]).astype(np.float32, copy=False)


class VariableWindowPublisher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.running = True
        self.manifest: Optional[dict] = None
        self.fs_hz: Optional[float] = None
        self.n_channels: Optional[int] = None
        self.ring: Optional[RingBuffer] = None
        self.windows = 0

    def stop(self, *_: object) -> None:
        self.running = False

    def _init_stream(self, header: dict) -> None:
        self.fs_hz = float(header.get("sample_rate_hz") or self.args.fs)
        self.n_channels = int(header.get("n_channels") or header.get("shape", [0, self.args.channels])[1])
        self.manifest = header.get("channel_manifest") or {
            "n_channels": self.n_channels,
            "channel_names": [f"ch_{i:03d}" for i in range(self.n_channels)],
            "channel_namespace": "generated_numeric",
        }
        window_samples = max(1, int(round(float(self.args.window_sec) * self.fs_hz)))
        self.ring = RingBuffer.create(window_samples, self.n_channels)
        print("Initialized variable-channel receiver")
        print(f"  fs: {self.fs_hz:g} Hz")
        print(f"  channels: {self.n_channels}")
        print(f"  window: {window_samples} samples = {self.args.window_sec:g}s")
        print(f"  namespace: {self.manifest.get('channel_namespace')}")

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        ctx = zmq.Context.instance()
        pull = ctx.socket(zmq.PULL)
        pull.bind(self.args.raw_pull) if self.args.bind_pull else pull.connect(self.args.raw_pull)
        pub = ctx.socket(zmq.PUB)
        pub.bind(self.args.window_pub) if self.args.bind_pub else pub.connect(self.args.window_pub)
        topic = self.args.topic.encode("utf-8")
        poller = zmq.Poller(); poller.register(pull, zmq.POLLIN)
        print("Starting variable-channel receiver/window publisher")
        print(f"  raw PULL: {self.args.raw_pull}")
        print(f"  window PUB: {self.args.window_pub}")
        while self.running:
            events = dict(poller.poll(250))
            if pull not in events:
                continue
            msg = pull.recv_multipart()
            if len(msg) == 1:
                header = json.loads(msg[0].decode("utf-8"))
                if header.get("type") == "stream_metadata":
                    self._init_stream(header)
                continue
            header = json.loads(msg[0].decode("utf-8"))
            if self.ring is None:
                self._init_stream(header)
            assert self.ring is not None and self.fs_hz is not None and self.n_channels is not None
            shape = header.get("shape")
            if not shape or int(shape[1]) != self.n_channels:
                raise ValueError(f"Incoming chunk shape {shape} incompatible with initialized n_channels={self.n_channels}")
            chunk = np.frombuffer(msg[1], dtype=np.float32).reshape(int(shape[0]), int(shape[1]))
            self.ring.add(chunk)
            if self.ring.total_samples >= self.ring.window_samples:
                window = np.ascontiguousarray(self.ring.latest())
                out_header = {
                    "type": "neural_window",
                    "stream_id": header.get("stream_id"),
                    "sequence_number": self.windows,
                    "source_chunk_sequence": header.get("sequence_number"),
                    "start_sample": int(max(0, self.ring.total_samples - self.ring.window_samples)),
                    "start_time_sec": float(max(0, self.ring.total_samples - self.ring.window_samples) / self.fs_hz),
                    "sample_rate_hz": self.fs_hz,
                    "n_channels": self.n_channels,
                    "shape": list(map(int, window.shape)),
                    "dtype": "float32",
                    "window_sec": float(self.args.window_sec),
                    "channel_manifest": self.manifest,
                }
                pub.send_multipart([topic, json.dumps(out_header).encode("utf-8"), window.tobytes(order="C")])
                self.windows += 1
                if self.windows % int(self.args.status_every) == 0:
                    print(f"status windows={self.windows} total_samples={self.ring.total_samples} channels={self.n_channels}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Variable-channel receiver/window publisher")
    p.add_argument("--raw-pull", default="tcp://127.0.0.1:5555")
    p.add_argument("--window-pub", default="tcp://127.0.0.1:5556")
    p.add_argument("--topic", default="eeg.window.1s")
    p.add_argument("--fs", type=float, default=1000.0, help="Fallback fs if player metadata is missing")
    p.add_argument("--channels", type=int, default=32, help="Fallback channel count if metadata is missing")
    p.add_argument("--window-sec", type=float, default=1.0)
    p.add_argument("--bind-pull", action="store_true", default=False)
    p.add_argument("--bind-pub", action="store_true", default=True)
    p.add_argument("--status-every", type=int, default=20)
    return p


def main(argv: list[str] | None = None) -> int:
    VariableWindowPublisher(build_arg_parser().parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
