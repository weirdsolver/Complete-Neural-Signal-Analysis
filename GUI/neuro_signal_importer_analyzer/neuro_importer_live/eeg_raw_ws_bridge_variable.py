from __future__ import annotations

import argparse
import asyncio
import json
import signal
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import websockets
import zmq
import zmq.asyncio

from .ws_utils import pack_binary_float32_frame


@dataclass
class RawBridge:
    args: argparse.Namespace
    clients: set = field(default_factory=set)
    latest_frame: Optional[bytes] = None
    running: bool = True

    async def ws_handler(self, websocket):
        self.clients.add(websocket)
        if self.latest_frame is not None:
            await websocket.send(self.latest_frame)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, frame: bytes) -> None:
        dead = []
        for client in list(self.clients):
            try:
                await client.send(frame)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)

    async def zmq_loop(self) -> None:
        ctx = zmq.asyncio.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self.args.window_sub)
        sub.setsockopt(zmq.SUBSCRIBE, self.args.topic.encode("utf-8"))
        while self.running:
            parts = await sub.recv_multipart()
            if len(parts) < 3:
                continue
            header = json.loads(parts[1].decode("utf-8"))
            shape = header.get("shape")
            if not shape:
                continue
            x = np.frombuffer(parts[2], dtype=np.float32).reshape(int(shape[0]), int(shape[1]))
            down = max(1, int(self.args.downsample))
            out = np.ascontiguousarray(x[::down, :], dtype=np.float32)
            out_header = dict(header)
            out_header.update({
                "type": "raw_neural_window_binary",
                "shape": list(map(int, out.shape)),
                "sample_rate_hz": float(header.get("sample_rate_hz", self.args.fs)) / down,
                "downsample_factor": down,
            })
            frame = pack_binary_float32_frame(out_header, out)
            self.latest_frame = frame
            await self.broadcast(frame)

    async def run(self) -> None:
        print("Starting variable-channel raw WebSocket bridge")
        print(f"  ZMQ SUB: {self.args.window_sub}")
        print(f"  WebSocket: ws://{self.args.ws_host}:{self.args.ws_port}")
        async with websockets.serve(self.ws_handler, self.args.ws_host, int(self.args.ws_port), max_size=None):
            await self.zmq_loop()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Variable-channel raw WebSocket bridge")
    p.add_argument("--window-sub", default="tcp://127.0.0.1:5556")
    p.add_argument("--topic", default="eeg.window.1s")
    p.add_argument("--ws-host", default="0.0.0.0")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--fs", type=float, default=1000.0)
    return p


def main(argv: list[str] | None = None) -> int:
    bridge = RawBridge(build_arg_parser().parse_args(argv))
    asyncio.run(bridge.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
