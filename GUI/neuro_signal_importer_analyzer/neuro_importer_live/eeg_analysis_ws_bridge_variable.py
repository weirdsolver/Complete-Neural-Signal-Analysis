from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import websockets
import zmq
import zmq.asyncio


@dataclass
class AnalysisBridge:
    args: argparse.Namespace
    clients: set = field(default_factory=set)
    latest_json: Optional[str] = None

    async def ws_handler(self, websocket):
        self.clients.add(websocket)
        if self.latest_json is not None:
            await websocket.send(self.latest_json)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, text: str) -> None:
        dead = []
        for client in list(self.clients):
            try:
                await client.send(text)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)

    async def zmq_loop(self) -> None:
        ctx = zmq.asyncio.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self.args.analysis_sub)
        sub.setsockopt(zmq.SUBSCRIBE, self.args.topic.encode("utf-8"))
        while True:
            parts = await sub.recv_multipart()
            if len(parts) < 2:
                continue
            header = json.loads(parts[1].decode("utf-8"))
            payload = dict(header)
            if self.args.include_psd_json and len(parts) >= 4:
                fshape = payload.get("frequency_shape", [0])
                pshape = payload.get("psd_shape", [0, 0])
                freq = np.frombuffer(parts[2], dtype=np.float32, count=int(fshape[0])).tolist()
                psd = np.frombuffer(parts[3], dtype=np.float32).reshape(int(pshape[0]), int(pshape[1])).tolist()
                payload["frequency_hz"] = freq
                payload["psd"] = psd
            text = json.dumps(payload, separators=(",", ":"))
            self.latest_json = text
            await self.broadcast(text)

    async def run(self) -> None:
        print("Starting variable-channel spectral WebSocket bridge")
        print(f"  ZMQ SUB: {self.args.analysis_sub}")
        print(f"  WebSocket: ws://{self.args.ws_host}:{self.args.ws_port}")
        async with websockets.serve(self.ws_handler, self.args.ws_host, int(self.args.ws_port), max_size=None):
            await self.zmq_loop()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Variable-channel spectral WebSocket bridge")
    p.add_argument("--analysis-sub", default="tcp://127.0.0.1:5557")
    p.add_argument("--topic", default="eeg.analysis.spectral")
    p.add_argument("--ws-host", default="0.0.0.0")
    p.add_argument("--ws-port", type=int, default=8766)
    p.add_argument("--include-psd-json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    asyncio.run(AnalysisBridge(build_arg_parser().parse_args(argv)).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
