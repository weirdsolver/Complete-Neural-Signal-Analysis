from __future__ import annotations

import argparse
import json
import signal
from typing import Optional

import numpy as np
import zmq

from .channel_manifest import ChannelManifest
from .spectral import compute_channel_metrics


class VariableSpectralAnalyzer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.running = True
        self.n = 0
        self.manifest: Optional[ChannelManifest] = None

    def stop(self, *_: object) -> None:
        self.running = False

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self.args.window_sub)
        sub.setsockopt(zmq.SUBSCRIBE, self.args.input_topic.encode("utf-8"))
        pub = ctx.socket(zmq.PUB)
        pub.bind(self.args.analysis_pub) if self.args.bind_pub else pub.connect(self.args.analysis_pub)
        out_topic = self.args.output_topic.encode("utf-8")
        poller = zmq.Poller(); poller.register(sub, zmq.POLLIN)
        print("Starting variable-channel spectral analyzer")
        print(f"  window SUB: {self.args.window_sub}")
        print(f"  analysis PUB: {self.args.analysis_pub}")
        print("  channel count: dynamic from stream metadata")
        while self.running:
            events = dict(poller.poll(250))
            if sub not in events:
                continue
            parts = sub.recv_multipart()
            if len(parts) < 3:
                continue
            header = json.loads(parts[1].decode("utf-8"))
            shape = header.get("shape")
            if not shape:
                continue
            x = np.frombuffer(parts[2], dtype=np.float32).reshape(int(shape[0]), int(shape[1]))
            manifest_data = header.get("channel_manifest") or {}
            manifest = ChannelManifest.from_dict(manifest_data, n_channels=x.shape[1])
            fs_hz = float(header.get("sample_rate_hz") or manifest.sample_rate_hz or self.args.fs)
            result = compute_channel_metrics(
                x,
                fs_hz,
                manifest.channel_names,
                manifest.groups,
                nperseg=int(self.args.nperseg),
                centroid_band=(float(self.args.centroid_low), float(self.args.centroid_high)),
                display_band=(float(self.args.display_low), float(self.args.display_high)),
            )
            out_header = {
                "type": "neural_spectral_analysis",
                "sequence_number": self.n,
                "window_sequence_number": header.get("sequence_number"),
                "start_time_sec": header.get("start_time_sec"),
                "sample_rate_hz": fs_hz,
                "n_channels": x.shape[1],
                "channel_manifest": manifest.to_dict(),
                "metrics_by_channel": result["metrics_by_channel"],
                "metrics_by_name": result["metrics_by_name"],
                "group_metrics": result["groups"],
                "summary": result["summary"],
                "psd_shape": list(map(int, result["psd"].shape)),
                "frequency_shape": [int(result["frequency_hz"].shape[0])],
            }
            pub.send_multipart([
                out_topic,
                json.dumps(out_header).encode("utf-8"),
                np.ascontiguousarray(result["frequency_hz"], dtype=np.float32).tobytes(),
                np.ascontiguousarray(result["psd"], dtype=np.float32).tobytes(order="C"),
            ])
            self.n += 1
            if self.n % int(self.args.status_every) == 0:
                print(f"analysis n={self.n} channels={x.shape[1]} mean_centroid={result['summary']['mean_centroid_hz']:.3f}Hz")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Variable-channel spectral analyzer")
    p.add_argument("--window-sub", default="tcp://127.0.0.1:5556")
    p.add_argument("--analysis-pub", default="tcp://127.0.0.1:5557")
    p.add_argument("--input-topic", default="eeg.window.1s")
    p.add_argument("--output-topic", default="eeg.analysis.spectral")
    p.add_argument("--fs", type=float, default=1000.0, help="Fallback fs")
    p.add_argument("--nperseg", type=int, default=500)
    p.add_argument("--centroid-low", type=float, default=2.0)
    p.add_argument("--centroid-high", type=float, default=45.0)
    p.add_argument("--display-low", type=float, default=1.0)
    p.add_argument("--display-high", type=float, default=60.0)
    p.add_argument("--bind-pub", action="store_true", default=True)
    p.add_argument("--status-every", type=int, default=20)
    return p


def main(argv: list[str] | None = None) -> int:
    VariableSpectralAnalyzer(build_arg_parser().parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
