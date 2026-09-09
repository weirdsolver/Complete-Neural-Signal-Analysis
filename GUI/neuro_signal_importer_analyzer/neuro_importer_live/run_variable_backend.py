from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    log_path: Path


def _launch(name: str, module: str, args: list[str], log_dir: Path) -> ManagedProcess:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    f = log_path.open("w", encoding="utf-8")
    cmd = [sys.executable, "-m", module] + args
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"started {name}: pid={proc.pid} log={log_path}")
    return ManagedProcess(name, proc, log_path)


class VariableBackendLauncher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.processes: list[ManagedProcess] = []
        self.running = True

    def stop(self, *_: object) -> None:
        self.running = False
        for mp in reversed(self.processes):
            if mp.process.poll() is None:
                mp.process.terminate()
        time.sleep(0.5)
        for mp in reversed(self.processes):
            if mp.process.poll() is None:
                mp.process.kill()

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        a = self.args
        log_dir = Path(a.log_dir)
        self.processes.append(_launch("receiver", "neuro_importer_live.eeg_receiver_zmq_window_pub_variable", [
            "--raw-pull", a.raw_zmq,
            "--window-pub", a.window_zmq,
            "--window-sec", str(a.window_sec),
            "--fs", str(a.fs),
        ], log_dir))
        time.sleep(0.5)
        self.processes.append(_launch("analyzer", "neuro_importer_live.eeg_analyzer_zmq_variable", [
            "--window-sub", a.window_zmq,
            "--analysis-pub", a.analysis_zmq,
            "--fs", str(a.fs),
        ], log_dir))
        if not a.no_raw_bridge:
            self.processes.append(_launch("raw_bridge", "neuro_importer_live.eeg_raw_ws_bridge_variable", [
                "--window-sub", a.window_zmq,
                "--ws-port", str(a.raw_ws_port),
                "--downsample", str(a.raw_downsample),
            ], log_dir))
        if not a.no_spectral_bridge:
            self.processes.append(_launch("spectral_bridge", "neuro_importer_live.eeg_analysis_ws_bridge_variable", [
                "--analysis-sub", a.analysis_zmq,
                "--ws-port", str(a.spectral_ws_port),
            ], log_dir))
        time.sleep(0.5)
        player_args = [
            "--source", a.source,
            "--address", a.raw_zmq,
            "--fs", str(a.fs),
            "--chunk-sec", str(a.chunk_sec),
            "--channel-profile", a.channel_profile,
            "--orientation", a.orientation,
            "--realtime", "1" if a.realtime else "0",
        ]
        if a.channels_csv:
            player_args += ["--channels-csv", a.channels_csv]
        if a.metadata_json:
            player_args += ["--metadata-json", a.metadata_json]
        if a.units:
            player_args += ["--units", a.units]
        self.processes.append(_launch("player", "neuro_importer_live.eeg_player_zmq_variable", player_args, log_dir))
        print("Variable-electrode live backend running.")
        print(f"  Raw WebSocket:      ws://127.0.0.1:{a.raw_ws_port}")
        print(f"  Spectral WebSocket: ws://127.0.0.1:{a.spectral_ws_port}")
        print("Press Ctrl+C to stop.")
        while self.running:
            for mp in self.processes:
                rc = mp.process.poll()
                if rc is not None and mp.name != "player":
                    print(f"process {mp.name} exited with {rc}; see {mp.log_path}")
                    self.stop()
                    return rc or 1
            time.sleep(1.0)
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch variable-electrode live neural backend")
    p.add_argument("--source", required=True, help="Path to signal.npy")
    p.add_argument("--channels-csv", default=None)
    p.add_argument("--metadata-json", default=None)
    p.add_argument("--fs", type=float, default=1000.0)
    p.add_argument("--units", default=None)
    p.add_argument("--channel-profile", default="auto")
    p.add_argument("--orientation", choices=["samples_by_channels", "channels_by_samples"], default="samples_by_channels")
    p.add_argument("--chunk-sec", type=float, default=0.25)
    p.add_argument("--window-sec", type=float, default=1.0)
    p.add_argument("--realtime", action="store_true", default=True)
    p.add_argument("--raw-zmq", default="tcp://127.0.0.1:5555")
    p.add_argument("--window-zmq", default="tcp://127.0.0.1:5556")
    p.add_argument("--analysis-zmq", default="tcp://127.0.0.1:5557")
    p.add_argument("--raw-ws-port", type=int, default=8765)
    p.add_argument("--spectral-ws-port", type=int, default=8766)
    p.add_argument("--raw-downsample", type=int, default=2)
    p.add_argument("--no-raw-bridge", action="store_true")
    p.add_argument("--no-spectral-bridge", action="store_true")
    p.add_argument("--log-dir", default="live_logs")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    return VariableBackendLauncher(build_arg_parser().parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
