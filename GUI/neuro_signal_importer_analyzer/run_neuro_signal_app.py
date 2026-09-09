#!/usr/bin/env python3
"""
One-command launcher for the Neuro Signal App.

Normal use:
    python3 run_neuro_signal_app.py

Default behavior:
  1. Checks whether this project is already installed for the current Python.
  2. Checks whether the required backend, frontend, live, and IO libraries import.
  3. Installs only if something is missing or the editable install points elsewhere.
  4. Starts the local FastAPI/HTML app.
  5. Waits until the health endpoint responds.
  6. Opens the app in the default browser. NeuroMouse opens automatically after analysis.

It does NOT reinstall on every run. Use --force-install only when you intentionally
want to reinstall/refresh dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
INSTALL_MARKER = PROJECT_ROOT / ".neuro_signal_app_install.json"
INSTALL_EXTRA = ".[all,frontend,live,dev]"
APP_VERSION = "0.11.33"

# Modules we expect after the normal full install. These cover conversion,
# HTML frontend, live backend, HDF5/MAT/EDF/NWB support, and optional zarr export.
REQUIRED_IMPORTS = {
    "neuro_importer": "local conversion backend",
    "neuro_signal_webapp": "local HTML/FastAPI app",
    "neuro_importer_live": "local live backend",
    "numpy": "numeric arrays",
    "pandas": "tables/metadata",
    "scipy": "signal processing/MAT support",
    "yaml": "YAML config/mappings",
    "h5py": "HDF5/NWB-like files",
    "mat73": "MATLAB v7.3 files",
    "openpyxl": "Excel metadata files",
    "mne": "EDF/BDF/FIF/EEG IO",
    "tables": "PyTables/HDF5 support",
    "pynwb": "NWB support",
    "zarr": "chunked large-file export",
    "fastapi": "local web API",
    "uvicorn": "local web server",
    "multipart": "browser file uploads",
    "zmq": "ZeroMQ live streaming",
    "websockets": "live browser streams",
}

LOCAL_MODULES = ["neuro_importer", "neuro_signal_webapp", "neuro_importer_live"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_marker() -> dict[str, Any] | None:
    if not INSTALL_MARKER.exists():
        return None
    try:
        return json.loads(INSTALL_MARKER.read_text())
    except Exception:
        return None


def write_marker() -> None:
    marker = {
        "python_executable": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "pyproject_sha256": sha256_file(PYPROJECT) if PYPROJECT.exists() else None,
        "install_extra": INSTALL_EXTRA,
        "installed_at_unix": time.time(),
    }
    INSTALL_MARKER.write_text(json.dumps(marker, indent=2))


def import_points_to_project(module_name: str) -> bool:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        return False
    try:
        origin = Path(spec.origin).resolve()
        return PROJECT_ROOT in origin.parents or origin == PROJECT_ROOT
    except Exception:
        return False


def dependency_check() -> tuple[bool, str]:
    missing: list[str] = []
    for mod, purpose in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(f"{mod} ({purpose})")
    if missing:
        return False, "missing required imports: " + ", ".join(missing)

    wrong_location: list[str] = []
    for mod in LOCAL_MODULES:
        if not import_points_to_project(mod):
            wrong_location.append(mod)
    if wrong_location:
        return False, "local package imports point outside this project folder: " + ", ".join(wrong_location)

    return True, "dependency and local editable install checks passed"


def install_needed(force: bool = False) -> tuple[bool, str]:
    if force:
        return True, "forced reinstall requested"

    ok, reason = dependency_check()
    if not ok:
        return True, reason

    marker = read_marker()
    if marker is None:
        # Important: do not reinstall just because the marker is missing.
        # If imports are good and point to this folder, recreate the marker and run.
        write_marker()
        return False, "installed already; marker was missing and has been recreated"

    if marker.get("python_executable") != sys.executable:
        return True, "current Python differs from the Python used for prior install"

    if PYPROJECT.exists() and marker.get("pyproject_sha256") != sha256_file(PYPROJECT):
        return True, "pyproject.toml changed since prior install"

    return False, "already installed"


def run_install() -> None:
    print("Installing Neuro Signal App and required libraries for this Python environment...")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Python:  {sys.executable}")
    cmd = [sys.executable, "-m", "pip", "install", "-e", INSTALL_EXTRA]
    print("Command:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))
    write_marker()
    print("Install complete. Future launches will skip reinstall unless something changes.\n")


def health_json(url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:
            if not (200 <= resp.status < 300):
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def health_ok(url: str) -> bool:
    return health_json(url) is not None


def wait_for_health(health_url: str, timeout_sec: float = 60.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        if health_ok(health_url):
            return True
        time.sleep(0.25)
    return False


def urls_for_open_mode(base_url: str, open_mode: str) -> list[str]:
    """Return browser URLs for the requested launch target.

    app        -> our launcher/dashboard
    neuromouse -> original NeuroMouse workbench
    both       -> both tabs, app first then NeuroMouse
    """
    app_url = base_url
    neuromouse_url = f"{base_url}/neuromouse/"
    if open_mode == "app":
        return [app_url]
    if open_mode == "neuromouse":
        return [neuromouse_url]
    return [app_url, neuromouse_url]


def open_requested_pages(base_url: str, open_mode: str) -> None:
    for url in urls_for_open_mode(base_url, open_mode):
        print(f"Opening: {url}")
        webbrowser.open(url)
        # Give the browser a short moment so two tabs do not race each other.
        time.sleep(0.35)


def start_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "neuro_signal_webapp.app_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--no-browser",
    ]
    if args.workspace:
        cmd.extend(["--workspace", args.workspace])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print("Starting Neuro Signal App server...")
    print("Command:", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check/install dependencies if needed, then launch the HTML Neuro Signal App.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--force-install", action="store_true", help="Force pip install even if checks pass.")
    parser.add_argument("--no-browser", action="store_true", help="Start server but do not open browser.")
    parser.add_argument(
        "--open",
        choices=["app", "neuromouse", "both"],
        default="app",
        help="Which browser page to open after the server starts. Default: app. NeuroMouse opens automatically after Analyze in NeuroMouse completes.",
    )
    parser.add_argument("--skip-install-check", action="store_true", help="Skip checks and try to run directly.")
    args = parser.parse_args()

    if not PYPROJECT.exists():
        print(f"ERROR: pyproject.toml not found at {PYPROJECT}", file=sys.stderr)
        return 2

    if not args.skip_install_check:
        needed, reason = install_needed(args.force_install)
        if needed:
            print(f"Install check: install needed ({reason}).")
            try:
                run_install()
            except subprocess.CalledProcessError as exc:
                print(f"ERROR: install failed with exit code {exc.returncode}", file=sys.stderr)
                return exc.returncode or 1
        else:
            print(f"Install check: {reason}. Skipping pip install.")

    url = f"http://{args.host}:{args.port}"
    health_url = f"{url}/api/health"

    existing_health = health_json(health_url)
    if existing_health:
        existing_version = str(existing_health.get("version", "unknown"))
        if existing_version != APP_VERSION:
            print(f"ERROR: A Neuro Signal server is already running at {url}, but it reports version {existing_version}; this launcher is version {APP_VERSION}.", file=sys.stderr)
            print("Stop the old server first, or launch this version on another port, for example:", file=sys.stderr)
            print("  pkill -f neuro_signal_webapp.app_server", file=sys.stderr)
            print(f"  {sys.executable} run_neuro_signal_app.py --force-install", file=sys.stderr)
            print("or:", file=sys.stderr)
            print(f"  {sys.executable} run_neuro_signal_app.py --port 8790", file=sys.stderr)
            return 3
        print(f"Neuro Signal App v{existing_version} is already running at {url}")
        if not args.no_browser:
            open_requested_pages(url, args.open)
        return 0

    proc = start_server(args)
    try:
        if wait_for_health(health_url):
            print(f"Neuro Signal App is ready: {url}")
            if not args.no_browser:
                open_requested_pages(url, args.open)
        else:
            print("WARNING: server did not respond to health check yet.")
            print(f"Try opening manually: {url}")

        print("Press Ctrl+C to stop the server.")
        return proc.wait()
    except KeyboardInterrupt:
        print("\nStopping Neuro Signal App server...")
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
