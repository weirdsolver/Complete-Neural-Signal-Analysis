from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_importer_analysis import list_advanced_methods, run_advanced_method


def _make_recording(root: Path, n_samples: int = 7000, n_channels: int = 3) -> Path:
    fs = 1000.0
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(222)
    signals = []
    for i in range(n_channels):
        signals.append(np.sin(2 * np.pi * (6 + i * 3) * t) + 0.05 * rng.normal(size=t.size))
    np.save(root / "signal.npy", np.column_stack(signals).astype("float32"))
    pd.DataFrame({"name": ["Fp1", "Fp2", "Cz"][:n_channels]}).to_csv(root / "channels.csv", index=False)
    (root / "metadata.json").write_text(json.dumps({"sampling_rate_hz": fs}), encoding="utf-8")
    return root


def test_manual_mfdfa_spectrum_registered_and_gui() -> None:
    methods = {m["id"]: m for m in list_advanced_methods()}
    assert "manual_mfdfa_spectrum" in methods
    spec = methods["manual_mfdfa_spectrum"]
    mode_param = next(p for p in spec["parameters"] if p["name"] == "mode")
    assert mode_param["default"] == "ultra"
    assert mode_param["options"] == ["ultra", "fast", "balanced", "full"]

    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    server = Path("neuro_signal_webapp/app_server.py").read_text(encoding="utf-8")
    assert "mfdfaSpectrumTab" in html
    assert "Manual MFDFA Spectrum" in html
    assert "manual_mfdfa_spectrum" in js
    assert "renderMfdfaSpectrumPanel" in js
    assert "/mfdfa-spectrum" in server


def test_manual_mfdfa_spectrum_runs_and_saves_outputs(tmp_path: Path) -> None:
    root = _make_recording(tmp_path)
    result = run_advanced_method("manual_mfdfa_spectrum", root, {"mode": "ultra", "max_channels": 3})
    assert result["ok"] is True
    payload = result["result"]["manual_mfdfa_spectrum"]
    assert payload["summary"]["mode"] == "ultra"
    assert payload["summary"]["n_channels"] == 3
    assert payload["summary"]["q_count"] == 17
    assert len(payload["rows"]) == 3
    assert len(payload["plot_paths"]) == 4
    for plot in payload["plot_paths"]:
        assert Path(plot["path"]).exists()
    for path in payload["per_channel_npz"]:
        assert Path(path).exists()
    assert Path(payload["outputs"]["summary_csv"]).exists()
    assert Path(payload["outputs"]["summary_txt"]).exists()


def test_all_speed_mode_defaults_are_ultra() -> None:
    for method in list_advanced_methods():
        for param in method.get("parameters", []):
            if param.get("name") == "mode" and "ultra" in param.get("options", []):
                assert param.get("default") == "ultra", method["id"]
