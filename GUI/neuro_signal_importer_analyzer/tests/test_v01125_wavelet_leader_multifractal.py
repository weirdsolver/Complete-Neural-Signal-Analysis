from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_importer_analysis import list_advanced_methods, run_advanced_method


def _make_recording(root: Path, n_samples: int = 4096, n_channels: int = 3) -> Path:
    fs = 512.0
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(225)
    signals = []
    for i in range(n_channels):
        walk = np.cumsum(0.02 * rng.normal(size=t.size))
        signals.append(np.sin(2 * np.pi * (6 + i * 2) * t) + walk + 0.04 * rng.normal(size=t.size))
    np.save(root / "signal.npy", np.column_stack(signals).astype("float32"))
    pd.DataFrame({"name": ["C3", "C4", "Pz"][:n_channels]}).to_csv(root / "channels.csv", index=False)
    (root / "metadata.json").write_text(json.dumps({"sampling_rate_hz": fs}), encoding="utf-8")
    return root


def test_wavelet_leader_registered_and_gui() -> None:
    methods = {m["id"]: m for m in list_advanced_methods()}
    assert "wavelet_leader_multifractal" in methods
    spec = methods["wavelet_leader_multifractal"]
    mode_param = next(p for p in spec["parameters"] if p["name"] == "mode")
    assert mode_param["default"] == "ultra"
    assert mode_param["options"] == ["ultra", "fast", "balanced", "full"]

    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    server = Path("neuro_signal_webapp/app_server.py").read_text(encoding="utf-8")
    assert "waveletLeaderTab" in html
    assert "Wavelet Leader Multifractal" in html
    assert "wavelet_leader_multifractal" in js
    assert "renderWaveletLeaderPanel" in js
    assert "/wavelet-leader" in server


def test_wavelet_leader_runs_all_available_channels_and_saves_outputs(tmp_path: Path) -> None:
    root = _make_recording(tmp_path)
    result = run_advanced_method(
        "wavelet_leader_multifractal",
        root,
        {"mode": "ultra", "max_channels": 3, "max_analysis_samples": 4096},
    )
    assert result["ok"] is True
    payload = result["result"]["wavelet_leader_multifractal"]
    assert payload["summary"]["mode"] == "ultra"
    assert payload["summary"]["n_channels"] == 3
    assert payload["summary"]["q_count"] == 17
    assert len(payload["rows"]) == 3
    assert len(payload["zeta_rows"]) >= 3 * 10
    assert len(payload["plot_paths"]) >= 5
    for plot in payload["plot_paths"]:
        assert Path(plot["path"]).exists()
    for path in payload["per_channel_npz"]:
        assert Path(path).exists()
    for key in ("summary_csv", "zeta_csv", "summary_txt"):
        assert Path(payload["outputs"][key]).exists()
