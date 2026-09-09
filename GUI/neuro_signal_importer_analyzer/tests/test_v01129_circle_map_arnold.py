from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_importer_analysis import list_advanced_methods, run_advanced_method


def _make_recording(root: Path, n_samples: int = 512, n_channels: int = 4) -> Path:
    fs = 128.0
    t = np.arange(n_samples) / fs
    sig = np.column_stack([
        np.sin(2 * np.pi * (4 + i) * t + 0.2 * i) + 0.03 * np.cos(2 * np.pi * 11 * t)
        for i in range(n_channels)
    ]).astype("float32")
    np.save(root / "signal.npy", sig)
    pd.DataFrame({"name": ["Fp1", "Fp2", "C3", "C4"][:n_channels]}).to_csv(root / "channels.csv", index=False)
    (root / "metadata.json").write_text(json.dumps({"sampling_rate_hz": fs}), encoding="utf-8")
    return root


def test_circle_map_arnold_registered_and_gui() -> None:
    methods = {m["id"]: m for m in list_advanced_methods()}
    assert "circle_map_arnold_tongues" in methods
    spec = methods["circle_map_arnold_tongues"]
    mode_param = next(p for p in spec["parameters"] if p["name"] == "mode")
    assert mode_param["default"] == "ultra"
    assert mode_param["options"] == ["ultra", "fast", "balanced", "full"]

    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    server = Path("neuro_signal_webapp/app_server.py").read_text(encoding="utf-8")
    assert "circleMapArnoldTab" in html
    assert "Circle Map Arnold Tongues" in html
    assert "circle_map_arnold_tongues" in js
    assert "renderCircleMapArnoldPanel" in js
    assert "/circle-map" in server


def test_circle_map_arnold_runs_all_available_channels_and_saves_outputs(tmp_path: Path) -> None:
    rec = _make_recording(tmp_path)
    result = run_advanced_method(
        "circle_map_arnold_tongues",
        rec,
        {
            "mode": "ultra",
            "n_omega": 4,
            "n_K": 4,
            "iterations": 4,
            "max_phase_samples": 64,
            "max_channels": 4,
            "max_analysis_samples": 512,
        },
    )
    assert result["ok"] is True
    payload = result["result"]["circle_map_arnold_tongues"]
    assert payload["summary"]["mode"] == "ultra"
    assert payload["summary"]["n_channels"] == 4
    assert payload["summary"]["grid_points"] == 16
    assert payload["summary"]["phase_samples_used"] <= 64
    assert payload["summary"]["plot_count"] >= 7
    assert "proportion_fixed_point_locked" in payload["grid_rows"][0]
    assert "phase_cycles" in payload["phase_rows"][0]
    for key in ("npz", "grid_csv", "phase_csv", "summary_txt"):
        assert Path(payload["outputs"][key]).exists()
    for plot in payload["plot_paths"]:
        assert Path(plot["path"]).exists()
