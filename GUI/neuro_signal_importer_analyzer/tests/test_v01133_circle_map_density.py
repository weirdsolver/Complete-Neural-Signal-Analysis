from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_importer_analysis import list_advanced_methods, run_advanced_method


def _make_recording(root: Path, n_samples: int = 512, n_channels: int = 4) -> Path:
    fs = 128.0
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(133)
    sig = np.column_stack([
        np.sin(2 * np.pi * (5 + i) * t + 0.2 * i) + 0.03 * rng.normal(size=n_samples)
        for i in range(n_channels)
    ]).astype("float32")
    np.save(root / "signal.npy", sig)
    pd.DataFrame({"name": ["Fp1", "Fp2", "C3", "C4"][:n_channels]}).to_csv(root / "channels.csv", index=False)
    (root / "metadata.json").write_text(json.dumps({"sampling_rate_hz": fs}), encoding="utf-8")
    return root


def test_circle_map_converged_density_registered_and_gui() -> None:
    methods = {m["id"]: m for m in list_advanced_methods()}
    assert "circle_map_converged_density" in methods
    spec = methods["circle_map_converged_density"]
    mode_param = next(p for p in spec["parameters"] if p["name"] == "mode")
    assert mode_param["default"] == "ultra"
    assert mode_param["options"] == ["ultra", "fast", "balanced", "full"]

    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    server = Path("neuro_signal_webapp/app_server.py").read_text(encoding="utf-8")
    assert "circleMapDensityTab" in html
    assert "Circle Map Converged Density" in html
    assert "circle_map_converged_density" in js
    assert "renderCircleMapDensityPanel" in js
    assert "/circle-density" in server


def test_circle_map_converged_density_runs_all_available_channels_and_saves_outputs(tmp_path: Path) -> None:
    rec = _make_recording(tmp_path)
    result = run_advanced_method(
        "circle_map_converged_density",
        rec,
        {
            "mode": "ultra",
            "n_K": 5,
            "iterations": 3,
            "n_bins": 8,
            "max_samples_per_channel": 64,
            "max_channels": 4,
            "max_analysis_samples": 512,
            "phase_max_hz": 45,
        },
    )
    assert result["ok"] is True
    payload = result["result"]["circle_map_converged_density"]
    assert payload["summary"]["mode"] == "ultra"
    assert payload["summary"]["n_channels"] == 4
    assert payload["summary"]["K_count"] == 5
    assert payload["summary"]["n_bins"] == 8
    assert payload["summary"]["plot_count"] >= 8
    assert len(payload["channel_rows"]) == 4
    assert "avg_probability" in payload["density_rows"][0]
    assert "normalized_concentration" in payload["aggregate_rows"][0]
    for key in ("npz", "density_csv", "summary_csv", "aggregate_csv", "per_channel_csv", "initial_phase_csv", "summary_txt"):
        assert Path(payload["outputs"][key]).exists()
    for plot in payload["plot_paths"]:
        assert Path(plot["path"]).exists()
