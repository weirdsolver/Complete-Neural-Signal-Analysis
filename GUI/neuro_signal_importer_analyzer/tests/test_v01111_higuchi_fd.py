from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from neuro_importer_analysis import list_advanced_methods, run_advanced_method


def _make_eeg_recording(tmp_path: Path) -> Path:
    root = tmp_path / "eeg_recording"
    root.mkdir()
    fs = 250.0
    t = np.arange(0, 24, 1 / fs)
    rng = np.random.default_rng(1111)
    channels = ["Fp1", "Fp2", "C3", "C4", "O1", "O2"]
    signal = []
    for idx, _ch in enumerate(channels):
        trace = np.sin(2 * np.pi * (7 + idx) * t) + 0.25 * rng.normal(size=t.size)
        signal.append(trace)
    np.save(root / "signal.npy", np.asarray(signal, dtype=np.float32).T)
    (root / "metadata.json").write_text(json.dumps({"sampling_rate_hz": fs}), encoding="utf-8")
    pd.DataFrame({"name": channels}).to_csv(root / "channels.csv", index=False)
    return root


def test_higuchi_method_registered_and_runs(tmp_path: Path):
    rec = _make_eeg_recording(tmp_path)
    method_ids = {method["id"] for method in list_advanced_methods()}
    assert "higuchi_fractal_dimension" in method_ids

    payload = run_advanced_method(
        "higuchi_fractal_dimension",
        rec,
        {
            "mode": "ultrafast",
            "rolling": "true",
            "max_samples": 1500,
            "rolling_max_windows": 6,
        },
    )
    assert payload["ok"] is True
    result = payload["result"]["higuchi_fractal_dimension"]
    assert result["summary"]["n_channels"] == 6
    assert result["rows"]
    assert result["fit_diagnostics"]
    assert result["curves"]
    assert result["scalp_layout"]["points"]
    assert result["regional_summary"]
    assert result["asymmetry"]
    assert result["rolling"]["enabled"] is True
    assert result["rolling"]["matrix"]
    assert Path(result["outputs"]["summary_csv"]).exists()
    assert Path(result["outputs"]["fit_diagnostics_csv"]).exists()


def test_higuchi_api_returns_frontend_safe_json(tmp_path: Path):
    import neuro_signal_webapp.app_server as app_server
    from neuro_signal_webapp.job_manager import JobManager

    rec = _make_eeg_recording(tmp_path)
    app_server.manager = JobManager(tmp_path / "workspace")
    client = TestClient(app_server.app)

    response = client.post(
        "/api/advanced-methods/run",
        json={
            "method_id": "higuchi_fractal_dimension",
            "recording_dir": str(rec),
            "params": {"mode": "ultrafast", "rolling_max_windows": 4, "max_samples": 1200},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["result"]["higuchi_fractal_dimension"]["summary"]["mode"] == "ultrafast"
    assert Path(payload["saved_result_json"]).exists()


def test_higuchi_frontend_renderer_is_packaged():
    app_js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    index_html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    assert "renderHiguchiFdPlots" in app_js
    assert "higuchi_fractal_dimension" in app_js
    assert "0.11.32-neuromouse-choice" in index_html
