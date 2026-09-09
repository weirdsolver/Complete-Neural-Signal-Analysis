from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from neuro_importer_analysis import list_advanced_methods, run_advanced_method


def _make_recording(tmp_path: Path) -> Path:
    root = tmp_path / "recording"
    root.mkdir()
    fs = 1000.0
    rng = np.random.default_rng(111)
    signal = rng.normal(0.0, 0.08, size=(2000, 4))
    for ch in range(signal.shape[1]):
        for idx in (200 + ch * 11, 760 + ch * 7, 1320 + ch * 13):
            signal[idx, ch] -= 5.0
    np.save(root / "signal.npy", signal.astype("float32"))
    (root / "metadata.json").write_text(json.dumps({"sampling_rate_hz": fs}), encoding="utf-8")
    pd.DataFrame({"name": ["E1", "E2", "E3", "E4"]}).to_csv(root / "channels.csv", index=False)
    return root


def test_advanced_method_registry_contains_native_neuromouse_methods():
    method_ids = {method["id"] for method in list_advanced_methods()}
    assert {"band_power_summary", "spike_detect", "network_burst", "electrode_connectivity"}.issubset(method_ids)


def test_advanced_methods_run_on_converted_recording(tmp_path: Path):
    rec = _make_recording(tmp_path)
    band = run_advanced_method("band_power_summary", rec, {"band": "alpha", "max_analysis_samples": 2000})
    assert band["ok"] is True
    assert band["result"]["band_power_summary"]["rows"]

    spikes = run_advanced_method("spike_detect", rec, {"threshold_multiplier": 4, "polarity": "negative", "max_analysis_samples": 2000})
    assert spikes["ok"] is True
    assert spikes["result"]["spike_detect"]["summary"]["total_spikes"] >= 12

    bursts = run_advanced_method("network_burst", rec, {"threshold_multiplier": 4, "polarity": "negative", "threshold_count": 2, "max_analysis_samples": 2000})
    assert bursts["ok"] is True
    assert "timeline" in bursts["result"]["network_burst"]

    conn = run_advanced_method("electrode_connectivity", rec, {"threshold_multiplier": 4, "polarity": "negative", "max_analysis_samples": 2000})
    assert conn["ok"] is True
    matrix = conn["result"]["electrode_connectivity"]["matrix"]
    assert len(matrix) == 4
    assert len(matrix[0]) == 4


def test_advanced_method_api_lists_and_runs(tmp_path: Path):
    import neuro_signal_webapp.app_server as app_server
    from neuro_signal_webapp.job_manager import JobManager

    rec = _make_recording(tmp_path)
    app_server.manager = JobManager(tmp_path / "workspace")
    client = TestClient(app_server.app)

    listed = client.get("/api/advanced-methods")
    assert listed.status_code == 200
    assert any(method["id"] == "spike_detect" for method in listed.json()["methods"])

    response = client.post(
        "/api/advanced-methods/run",
        json={
            "method_id": "spike_detect",
            "recording_dir": str(rec),
            "params": {"threshold_multiplier": 4, "polarity": "negative", "max_analysis_samples": 2000},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert Path(payload["saved_result_json"]).exists()
    assert payload["result"]["spike_detect"]["summary"]["active_channels"] == 4
