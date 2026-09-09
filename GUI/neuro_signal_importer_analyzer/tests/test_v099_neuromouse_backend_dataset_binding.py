from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from neuro_signal_webapp.job_manager import JobManager


def test_normal_convert_job_generates_neuromouse_dataset_from_converted_data(tmp_path: Path):
    source = tmp_path / "uploaded_signal.npy"
    np.save(source, np.random.default_rng(99).normal(size=(512, 5)).astype("float32"))

    manager = JobManager(tmp_path / "workspace")
    job = manager.start_convert_job(
        [source],
        options={"sampling_rate": 1000, "neuromouse_max_analysis_samples": 512, "neuromouse_max_windows": 5},
    )

    deadline = time.time() + 15
    while manager.get(job.job_id).status not in {"complete", "failed"} and time.time() < deadline:
        time.sleep(0.05)

    record = manager.get(job.job_id)
    assert record.status == "complete", record.error
    result = record.result or {}
    assert result["primary_neuromouse_url"].startswith(f"/neuromouse-job/{job.job_id}/")
    assert result["primary_neuromouse_dataset_url"] == f"/api/jobs/{job.job_id}/neuromouse/data.json"
    assert result["neuromouse_datasets"]

    data_json = Path(result["neuromouse_datasets"][0]["data_json"])
    assert data_json.exists()
    data = json.loads(data_json.read_text(encoding="utf-8"))
    assert data["meta"]["dataset_id"] == "uploaded_signal"
    assert data["meta"]["n_channels"] == 5
    assert data["meta"]["source_signal_path"].endswith("signal.npy")


def test_neuromouse_job_page_hard_binds_backend_data_and_disables_demo_fallback():
    app_server = Path("neuro_signal_webapp/app_server.py").read_text(encoding="utf-8")
    static_source = Path("neuro_signal_webapp/neuromouse/js/sources/static-source.js").read_text(encoding="utf-8")
    app_js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")

    assert "disableDemoFallback" in app_server
    assert "window.fetch = function" in app_server
    assert "NEURO_SIGNAL_LAST_BACKEND_DATASET_URL" in app_server
    assert "NEURO_SIGNAL_LAST_BACKEND_DATASET_URL" in static_source
    assert "?demo=1" in static_source
    assert "backend-dataset mode" in static_source
    assert "neuromouse-job" in app_js
    assert "primary_neuromouse_dataset_url" in app_js
