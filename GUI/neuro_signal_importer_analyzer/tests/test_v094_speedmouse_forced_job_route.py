from pathlib import Path


def test_neuromouse_job_route_exists_and_static_source_reads_injected_dataset():
    app_server = Path("neuro_signal_webapp/app_server.py").read_text()
    static_source = Path("neuro_signal_webapp/neuromouse/js/sources/static-source.js").read_text()
    job_manager = Path("neuro_signal_webapp/job_manager.py").read_text()
    app_js = Path("neuro_signal_webapp/static/app.js").read_text()
    assert "/neuromouse-job/{job_id}/" in app_server
    assert "NEURO_SIGNAL_BACKEND_DATASET" in app_server
    assert "NEURO_SIGNAL_BACKEND_DATASET?.datasetUrl" in static_source
    assert "/neuromouse-job/{job_id}/" in job_manager
    assert "/neuromouse/?dataset=/api/jobs/{job_id}/neuromouse/data.json" not in job_manager
    assert "window.location.assign" not in app_js
    assert "current page stays open" in app_js
