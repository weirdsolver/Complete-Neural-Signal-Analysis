from pathlib import Path

from fastapi.testclient import TestClient

from neuro_signal_webapp.app_server import app, manager


def test_persisted_neuromouse_data_endpoint_works_after_restart_without_job_record(tmp_path):
    job_id = "persisted_job_0108"
    root = manager.workspace / "outputs" / job_id / "neuromouse" / "001_demo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.json").write_text('{"meta":{"name":"persisted"}}', encoding="utf-8")

    client = TestClient(app)
    response = client.get(f"/api/jobs/{job_id}/neuromouse/data.json")
    assert response.status_code == 200
    assert response.json()["meta"]["name"] == "persisted"


def test_persisted_neuromouse_manifest_endpoint_recovers_from_disk(tmp_path):
    job_id = "persisted_manifest_0108"
    root = manager.workspace / "outputs" / job_id / "neuromouse" / "001_demo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.json").write_text('{"meta":{"name":"persisted"}}', encoding="utf-8")

    client = TestClient(app)
    response = client.get(f"/api/jobs/{job_id}/neuromouse/manifest.json")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "persisted"
    assert payload["result"]["primary_neuromouse_dataset_url"] == f"/api/jobs/{job_id}/neuromouse/data.json"
