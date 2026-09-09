from __future__ import annotations

import json
from pathlib import Path

from neuro_signal_webapp.job_manager import JobManager


def test_job_manager_writes_raw_text_and_jsonl_logs(tmp_path: Path):
    manager = JobManager(tmp_path)
    job = manager.create_job("unit_test")

    manager.emit(job.job_id, "state", {
        "status": "running",
        "step": "Adapter selected",
        "percent": 25,
        "message": "Selected synthetic adapter.",
        "adapter": "synthetic",
        "detected_shape": [100, 4],
    })
    manager.emit(job.job_id, "state", {
        "status": "complete",
        "step": "Complete",
        "percent": 100,
        "message": "Done.",
        "output_dir": job.output_dir,
    })

    txt = Path(job.raw_log_path)
    jsonl = Path(job.raw_jsonl_path)
    assert txt.exists()
    assert jsonl.exists()

    text = txt.read_text(encoding="utf-8")
    assert "Neuro Signal raw job log" in text
    assert "Adapter selected" in text
    assert "Selected synthetic adapter" in text
    assert "status=complete" in text

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["status"] == "queued"
    assert any(row.get("adapter") == "synthetic" for row in rows)
    assert rows[-1]["status"] == "complete"
