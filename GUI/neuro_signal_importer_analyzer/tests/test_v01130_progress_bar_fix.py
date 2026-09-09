from pathlib import Path

from neuro_signal_webapp.job_manager import JobManager


def test_job_progress_percent_is_clamped_and_terminal_is_full(tmp_path):
    manager = JobManager(tmp_path)
    record = manager.create_job("unit")

    manager.emit(record.job_id, "progress", {"status": "running", "step": "bad high", "percent": 250})
    assert manager.get(record.job_id).progress_percent == 99

    manager.emit(record.job_id, "progress", {"status": "running", "step": "half", "step_index": 2, "total_steps": 4})
    # Running progress is monotonic at the job level, so nested steps cannot
    # move the loader backward.
    assert manager.get(record.job_id).progress_percent == 99

    manager.emit(record.job_id, "state", {"status": "complete", "step": "Done", "percent": 0})
    assert manager.get(record.job_id).progress_percent == 100
    assert manager.get(record.job_id).status == "complete"


def test_frontend_progress_bar_renders_percent_text_and_terminal_heartbeats():
    app_js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    styles = Path("neuro_signal_webapp/static/styles.css").read_text(encoding="utf-8")

    assert "normalizeProgressPercent" in app_js
    assert "Full (100%)" in app_js
    assert "data.type === 'heartbeat' && !data.status" in app_js
    assert "Completed step: ${data.step || 'Complete'} (100%)" in app_js
    assert "aria-valuenow" in app_js
    assert "line-height: 22px" in styles


def test_nested_progress_complete_does_not_finish_overall_job(tmp_path):
    manager = JobManager(tmp_path)
    record = manager.create_job("convert")
    manager.emit(record.job_id, "state", {"status": "running", "step": "Starting", "percent": 1})
    manager.emit(record.job_id, "progress", {"status": "complete", "step": "Read source file", "percent": 100})

    saved = manager.get(record.job_id)
    assert saved.status == "running"
    assert saved.progress_percent == 100

    manager.emit(record.job_id, "progress", {"status": "running", "step": "Later nested step", "percent": 10})
    assert manager.get(record.job_id).progress_percent == 100


def test_conversion_callback_maps_nested_step_percent_to_global_range(tmp_path):
    manager = JobManager(tmp_path)
    record = manager.create_job("convert")
    cb = manager._progress_callback(record.job_id, percent_start=3, percent_end=44)

    cb({"status": "running", "stage": "Read source file", "percent": 10, "message": "Reading"})
    first = manager.get(record.job_id).events[-1]
    cb({"status": "complete", "stage": "Read source file", "percent": 100, "message": "Loaded"})
    second = manager.get(record.job_id).events[-1]

    assert first["percent"] < second["percent"] < 44
    assert second["status"] == "running"
    assert second["step_status"] == "complete"


def test_neuromouse_completion_does_not_auto_navigate_current_window():
    app_js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    assert "window.location.assign" not in app_js
    assert "current page stays open" in app_js
    assert "showNeuroMouseOpenLink(absolute.toString(), label)" in app_js
    assert "openBackendNeuroMouseUrl(r.primary_neuromouse_url" in app_js
