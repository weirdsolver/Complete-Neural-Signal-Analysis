from neuro_importer.progress import ProgressEvent
from neuro_signal_webapp.job_manager import JobManager


def test_webapp_progress_callback_accepts_stage_progress_event(tmp_path):
    jm = JobManager(tmp_path)
    record = jm.create_job("convert")
    cb = jm._progress_callback(record.job_id)
    cb(ProgressEvent(
        stage="Extract neural signal",
        status="running",
        step_index=4,
        total_steps=9,
        percent=44,
        message="Extracting neural signal",
    ))
    updated = jm.get(record.job_id)
    assert updated is not None
    assert updated.status == "running"
    assert updated.current_step == "Extract neural signal"
    assert updated.progress_percent == 44
    assert updated.events[-1]["stage"] == "Extract neural signal"
