from __future__ import annotations

import numpy as np

from neuro_importer.pipeline import NeuroImportPipeline
from neuro_importer.progress import ProgressEvent


def test_pipeline_emits_stage_based_progress_events(tmp_path):
    source = tmp_path / "signal.npy"
    np.save(source, np.arange(100, dtype=float).reshape(25, 4))
    events: list[ProgressEvent] = []

    NeuroImportPipeline(load_plugins=False).convert(
        source,
        output_dir=tmp_path / "out",
        sampling_rate=1000,
        export_config={"save_signal_csv": False},
        progress_callback=events.append,
    )

    assert events
    assert events[-1].stage == "Complete"
    assert events[-1].status == "complete"
    assert events[-1].percent == 100
    completed_stages = {e.stage for e in events if e.status == "complete"}
    assert "Read source file" in completed_stages
    assert "Extract neural signal" in completed_stages
    assert "Export canonical files" in completed_stages
