from pathlib import Path

from neuro_signal_webapp.job_manager import discover_primary_signal_files, is_primary_signal_file


def test_fdt_is_not_primary_and_set_is_primary(tmp_path: Path):
    set_path = tmp_path / "sub-001_task-Rest_eeg.set"
    fdt_path = tmp_path / "sub-001_task-Rest_eeg.fdt"
    set_path.write_text("header")
    fdt_path.write_bytes(b"signal")

    assert is_primary_signal_file(set_path)
    assert not is_primary_signal_file(fdt_path)

    discovery = discover_primary_signal_files([fdt_path, set_path])
    primary = [p.name for p in discovery["primary_files"]]
    assert primary == ["sub-001_task-Rest_eeg.set"]
    assert any(item["path"].endswith(".fdt") for item in discovery["skipped"])
    assert any(item["sidecar"].endswith(".fdt") for item in discovery["promoted"])


def test_bids_sidecars_are_skipped(tmp_path: Path):
    edf = tmp_path / "chb01_01.edf"
    channels = tmp_path / "sub-001_task-Rest_channels.tsv"
    eeg_json = tmp_path / "sub-001_task-Rest_eeg.json"
    edf.write_text("edf")
    channels.write_text("name\ttype\nCz\tEEG\n")
    eeg_json.write_text("{}")

    discovery = discover_primary_signal_files([tmp_path])
    primary = [p.name for p in discovery["primary_files"]]
    skipped = [Path(item["path"]).name for item in discovery["skipped"]]
    assert primary == ["chb01_01.edf"]
    assert "sub-001_task-Rest_channels.tsv" in skipped
    assert "sub-001_task-Rest_eeg.json" in skipped
