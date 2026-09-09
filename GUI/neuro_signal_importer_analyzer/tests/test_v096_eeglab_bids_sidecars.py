from __future__ import annotations

from pathlib import Path

from neuro_importer.detect import detect_file_type
from neuro_signal_webapp.app_server import _safe_upload_target, _primary_upload_paths
from neuro_signal_webapp.job_manager import is_primary_signal_file


def test_eeglab_and_brainvision_primary_files_are_detected():
    assert detect_file_type('sub-001_task-Rest_eeg.set') == 'mne'
    assert detect_file_type('sub-001_task-Rest_eeg.vhdr') == 'mne'
    assert detect_file_type('recording.fif') == 'mne'
    assert detect_file_type('recording.fif.gz') == 'mne'


def test_eeglab_bids_sidecars_are_not_converted_as_standalone_recordings():
    assert is_primary_signal_file('sub-001_task-Rest_eeg.set')
    assert not is_primary_signal_file('sub-001_task-Rest_eeg.fdt')
    assert not is_primary_signal_file('sub-001_task-Rest_channels.tsv')
    assert not is_primary_signal_file('sub-001_task-Rest_electrodes.tsv')
    assert not is_primary_signal_file('sub-001_task-Rest_events.tsv')
    assert not is_primary_signal_file('sub-001_task-Rest_eeg.json')
    assert is_primary_signal_file('plain_signal_table.tsv')


def test_folder_upload_relative_paths_are_preserved_safely(tmp_path: Path):
    root = tmp_path / 'uploads'
    root.mkdir()
    target = _safe_upload_target(root, 'sub-001/eeg/sub-001_task-Rest_eeg.set')
    assert target == root / 'sub-001' / 'eeg' / 'sub-001_task-Rest_eeg.set'

    traversal = _safe_upload_target(root, '../../evil.set')
    assert traversal.parent == root
    assert traversal.name == 'evil.set'


def test_upload_handler_converts_only_primary_files_and_keeps_sidecars(tmp_path: Path):
    root = tmp_path / 'uploads'
    root.mkdir()
    set_file = root / 'sub-001_task-Rest_eeg.set'
    fdt_file = root / 'sub-001_task-Rest_eeg.fdt'
    channels = root / 'sub-001_task-Rest_channels.tsv'
    edf_file = root / 'chb01_01.edf'
    for p in (set_file, fdt_file, channels, edf_file):
        p.write_bytes(b'test')

    selected = _primary_upload_paths([str(fdt_file), str(edf_file), str(channels), str(set_file)])

    assert str(set_file) in selected
    assert str(edf_file) in selected
    assert str(fdt_file) not in selected
    assert str(channels) not in selected
    assert selected == sorted(selected, key=lambda x: str(x).lower())
