from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_importer_neuromouse import write_neuromouse_dataset, build_neuromouse_comparison_pack


def make_recording(path: Path, n_channels: int = 8, fs: float = 1000.0):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(4096, n_channels)).astype('float32')
    np.save(path / 'signal.npy', signal)
    pd.DataFrame({'name': [f'e{i}' for i in range(n_channels)], 'type': ['MEA'] * n_channels}).to_csv(path / 'channels.csv', index=False)
    (path / 'metadata.json').write_text(json.dumps({'sampling_rate_hz': fs, 'modality': 'MEA'}))


def test_neuromouse_dataset_variable_channels(tmp_path: Path):
    rec = tmp_path / 'rec8'
    make_recording(rec, 8)
    out = tmp_path / 'sm'
    paths = write_neuromouse_dataset(rec, out, max_analysis_samples=4096, max_windows=10)
    data = json.loads(Path(paths['data_json']).read_text())
    assert data['meta']['n_channels'] == 8
    assert len(data['meta']['channels']) == 8
    assert len(data['welch_psd']['psd']) == 8
    assert len(data['geometry']['centroid']) == 8


def test_neuromouse_comparison_pack(tmp_path: Path):
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    make_recording(a, 8)
    make_recording(b, 12)
    result = build_neuromouse_comparison_pack([a], [b], output_dir=tmp_path / 'cmp', comparison_name='cmp')
    assert Path(result['comparison_manifest']).exists()
    assert Path(result['sessions_zip']).exists()
    manifest = json.loads(Path(result['comparison_manifest']).read_text())
    assert len(manifest['datasets']) == 2
