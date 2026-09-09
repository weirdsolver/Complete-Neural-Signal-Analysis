import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_importer_analysis import list_advanced_methods, run_advanced_method


def test_v01115_minimal_dashboard_has_expected_primary_actions():
    html = Path('neuro_signal_webapp/static/index.html').read_text(encoding='utf-8')
    assert 'v0.11.32 — Minimal Dashboard' in html
    assert 'homeTab' in html
    assert 'Import dataset / datasets' in html
    assert 'Compare datasets' in html
    assert 'Open normal NeuroMouse analysis' not in html
    assert 'Run Higuchi FD' in html
    assert 'Run embedded FD' in html
    assert 'optionalTabs' in html
    assert 'More tools' in html


def test_v01114_embedded_fractal_dimension_method_runs(tmp_path):
    fs = 200.0
    n = 2500
    t = np.arange(n) / fs
    rng = np.random.default_rng(7)
    signal = np.column_stack([
        np.sin(2 * np.pi * 8 * t) + 0.05 * rng.normal(size=n),
        np.sin(2 * np.pi * 11 * t) + 0.05 * rng.normal(size=n),
        np.sin(2 * np.pi * 16 * t) + 0.05 * rng.normal(size=n),
    ])
    np.save(tmp_path / 'signal.npy', signal)
    pd.DataFrame({'name': ['Fp1', 'Fp2', 'Cz']}).to_csv(tmp_path / 'channels.csv', index=False)
    (tmp_path / 'metadata.json').write_text(json.dumps({'sampling_rate_hz': fs}), encoding='utf-8')

    ids = {m['id'] for m in list_advanced_methods()}
    assert 'embedded_fractal_dimension' in ids

    payload = run_advanced_method('embedded_fractal_dimension', tmp_path, {
        'mode': 'ultra',
        'embedding_dims': '2,3',
        'max_channels': 2,
        'tau_ms': 10,
        'random_seed': 0,
    })
    assert payload['ok'] is True
    result = payload['result']['embedded_fractal_dimension']
    assert result['summary']['ok_rows'] >= 2
    assert result['summary']['generated_embeddings'] >= 2
    assert result['mean_by_dimension']
    assert result['channel_summary']
    assert any(row['corr_dim_d2'] is not None for row in result['rows'])
