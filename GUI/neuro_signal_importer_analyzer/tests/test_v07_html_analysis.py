from __future__ import annotations

import numpy as np
import pandas as pd

from neuro_importer_analysis import extract_features_from_signal, run_comparative_analysis


def test_v07_feature_extractor_variable_channels(tmp_path):
    x = np.random.default_rng(0).normal(size=(1000, 8)).astype('float32')
    df, summary = extract_features_from_signal(x, sampling_rate=1000, dataset_id='eight')
    assert len(df) == 8
    assert summary['n_channels'] == 8
    assert 'spectral_centroid_hz' in df.columns


def test_v07_comparative_analysis(tmp_path):
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    a.mkdir(); b.mkdir()
    np.save(a / 'signal.npy', np.random.default_rng(1).normal(size=(500, 4)).astype('float32'))
    np.save(b / 'signal.npy', np.random.default_rng(2).normal(size=(500, 6)).astype('float32'))
    pd.DataFrame({'name': [f'a{i}' for i in range(4)]}).to_csv(a / 'channels.csv', index=False)
    pd.DataFrame({'name': [f'b{i}' for i in range(6)]}).to_csv(b / 'channels.csv', index=False)
    (a / 'metadata.json').write_text('{"sampling_rate": 1000}')
    (b / 'metadata.json').write_text('{"sampling_rate": 1000}')
    out = tmp_path / 'cmp'
    result = run_comparative_analysis([a], [b], output_dir=out)
    assert (out / 'comparative_report.html').exists()
    assert (out / 'comparison_metrics.csv').exists()
    assert result['metadata']['comparison_mode'] == 'feature_level'


def test_v07_webapp_imports():
    import neuro_signal_webapp.app_server as app_server
    assert app_server.app.title == 'Neuro Signal App'
