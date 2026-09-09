import numpy as np

from neuro_importer.adapters.dsamp_mat_adapter import DSampMatAdapter
from neuro_importer.export import CanonicalExporter, NeuralSignalExporter
from neuro_importer.validate import validate_recording


def test_exporters_write_neural_signal_outputs(tmp_path):
    raw = {
        "DSamp": {
            "EEGdata": np.vstack([np.arange(4), np.arange(4) + 10]),
            "fs": 1000,
            "time": np.arange(4) / 1000,
            "label": ["C3", "C4"],
            "nchan": 2,
            "npt": 4,
        }
    }
    rec = validate_recording(DSampMatAdapter().convert(raw))

    canonical = CanonicalExporter().export(rec, tmp_path / "canonical")
    neural = NeuralSignalExporter().export(rec, tmp_path / "neural")

    assert (tmp_path / "canonical" / "signal.npy").exists()
    assert (tmp_path / "canonical" / "channels.csv").exists()
    assert (tmp_path / "neural" / "eeg_df.csv").exists()
    assert (tmp_path / "neural" / "neural_signal.npy").exists()
    assert "signal" in canonical
    assert "eeg_df" in neural
