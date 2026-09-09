import numpy as np

from neuro_importer.adapters.dsamp_mat_adapter import DSampMatAdapter
from neuro_importer.validate import validate_recording


def test_dsamp_adapter_synthetic_channels_by_samples():
    raw = {
        "DSamp": {
            "EEGdata": np.vstack([
                np.arange(10),
                np.arange(10) + 100,
                np.arange(10) + 200,
                np.arange(10) + 300,
            ]),  # channels × samples
            "fs": 1000,
            "time": np.arange(10) / 1000,
            "label": ["C3", "C4", "BIP1", "RESP1"],
            "nchan": 4,
            "npt": 10,
            "Subj": "1",
            "triggers": [
                {"Time": 2, "EventDescription": "Stim Start", "StimType": "F5"},
                {"Time": 5, "EventDescription": "Stim Stop", "StimType": "F5"},
            ],
        }
    }

    rec = DSampMatAdapter().convert(raw, subject=1, session=1)
    validate_recording(rec)

    assert rec.signal.shape == (10, 2)  # BIP1 and RESP1 excluded
    assert rec.channel_names() == ["C3", "C4"]
    assert rec.sampling_rate == 1000
    assert rec.metadata["triggers_ignored"] is True
    assert any("triggers" in msg for msg in rec.quality.infos)


def test_dsamp_adapter_can_keep_aux_channels():
    raw = {
        "DSamp": {
            "EEGdata": np.vstack([np.arange(5), np.arange(5) + 10, np.arange(5) + 20]),
            "fs": 500,
            "time": np.arange(5) / 500,
            "label": ["C3", "BIP1", "RESP1"],
            "nchan": 3,
            "npt": 5,
        }
    }

    rec = DSampMatAdapter(include_aux=True).convert(raw)
    validate_recording(rec)

    assert rec.signal.shape == (5, 3)
    assert rec.channel_names() == ["C3", "BIP1", "RESP1"]
