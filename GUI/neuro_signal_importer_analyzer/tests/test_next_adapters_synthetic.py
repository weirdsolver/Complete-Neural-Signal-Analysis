import numpy as np
import pandas as pd

from neuro_importer.adapters import EEGLABAdapter, FieldTripAdapter, GenericMatAdapter, NumpyArrayAdapter, TableAdapter
from neuro_importer.validate import validate_recording


def test_eeglab_adapter_synthetic_channels_by_samples():
    raw = {
        "EEG": {
            "data": np.vstack([np.arange(6), np.arange(6) + 10, np.arange(6) + 20]),
            "srate": 250,
            "times": np.arange(6) * 4,  # ms
            "chanlocs": [{"labels": "C3"}, {"labels": "C4"}, {"labels": "EOG1"}],
            "nbchan": 3,
            "pnts": 6,
            "trials": 1,
        }
    }
    rec = validate_recording(EEGLABAdapter().convert(raw))
    assert rec.signal.shape == (6, 2)
    assert rec.channel_names() == ["C3", "C4"]
    assert rec.sampling_rate == 250
    assert np.isclose(rec.time[1], 0.004)


def test_fieldtrip_adapter_synthetic_two_trials():
    raw = {
        "data": {
            "trial": [
                np.vstack([np.arange(4), np.arange(4) + 10]),
                np.vstack([np.arange(4) + 20, np.arange(4) + 30]),
            ],
            "time": [np.arange(4) / 100, np.arange(4) / 100],
            "label": ["C3", "C4"],
            "fsample": 100,
        }
    }
    rec = validate_recording(FieldTripAdapter().convert(raw))
    assert rec.signal.shape == (8, 2)
    assert rec.channel_names() == ["C3", "C4"]
    assert rec.metadata["n_trials"] == 2


def test_generic_mat_adapter_unambiguous():
    raw = {
        "my_signal_data": np.column_stack([np.arange(10), np.arange(10) + 10]),
        "fs": 1000,
        "labels": ["A", "B"],
    }
    rec = validate_recording(GenericMatAdapter().convert(raw))
    assert rec.signal.shape == (10, 2)
    assert rec.sampling_rate == 1000
    assert rec.channel_names() == ["A", "B"]


def test_table_adapter_time_and_channels():
    df = pd.DataFrame({"Time": [0.0, 0.1, 0.2], "C3": [1, 2, 3], "C4": [4, 5, 6], "EOG1": [0, 0, 0]})
    rec = validate_recording(TableAdapter().convert({"dataframe": df}))
    assert rec.signal.shape == (3, 2)
    assert rec.channel_names() == ["C3", "C4"]
    assert abs(rec.sampling_rate - 10.0) < 1e-9


def test_numpy_adapter_npz_like_dict():
    raw = {"signal": np.vstack([np.arange(5), np.arange(5) + 10]), "fs": 500, "labels": ["C3", "C4"]}
    rec = validate_recording(NumpyArrayAdapter().convert(raw))
    assert rec.signal.shape == (5, 2)
    assert rec.channel_names() == ["C3", "C4"]
    assert rec.sampling_rate == 500
