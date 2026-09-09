from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from neuro_importer.pipeline import NeuroImportPipeline


def test_generic_hdf5_continuous_adapter(tmp_path: Path):
    path = tmp_path / "generic.h5"
    with h5py.File(path, "w") as f:
        f.attrs["sampling_rate"] = 1000.0
        f.create_dataset("raw_voltage", data=np.arange(1000 * 4, dtype=float).reshape(1000, 4))
        f.create_dataset("channel_labels", data=np.array([b"A", b"B", b"C", b"D"]))
    out = tmp_path / "out_generic"
    result = NeuroImportPipeline().convert(path, output_dir=out)
    assert result["recording"].signal.shape == (1000, 4)
    assert result["adapter"] == "generic_hdf5_continuous"
    assert (out / "file_tree_report.json").exists()


def test_finalspark_live_mea_adapter(tmp_path: Path):
    path = tmp_path / "finalspark_live.h5"
    with h5py.File(path, "w") as f:
        f.attrs["sampling_rate"] = 3750.0
        for chunk in range(3):
            g = f.create_group(f"timestamp_{chunk}")
            for e in range(4):
                g.create_dataset(f"electrode_{e}", data=np.ones(8) * (chunk * 10 + e))
    out = tmp_path / "out_fs"
    result = NeuroImportPipeline().convert(path, output_dir=out)
    rec = result["recording"]
    assert result["adapter"] == "finalspark_live_mea"
    assert rec.signal.shape == (24, 4)
    assert rec.sampling_rate == 3750.0
    assert (out / "electrodes.csv").exists()


def test_cortical_labs_cl1_adapter(tmp_path: Path):
    path = tmp_path / "cl1.h5"
    with h5py.File(path, "w") as f:
        f.attrs["system_id"] = "cl1-test"
        f.attrs["channel_count"] = 4
        f.attrs["sampling_frequency"] = 25000.0
        f.attrs["uV_per_sample_unit"] = 0.195
        f.create_dataset("samples", data=np.ones((250, 4), dtype=np.int16) * 10)
        f.create_dataset("spikes", data=np.ones((10, 3)))  # must be ignored
    out = tmp_path / "out_cl1"
    result = NeuroImportPipeline().convert(path, output_dir=out)
    rec = result["recording"]
    assert result["adapter"] == "cortical_labs_cl1"
    assert rec.signal.shape == (250, 4)
    assert np.isclose(rec.signal[0, 0], 1.95)
    assert "spikes" not in rec.metadata.get("format", "").lower()
