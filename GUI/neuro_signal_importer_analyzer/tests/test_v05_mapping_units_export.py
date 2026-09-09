from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from neuro_importer.mapping import MappingSpec, write_mapping
from neuro_importer.pipeline import NeuroImportPipeline
from neuro_importer.units import UnitCalibration


def test_unit_calibration_microvolts_to_millivolts():
    arr = np.array([[1000.0, 2000.0]])
    cal = UnitCalibration.from_values(original_units="microvolts", target_units="millivolts")
    out, info = cal.apply_array(arr)
    assert np.allclose(out, [[1.0, 2.0]])
    assert info["final_units"] == "millivolts"


def test_manual_mapping_numpy_array(tmp_path: Path):
    data = np.arange(40, dtype=float).reshape(10, 4)
    source = tmp_path / "weird.npy"
    np.save(source, data)
    mapping = MappingSpec(signal_path="array", sampling_rate=1000, orientation="samples_by_channels", original_units="microvolts", target_units="millivolts")
    mapping_path = tmp_path / "mapping.yaml"
    write_mapping(mapping_path, mapping)
    out = tmp_path / "out"
    result = NeuroImportPipeline(load_plugins=False).convert(source, output_dir=out, mapping_path=mapping_path, export_config={"format": "memmap", "save_signal_csv": False})
    assert result["adapter"] == "manual_mapping"
    assert (out / "signal_memmap.npy").exists()
    assert not (out / "signal.csv").exists()
    loaded = np.load(out / "signal_memmap.npy")
    assert loaded.shape == (10, 4)
    assert np.isclose(loaded[1, 0], data[1, 0] / 1000.0)
    assert (out / "provenance.json").exists()


def test_generate_mapping_template_hdf5(tmp_path: Path):
    import h5py
    source = tmp_path / "x.h5"
    with h5py.File(source, "w") as f:
        f.create_dataset("samples", data=np.zeros((20, 3)))
        f.attrs["sampling_frequency"] = 30000
    out = tmp_path / "mapping"
    paths = NeuroImportPipeline(load_plugins=False).generate_mapping_template(source, output_dir=out, sampling_rate=30000)
    assert Path(paths["failure_report"]).exists()
    assert Path(paths["mapping_template"]).exists()
    mapping_data = yaml.safe_load(Path(paths["mapping_template"]).read_text())
    assert mapping_data["signal_path"] == "/samples"
    assert mapping_data["sampling_rate"] == 30000
