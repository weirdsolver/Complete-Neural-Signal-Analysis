from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

from neuro_signal_webapp.job_manager import discover_primary_signal_files
from neuro_importer.pipeline import NeuroImportPipeline


def test_zip_archive_extracts_and_discovers_primary_mat(tmp_path: Path):
    mat_path = tmp_path / "BCICIV_calib_ds1a.mat"
    signal = np.arange(1000, dtype=np.int16).reshape(100, 10)
    nfo = {
        "fs": np.array([[100]]),
        "clab": np.array([[f"C{i}" for i in range(10)]], dtype=object),
    }
    sio.savemat(mat_path, {"cnt": signal, "nfo": nfo})

    zip_path = tmp_path / "BCICIV_1_mat.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(mat_path, arcname="BCICIV_calib_ds1a.mat")

    discovery = discover_primary_signal_files([zip_path])

    assert len(discovery["archives"]) == 1
    assert len(discovery["primary_files"]) == 1
    assert discovery["primary_files"][0].name == "BCICIV_calib_ds1a.mat"

    out_dir = tmp_path / "converted"
    NeuroImportPipeline().convert(discovery["primary_files"][0], output_dir=out_dir)
    assert (out_dir / "signal.npy").exists()
    channels_text = (out_dir / "channels.csv").read_text()
    assert "C0" in channels_text
