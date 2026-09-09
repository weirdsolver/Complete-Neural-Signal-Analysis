from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_importer.batch import BatchConverter, scan_dataset
from neuro_importer.config import load_project_config, write_default_config
from neuro_importer.pipeline import NeuroImportPipeline
from neuro_importer.qc import signal_qc
from neuro_importer.windows import build_windows


def test_scan_dataset_finds_numpy(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((10, 2)))
    (tmp_path / "ignore.txt").write_text("no")
    config = load_project_config()
    df = scan_dataset(tmp_path, config)
    assert len(df) == 1
    assert df.iloc[0]["filename"] == "a.npy"


def test_convert_with_qc_and_windows(tmp_path):
    arr = np.random.default_rng(1).normal(size=(100, 4)).astype("float32")
    path = tmp_path / "signal.npy"
    np.save(path, arr)
    out = tmp_path / "converted"
    result = NeuroImportPipeline().convert(
        path,
        output_dir=out,
        sampling_rate=100.0,
        qc_config={"enabled": True},
        window_config={"enabled": True, "window_samples": 20, "step_samples": 10},
        min_confidence=0.4,
    )
    assert (out / "signal.npy").exists()
    assert (out / "qc_report.html").exists()
    assert (out / "windows" / "X_windows.npy").exists()
    X = np.load(out / "windows" / "X_windows.npy")
    assert X.shape[1:] == (20, 4)
    assert result["window_paths"]["window_index"].endswith("window_index.csv")


def test_convert_with_preprocessing_preserves_raw(tmp_path):
    arr = np.arange(200, dtype=float).reshape(100, 2)
    path = tmp_path / "signal.npy"
    np.save(path, arr)
    out = tmp_path / "converted_pre"
    NeuroImportPipeline().convert(
        path,
        output_dir=out,
        sampling_rate=100.0,
        preprocess_config={"enabled": True, "demean": True, "astype": "float32"},
        min_confidence=0.4,
    )
    assert (out / "raw" / "signal.npy").exists()
    assert (out / "processed" / "signal.npy").exists()
    raw = np.load(out / "raw" / "signal.npy")
    processed = np.load(out / "processed" / "signal.npy")
    assert raw.shape == processed.shape
    assert not np.allclose(raw, processed)


def test_batch_converter_writes_manifest(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    np.save(raw / "a.npy", np.ones((50, 3), dtype="float32"))
    np.save(raw / "b.npy", np.ones((60, 2), dtype="float32"))
    out = tmp_path / "out"
    result = BatchConverter().convert_dataset(raw, out, continue_on_error=True)
    assert result["n_scanned"] == 2
    assert (out / "dataset_manifest.csv").exists()
    manifest = pd.read_csv(out / "dataset_manifest.csv")
    assert len(manifest) == 2
    assert set(manifest["status"]) == {"converted"}


def test_write_default_config(tmp_path):
    p = tmp_path / "project_config.yaml"
    written = write_default_config(p)
    assert Path(written).exists()
    cfg = load_project_config(p)
    assert "preprocessing" in cfg
