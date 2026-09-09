from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_importer_analysis import list_advanced_methods, run_advanced_method


def _make_recording(tmp_path: Path) -> Path:
    fs = 1000.0
    t = np.arange(6000) / fs
    rng = np.random.default_rng(123)
    signal = np.column_stack([
        np.sin(2 * np.pi * 8 * t) + 0.05 * rng.normal(size=t.size),
        np.sin(2 * np.pi * 11 * t) + 0.05 * rng.normal(size=t.size),
        np.sin(2 * np.pi * 15 * t) + 0.05 * rng.normal(size=t.size),
        np.sin(2 * np.pi * 20 * t) + 0.05 * rng.normal(size=t.size),
    ])
    np.save(tmp_path / "signal.npy", signal)
    pd.DataFrame({"name": ["Fp1", "Fp2", "Cz", "O1"]}).to_csv(tmp_path / "channels.csv", index=False)
    (tmp_path / "metadata.json").write_text(json.dumps({"sampling_rate_hz": fs}), encoding="utf-8")
    return tmp_path


def test_manual_expert_mfdfa_registered() -> None:
    ids = {m["id"] for m in list_advanced_methods()}
    assert "manual_expert_mfdfa" in ids


def test_manual_expert_mfdfa_runs_and_saves_plots(tmp_path: Path) -> None:
    root = _make_recording(tmp_path)
    result = run_advanced_method("manual_expert_mfdfa", root, {"mode": "ultra", "max_channels": 4, "temporal_stability": "false"})
    assert result["ok"] is True
    payload = result["result"]["manual_expert_mfdfa"]
    assert payload["summary"]["n_channels"] == 4
    assert payload["summary"]["q_count"] == 17
    assert len(payload["rows"]) == 4
    assert len(payload["plot_paths"]) >= 8
    for plot in payload["plot_paths"]:
        assert Path(plot["path"]).exists()
    assert Path(payload["outputs"]["summary_csv"]).exists()
    assert Path(payload["outputs"]["hq_csv"]).exists()
    assert Path(payload["outputs"]["Fq_csv"]).exists()
