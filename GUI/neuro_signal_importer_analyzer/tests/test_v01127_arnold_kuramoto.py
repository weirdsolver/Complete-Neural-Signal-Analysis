from pathlib import Path
import json
import numpy as np
import pandas as pd

from neuro_importer_analysis.advanced_methods import list_advanced_methods, run_advanced_method


def _make_recording(tmp_path: Path) -> Path:
    fs = 100.0
    n = 900
    t = np.arange(n) / fs
    sig = np.column_stack([
        np.sin(2 * np.pi * 5 * t),
        np.sin(2 * np.pi * 8 * t + 0.3),
        np.sin(2 * np.pi * 12 * t + 0.7),
    ])
    np.save(tmp_path / "signal.npy", sig)
    (tmp_path / "metadata.json").write_text(json.dumps({"sampling_rate_hz": fs}), encoding="utf-8")
    pd.DataFrame({"name": ["A", "B", "C"]}).to_csv(tmp_path / "channels.csv", index=False)
    return tmp_path


def test_arnold_kuramoto_method_registered():
    ids = {m["id"] for m in list_advanced_methods()}
    assert "arnold_tongues_kuramoto" in ids


def test_arnold_kuramoto_runs_and_saves_plots(tmp_path):
    rec = _make_recording(tmp_path)
    payload = run_advanced_method(
        "arnold_tongues_kuramoto",
        rec,
        {"mode": "ultra", "a_count": 3, "b_count": 3, "t_end": 2, "n_t_eval": 30, "freq_min": 1, "freq_max": 20},
    )
    assert payload["ok"] is True
    result = payload["result"]["arnold_tongues_kuramoto"]
    assert result["summary"]["n_channels"] == 3
    assert result["summary"]["grid_points"] == 9
    assert result["summary"]["plot_count"] >= 19
    assert "max_drive_lock" in result["summary"]
    assert "mean_frequency_error" in result["summary"]
    assert "drive_lock" in result["grid_rows"][0]
    assert "frequency_error" in result["grid_rows"][0]
    for key in ("grid_csv", "omega_csv", "npz", "summary_txt"):
        assert Path(result["outputs"][key]).exists()
    assert Path(result["plot_paths"][0]["path"]).exists()


def test_arnold_kuramoto_gui_present():
    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    assert "arnoldKuramotoTab" in html
    assert "Run Arnold/Kuramoto" in html
    assert "arnold_tongues_kuramoto" in js
    assert "renderArnoldKuramotoPanel" in js
