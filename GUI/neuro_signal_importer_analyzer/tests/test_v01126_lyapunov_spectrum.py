from pathlib import Path
import numpy as np
import pandas as pd

from neuro_importer_analysis.advanced_methods import get_method_spec, run_advanced_method


def test_v01126_lyapunov_method_registered_and_gui_present():
    spec = get_method_spec("lyapunov_spectrum_custom")
    assert spec["name"] == "Lyapunov spectrum custom"
    assert any(p["name"] == "mode" and p["default"] == "ultra" for p in spec["parameters"])
    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    assert "lyapunovSpectrumTab" in html
    assert "Run Lyapunov Spectrum" in html
    assert "lyapunov_spectrum_custom" in js
    assert "renderLyapunovSpectrumPanel" in js


def test_v01126_lyapunov_runs_on_small_recording(tmp_path):
    n = 1000
    t = np.linspace(0, 16, n)
    rng = np.random.default_rng(42)
    x = np.column_stack([
        np.sin(t) + 0.01 * rng.normal(size=n),
        np.cos(1.7 * t) + 0.01 * rng.normal(size=n),
    ])
    np.save(tmp_path / "signal.npy", x)
    pd.DataFrame({"name": ["C3", "C4"]}).to_csv(tmp_path / "channels.csv", index=False)
    (tmp_path / "metadata.json").write_text('{"sampling_rate_hz": 1000}', encoding="utf-8")

    payload = run_advanced_method(
        "lyapunov_spectrum_custom",
        tmp_path,
        {
            "mode": "ultra",
            "embedding_dims": "2",
            "max_channels": 2,
            "max_points": 250,
            "max_steps_per_file": 16,
            "k_neighbors": 4,
            "theiler": 2,
            "stride": 10,
        },
    )
    assert payload["ok"] is True
    data = payload["result"]["lyapunov_spectrum_custom"]
    assert data["summary"]["n_channels"] == 2
    assert data["summary"]["n_files"] == 2
    assert data["summary"]["successful_files"] >= 1
    assert Path(data["outputs"]["enriched_csv"]).exists()
    assert len(data["plot_paths"]) >= 6
    for plot in data["plot_paths"][:3]:
        assert Path(plot["path"]).exists()
