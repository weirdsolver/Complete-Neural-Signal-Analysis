from pathlib import Path
import json

import numpy as np
import pandas as pd

from neuro_importer_analysis.advanced_methods import list_advanced_methods, run_advanced_method


def _write_recording(root: Path, n_samples: int = 6000, n_channels: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    t = np.linspace(0, 8 * np.pi, n_samples)
    data = []
    for idx in range(n_channels):
        sig = np.sin((idx + 1) * t) + 0.1 * rng.normal(size=n_samples)
        data.append(sig)
    x = np.column_stack(data)
    np.save(root / "signal.npy", x.astype("float32"))
    pd.DataFrame({"name": ["Fp1", "Fp2", "Cz"][:n_channels]}).to_csv(root / "channels.csv", index=False)
    (root / "metadata.json").write_text(json.dumps({"sampling_rate_hz": 1000}), encoding="utf-8")


def test_mfdfa_plot_viewer_registered_and_gui() -> None:
    ids = {m["id"] for m in list_advanced_methods()}
    assert "mfdfa_plot_viewer" in ids
    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    assert "mfdfaViewerTab" in html
    assert "MFDFA Plot Viewer" in html
    assert "mfdfa_plot_viewer" in js
    assert "renderMfdfaViewerPanel" in js
    server = Path("neuro_signal_webapp/app_server.py").read_text(encoding="utf-8")
    assert "/mfdfa-viewer" in server


def test_mfdfa_plot_viewer_runs_after_generating_inputs(tmp_path: Path) -> None:
    root = tmp_path / "recording"
    _write_recording(root)
    result = run_advanced_method(
        "mfdfa_plot_viewer",
        root,
        {"mode": "ultra", "run_mfdfa_if_missing": "true", "max_channels": 3, "max_images_to_show": 8, "max_analysis_samples": 6000},
    )
    assert result["ok"] is True
    payload = result["result"]["mfdfa_plot_viewer"]
    assert payload["summary"]["images_found"] >= 1
    assert payload["summary"]["viewer_plot_count"] >= 3
    assert payload["summary"]["generated_mfdfa_first"] is True
    assert any("contact sheet" in p["title"].lower() for p in payload["plot_paths"])
    for key in ("manifest_csv", "display_csv", "summary_txt"):
        assert Path(payload["outputs"][key]).exists()
