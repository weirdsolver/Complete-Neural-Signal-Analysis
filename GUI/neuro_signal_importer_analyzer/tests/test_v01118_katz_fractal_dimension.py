from pathlib import Path
import numpy as np
import pandas as pd


def _make_recording(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    signal = rng.normal(size=(1200, 2)).astype(float)
    np.save(root / "signal.npy", signal)
    pd.DataFrame({"name": ["Fp1", "Fp2"]}).to_csv(root / "channels.csv", index=False)
    (root / "metadata.json").write_text('{"sampling_rate_hz": 1000}', encoding="utf-8")


def test_v01118_katz_method_registered():
    from neuro_importer_analysis.advanced_methods import list_advanced_methods
    ids = {m["id"] for m in list_advanced_methods()}
    assert "katz_fractal_dimension" in ids


def test_v01118_katz_backend_generates_delay_embeddings(tmp_path):
    from neuro_importer_analysis.advanced_methods import run_advanced_method
    rec = tmp_path / "rec"
    _make_recording(rec)
    payload = run_advanced_method("katz_fractal_dimension", rec, {"mode": "ultra", "embedding_dims": "2,3", "max_channels": 2})
    kfd = payload["result"]["katz_fractal_dimension"]
    assert kfd["summary"]["ok_rows"] >= 2
    assert kfd["by_dimension"]
    assert kfd["by_channel"]
    assert kfd["channel_dimension_matrix"]
    assert Path(kfd["outputs"]["per_embedding_csv"]).exists()


def test_v01118_katz_gui_visible():
    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    assert "v0.11.32" in html or "0.11.32" in html
    assert "katzDirectCard" in html
    assert "Run Katz Fractal Dimension" in html
    assert "renderKatzPanel" in js
    assert "katz_fractal_dimension" in js
