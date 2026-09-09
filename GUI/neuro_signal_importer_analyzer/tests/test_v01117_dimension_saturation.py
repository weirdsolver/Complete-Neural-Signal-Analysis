from pathlib import Path
import pandas as pd


def test_v01117_dimension_saturation_method_registered():
    from neuro_importer_analysis.advanced_methods import list_advanced_methods
    ids = {m["id"] for m in list_advanced_methods()}
    assert "dimension_saturation_profiling" in ids


def test_v01117_dimension_saturation_backend_from_existing_embedded_csv(tmp_path):
    from neuro_importer_analysis.advanced_methods import run_advanced_method
    root = tmp_path / "rec"
    out = root / "advanced_methods" / "embedded_fractal_dimension"
    out.mkdir(parents=True)
    rows = []
    for ch, base in [("Fp1", 1.0), ("Fp2", 1.2)]:
        vals = [base, base+0.5, base+0.7, base+0.78, base+0.82, base+0.84]
        for m, d in zip(range(2, 8), vals):
            rows.append({"channel": ch, "emb_dim": m, "corr_dim_d2": d, "boxcount_fd": d*0.9})
    pd.DataFrame(rows).to_csv(out / "embedded_fractal_dimension_summary_fast.csv", index=False)
    payload = run_advanced_method("dimension_saturation_profiling", root, {"run_embedded_if_missing": "false"})
    dsp = payload["result"]["dimension_saturation_profiling"]
    assert dsp["summary"]["n_channels"] == 2
    assert len(dsp["summary_rows"]) == 2
    assert len(dsp["delta_rows"]) > 0
    assert Path(dsp["outputs"]["summary_csv"]).exists()


def test_v01117_saturation_gui_visible():
    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    assert "v0.11.32" in html or "0.11.32" in html
    assert "saturationDirectCard" in html
    assert "Run Dimension Saturation Profiling" in html
    assert "renderSaturationPanel" in js
    assert "dimension_saturation_profiling" in js
