from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_wavelet_hurst_gui_and_method_registered():
    html = (ROOT / "neuro_signal_webapp" / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "neuro_signal_webapp" / "static" / "app.js").read_text(encoding="utf-8")
    methods = (ROOT / "neuro_importer_analysis" / "advanced_methods.py").read_text(encoding="utf-8")
    assert "waveletTab" in html
    assert "Run Wavelet Hurst" in html
    assert "renderWaveletPanel" in js
    assert "wavelet_hurst_exponent" in methods

def test_canvas_axes_draw_numeric_tick_labels():
    js = (ROOT / "neuro_signal_webapp" / "static" / "app.js").read_text(encoding="utf-8")
    assert "function hfdFormatTick" in js
    assert "function hfdTickValues" in js
    assert "xRange = null, yRange = null" in js
    assert "ctx.fillText(hfdFormatTick(tick)" in js
    assert "hfdDrawAxes(ctx, left, top, width, height, 'Embedding Dimension', 'Mean Katz Fractal Dimension', xr, yr)" in js
