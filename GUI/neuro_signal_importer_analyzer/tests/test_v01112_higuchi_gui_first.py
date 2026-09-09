from __future__ import annotations

from pathlib import Path


def test_v01114_gui_has_simple_home_and_analysis_workflows():
    index = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    app_js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    css = Path("neuro_signal_webapp/static/styles.css").read_text(encoding="utf-8")

    assert "v0.11.32 — Minimal Dashboard" in index
    assert "homeTab" in index
    assert "higuchiTab" in index
    assert "higuchiDirectCard" in index
    assert "embeddedFdTab" in index
    assert "embeddedFdDirectCard" in index
    assert "Run Higuchi Fractal Dimension" in index
    assert "Run Embedded Fractal Dimension" in index
    assert "runHiguchiDirect" in app_js
    assert "runEmbeddedFdDirect" in app_js
    assert "embedded_fractal_dimension" in app_js
    assert "simple-action-grid" in css
    assert "embedded-fd-card" in css


def test_advanced_analysis_route_targets_higuchi_and_embedded_fd_has_route():
    server = Path("neuro_signal_webapp/app_server.py").read_text(encoding="utf-8")
    index = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")

    assert '@app.get("/advanced-analysis"' in server
    assert 'RedirectResponse(url="/#higuchiDirectCard"' in server
    assert '@app.get("/embedded-fractal-dimension"' in server
    assert '@app.get("/legacy-neuromouse-plots"' not in server
    assert 'href="/legacy-neuromouse-plots"' not in index
