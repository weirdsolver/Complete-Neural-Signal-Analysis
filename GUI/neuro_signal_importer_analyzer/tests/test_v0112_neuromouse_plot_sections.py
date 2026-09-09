from __future__ import annotations

from pathlib import Path


def test_launcher_prioritizes_simple_dashboard_with_analysis_cards():
    index = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    app_js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")

    assert 'id="homeTab" class="panel active simple-home-panel"' in index
    assert 'id="higuchiTab" class="panel"' in index
    assert 'id="embeddedFdTab" class="panel"' in index
    assert 'id="advancedMethodsTab" class="panel"' in index
    assert 'id="higuchiDirectCard"' in index
    assert 'id="embeddedFdDirectCard"' in index
    assert 'Higuchi Fractal Dimension — Backend Analysis + Frontend Plots' in index
    assert 'Embedded Fractal Dimension — Attractor Embeddings' in index
    assert "function runHiguchiDirect" in app_js
    assert "function runEmbeddedFdDirect" in app_js
    assert "method_id: 'higuchi_fractal_dimension'" in app_js
    assert "method_id: 'embedded_fractal_dimension'" in app_js


def test_versions_are_consistent_for_v01115_launcher():
    assert 'version = "0.11.32"' in Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'APP_VERSION = "0.11.32"' in Path("neuro_signal_webapp/job_manager.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "0.11.32"' in Path("run_neuro_signal_app.py").read_text(encoding="utf-8")
    assert 'v0.11.32 — Minimal Dashboard' in Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
