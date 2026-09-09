from pathlib import Path


def test_v01115_minimal_home_and_tabs():
    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    assert "v0.11.32 — Minimal Dashboard" in html
    assert "Import dataset / datasets" in html
    assert "Compare datasets" in html
    assert "Run Higuchi FD" in html
    assert "Run embedded FD" in html
    assert "higuchiTab" in html
    assert "embeddedFdTab" in html
    assert 'href="/legacy-neuromouse-plots"' not in html
    assert "showTab('higuchiTab')" in js
    assert "showTab('embeddedFdTab')" in js


def test_generic_runner_renders_fractal_plots():
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    assert "methodId === 'higuchi_fractal_dimension'" in js
    assert "renderHiguchiFdPanel" in js
    assert "methodId === 'embedded_fractal_dimension'" in js
    assert "renderEmbeddedFdPanel" in js
