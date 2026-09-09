from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_direct_advanced_analysis_page_points_to_fractal_workflows():
    html = (ROOT / "neuro_signal_webapp" / "static" / "advanced_analysis.html").read_text(encoding="utf-8")
    assert "Run Higuchi FD" in html
    assert "Run Embedded FD" in html
    assert "Polar Alpha Chronomap" not in html


def test_launcher_has_dedicated_fractal_tabs_without_legacy_link():
    html = (ROOT / "neuro_signal_webapp" / "static" / "index.html").read_text(encoding="utf-8")
    assert "homeTab" in html
    assert "higuchiDirectCard" in html
    assert "embeddedFdDirectCard" in html
    assert "Run Higuchi Fractal Dimension" in html
    assert "Run Embedded Fractal Dimension" in html
    assert "href=\"/legacy-neuromouse-plots\"" not in html


def test_server_exposes_direct_advanced_analysis_route():
    server = (ROOT / "neuro_signal_webapp" / "app_server.py").read_text(encoding="utf-8")
    assert '@app.get("/advanced-analysis"' in server
    assert 'RedirectResponse(url="/#higuchiDirectCard"' in server
    assert '@app.get("/legacy-neuromouse-plots"' not in server


def test_neuromouse_workbench_has_advanced_fallback_near_toggle():
    html = (ROOT / "neuro_signal_webapp" / "neuromouse" / "index.html").read_text(encoding="utf-8")
    assert "advanced-visible-fallback" in html
    for label in [
        "Polar Alpha Chronomap",
        "Kuramoto Animation",
        "Channel Network",
        "TDA View",
    ]:
        assert label in html
