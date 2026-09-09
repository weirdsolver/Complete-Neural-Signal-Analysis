from pathlib import Path
import json


def test_direct_advanced_analysis_no_longer_uses_demo_fallback_page():
    html = Path("neuro_signal_webapp/static/advanced_analysis.html").read_text(encoding="utf-8")
    assert "Run Higuchi FD" in html
    assert "bundled original NeuroMouse demo" not in html
    assert "Polar Alpha Chronomap" not in html


def test_launcher_defaults_to_dedicated_fractal_tabs():
    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    assert "higuchiTab" in html
    assert "embeddedFdTab" in html
    assert "optionalTabs" in html


def test_bundled_demo_has_required_advanced_objects():
    data = json.loads(Path("neuro_signal_webapp/neuromouse/data/data.json").read_text(encoding="utf-8"))
    assert data.get("polar_chronomap", {}).get("posterior_alpha")
    assert data.get("kuramoto", {}).get("channel_phases")
    assert data.get("channel_network", {}).get("composite_correlation")
    assert data.get("tda", {}).get("status") == "computed"
