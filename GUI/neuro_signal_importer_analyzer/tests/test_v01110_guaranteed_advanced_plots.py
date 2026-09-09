from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_server_rendered_advanced_plot_page_contains_real_inline_svgs():
    from neuro_importer_neuromouse.advanced_plot_renderer import render_advanced_analysis_page

    data = json.loads((ROOT / "neuro_signal_webapp" / "neuromouse" / "data" / "data.json").read_text(encoding="utf-8"))
    html = render_advanced_analysis_page(data, {"source": "test_demo", "path": "demo"}, app_version="0.11.10")
    assert "server-rendered" in html
    for label in ["Polar Alpha Chronomap", "Kuramoto Animation", "Channel Network", "TDA View"]:
        assert label in html
    assert html.count("<svg") >= 4
    assert "Missing polar_chronomap" not in html
    assert "Missing kuramoto" not in html
    assert "Missing channel_network" not in html
    assert "Missing computed" not in html


def test_new_neuromouse_advanced_plots_method_is_listed():
    from neuro_importer_analysis import list_advanced_methods

    ids = [m["id"] for m in list_advanced_methods()]
    assert "neuromouse_advanced_plots" in ids


def test_advanced_plot_summary_api_route_exists():
    server = (ROOT / "neuro_signal_webapp" / "app_server.py").read_text(encoding="utf-8")
    assert '@app.get("/api/neuromouse/advanced-plots")' in server
