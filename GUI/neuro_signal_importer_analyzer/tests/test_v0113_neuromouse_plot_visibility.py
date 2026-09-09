from __future__ import annotations

from pathlib import Path


def test_simple_dashboard_is_the_visible_default_path():
    index = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    css = Path("neuro_signal_webapp/static/styles.css").read_text(encoding="utf-8")
    app_js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")

    assert 'button class="tab active" data-tab="homeTab"' in index
    assert 'simple-action-grid' in index
    assert 'Run embedded FD' in index
    assert 'More tools' in index
    assert 'simple-action-grid' in css
    assert 'showTab(\'homeTab\')' in app_js
    assert 'setTimeout(() => loadLatestAdvancedPlots(), 50)' not in app_js


def test_neuromouse_workbench_is_available_only_under_more_tools():
    index = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    assert 'data-tab="neuromouseTab"' in index
    assert 'href="/legacy-neuromouse-plots"' not in index
    assert 'Open NeuroMouse Demo / Manual Loader' in index
