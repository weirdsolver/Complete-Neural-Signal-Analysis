from __future__ import annotations

from pathlib import Path


def test_analysis_cards_are_available_but_not_forced_visible_by_default():
    index = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    css = Path("neuro_signal_webapp/static/styles.css").read_text(encoding="utf-8")
    app_js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")

    assert 'id="higuchiTab" class="panel"' in index
    assert 'id="embeddedFdTab" class="panel"' in index
    assert 'id="advancedMethodsTab" class="panel"' in index
    assert 'id="higuchiDirectCard"' in index
    assert 'id="embeddedFdDirectCard"' in index
    assert 'function ensureAdvancedAnalysisVisible' in app_js
    assert 'function runEmbeddedFdDirect' in app_js
    assert 'embedded-fd-card' in css
    for label in ('HFD by channel', 'Scalp layout map', 'Regional mean ± SEM', 'Rolling HFD stability'):
        assert label in index
    for label in ('D2 vs embedding dimension', 'Box FD vs embedding dimension', 'Fit QC'):
        assert label in index


def test_embedded_neuromouse_advanced_views_are_still_packaged():
    index = Path("neuro_signal_webapp/neuromouse/index.html").read_text(encoding="utf-8")
    css = Path("neuro_signal_webapp/neuromouse/style.css").read_text(encoding="utf-8")

    assert 'id="advanced-toggle" class="advanced-toggle-btn" type="button" aria-expanded="true"' in index
    assert '<div id="advanced-views" style="display:grid">' in index
    assert 'setAdvancedOpen(true);' in index
    assert '#advanced-views' in css and 'display: grid !important' in css
