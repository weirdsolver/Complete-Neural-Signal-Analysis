from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_original_neuromouse_loader_falls_back_after_stale_backend_url():
    js = (ROOT / "neuro_signal_webapp" / "neuromouse" / "js" / "sources" / "static-source.js").read_text(encoding="utf-8")
    assert "resolveStaticDataCandidates" in js
    assert "NEURO_SIGNAL_LAST_BACKEND_DATASET_URL" in js
    assert "clearRememberedBackendDataset" in js
    assert '"/api/neuromouse/latest/data.json"' in js
    assert '"../../data/data.json"' in js
    assert 'fetch(candidate.url, { cache: "no-store" })' in js


def test_bundled_original_neuromouse_demo_contains_required_advanced_objects():
    import json
    data = json.loads((ROOT / "neuro_signal_webapp" / "neuromouse" / "data" / "data.json").read_text(encoding="utf-8"))
    assert data.get("polar_chronomap")
    assert data.get("kuramoto")
    assert data.get("channel_network")
    assert data.get("tda", {}).get("status") == "computed"
