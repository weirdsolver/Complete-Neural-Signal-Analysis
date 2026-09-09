from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_importer_analysis import list_advanced_methods, run_advanced_method


def _make_recording(root: Path, n_samples: int = 4096, n_channels: int = 2) -> Path:
    fs = 1000.0
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(224)
    signals = []
    for i in range(n_channels):
        signals.append(np.sin(2 * np.pi * (8 + i * 3) * t) + 0.06 * rng.normal(size=t.size))
    np.save(root / "signal.npy", np.column_stack(signals).astype("float32"))
    pd.DataFrame({"name": ["C3", "C4"][:n_channels]}).to_csv(root / "channels.csv", index=False)
    (root / "metadata.json").write_text(json.dumps({"sampling_rate_hz": fs}), encoding="utf-8")
    return root


def test_mfdfa_iaaft_registered_and_gui() -> None:
    methods = {m["id"]: m for m in list_advanced_methods()}
    assert "manual_mfdfa_iaaft_surrogate" in methods
    spec = methods["manual_mfdfa_iaaft_surrogate"]
    mode_param = next(p for p in spec["parameters"] if p["name"] == "mode")
    assert mode_param["default"] == "ultra"
    assert mode_param["options"] == ["ultra", "fast", "balanced", "full"]

    html = Path("neuro_signal_webapp/static/index.html").read_text(encoding="utf-8")
    js = Path("neuro_signal_webapp/static/app.js").read_text(encoding="utf-8")
    server = Path("neuro_signal_webapp/app_server.py").read_text(encoding="utf-8")
    assert "mfdfaIaaftTab" in html
    assert "MFDFA IAAFT Surrogate" in html
    assert "manual_mfdfa_iaaft_surrogate" in js
    assert "renderMfdfaIaaftPanel" in js
    assert "/mfdfa-iaaft" in server


def test_mfdfa_iaaft_runs_all_available_channels_and_saves_outputs(tmp_path: Path) -> None:
    root = _make_recording(tmp_path)
    result = run_advanced_method(
        "manual_mfdfa_iaaft_surrogate",
        root,
        {"mode": "ultra", "n_surrogates": 2, "iaaft_iters": 2, "max_channels": 2, "random_seed": 7, "max_analysis_samples": 4096},
    )
    assert result["ok"] is True
    payload = result["result"]["manual_mfdfa_iaaft_surrogate"]
    assert payload["summary"]["mode"] == "ultra"
    assert payload["summary"]["n_channels"] == 2
    assert payload["summary"]["n_surrogates"] == 2
    assert payload["summary"]["iaaft_iters"] == 2
    assert len(payload["rows"]) == 2
    assert len(payload["surrogate_rows"]) == 4
    assert len(payload["plot_paths"]) >= 4
    for plot in payload["plot_paths"]:
        assert Path(plot["path"]).exists()
    for path in payload["per_channel_npz"]:
        assert Path(path).exists()
    for key in ("summary_csv", "surrogate_csv", "summary_txt"):
        assert Path(payload["outputs"][key]).exists()
