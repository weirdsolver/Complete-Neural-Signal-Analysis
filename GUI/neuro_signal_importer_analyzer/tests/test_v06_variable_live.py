import json
from pathlib import Path

import numpy as np

from neuro_importer_live.channel_manifest import build_manifest, finalspark_32_manifest, generated_channel_names
from neuro_importer_live.spectral import compute_channel_metrics
from neuro_importer_live.ws_utils import pack_binary_float32_frame


def test_generated_manifest_64_channels():
    m = build_manifest(64, profile="generated_numeric", sample_rate_hz=1000)
    assert m.n_channels == 64
    assert m.channel_names[0] == "ch_000"
    assert m.channel_names[-1] == "ch_063"


def test_finalspark_manifest_groups():
    m = finalspark_32_manifest(sample_rate_hz=30000)
    assert m.n_channels == 32
    assert "organoid_0" in m.groups
    assert len(m.groups["organoid_3"]) == 8
    assert m.channel_names[0] == "mea0_organoid0_e0"


def test_channels_csv_manifest(tmp_path: Path):
    p = tmp_path / "channels.csv"
    p.write_text("name,type,x,y\nA,MEA,0,0\nB,MEA,1,0\n", encoding="utf-8")
    m = build_manifest(2, channels_csv=p, profile="auto")
    assert m.channel_names == ["A", "B"]
    assert m.geometry["B"]["x"] == 1


def test_spectral_metrics_variable_channel_counts():
    rng = np.random.default_rng(0)
    for n_channels in (8, 32, 64):
        x = rng.normal(size=(1000, n_channels)).astype(np.float32)
        names = generated_channel_names(n_channels)
        result = compute_channel_metrics(x, 1000.0, names)
        assert len(result["metrics_by_channel"]) == n_channels
        assert result["psd"].shape[1] == n_channels
        assert result["frequency_hz"].ndim == 1


def test_binary_frame_contains_dynamic_shape():
    payload = np.zeros((100, 17), dtype=np.float32)
    header = {"shape": [100, 17], "n_channels": 17}
    frame = pack_binary_float32_frame(header, payload)
    hlen = int.from_bytes(frame[:4], "little")
    off = int.from_bytes(frame[4:8], "little")
    decoded = json.loads(frame[8:8+hlen].decode("utf-8"))
    assert decoded["n_channels"] == 17
    assert len(frame) == off + payload.size * 4
