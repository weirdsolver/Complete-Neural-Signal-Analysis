from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "project_name": "neuro_signal_importer_project",
    "scan": {
        "include_extensions": [
            ".mat", ".set", ".edf", ".bdf", ".csv", ".tsv", ".xlsx",
            ".npy", ".npz", ".h5", ".hdf5", ".nwb"
        ],
        "exclude_dir_names": [".git", "__pycache__", "converted", "outputs", "output"],
        "max_files": None,
    },
    "conversion": {
        "min_confidence": 0.5,
        "include_aux": False,
        "export_neural_signal": False,
        "write_tree_report": True,
    },
    "export": {
        "format": "npy",
        "save_signal_csv": False,
        "csv_max_mb": 250.0,
        "compression": "gzip",
    },
    "units": {
        "original_units": None,
        "target_units": None,
        "scale_factor": None,
        "offset": None,
    },
    "preprocessing": {
        "enabled": False,
        "demean": False,
        "detrend": False,
        "notch_hz": None,
        "bandpass_hz": None,
        "downsample_to_hz": None,
        "normalization": None,
        "astype": "float32",
    },
    "windowing": {
        "enabled": False,
        "window_seconds": None,
        "window_samples": None,
        "step_seconds": None,
        "step_samples": None,
        "drop_last": True,
        "max_windows": None,
        "astype": "float32",
    },
    "qc": {
        "enabled": True,
        "flat_std_threshold": 1e-12,
        "nan_fraction_warn": 0.01,
        "outlier_z_warn": 12.0,
    },
    "channels": {
        "exclude_patterns": ["^AUX", "^RESP", "^TRIG", "^STIM", "^BIP"],
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries, returning a new dictionary."""
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("Install pyyaml to load config files: pip install pyyaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data or {}


def load_project_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load project config by merging defaults, optional YAML, and optional dict overrides."""
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        config = deep_merge(config, load_yaml(path))
    if overrides:
        config = deep_merge(config, overrides)
    return config


def write_default_config(path: str | Path) -> str:
    if yaml is None:
        raise ImportError("Install pyyaml to write config files: pip install pyyaml")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    return str(p)
