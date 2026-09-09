from __future__ import annotations

from typing import Any


def first_present(mapping: dict[str, Any], names: tuple[str, ...]) -> tuple[str | None, Any | None]:
    lower = {str(k).lower(): (str(k), v) for k, v in mapping.items()}
    for name in names:
        found = lower.get(name.lower())
        if found:
            return found
    return None, None


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def infer_sampling_rate(attrs: dict[str, Any]) -> tuple[str | None, float | None]:
    return first_present(
        attrs,
        (
            "sampling_rate",
            "sample_rate",
            "sampling_frequency",
            "fs",
            "srate",
            "sfreq",
            "frames_per_second",
            "rate",
        ),
    )[0], safe_float(first_present(
        attrs,
        (
            "sampling_rate",
            "sample_rate",
            "sampling_frequency",
            "fs",
            "srate",
            "sfreq",
            "frames_per_second",
            "rate",
        ),
    )[1])


def infer_voltage_scale(attrs: dict[str, Any]) -> tuple[str | None, float | None, str | None]:
    """Return (attribute_name, scale_to_apply, output_unit) when obvious.

    The scale is applied multiplicatively to raw samples. This is conservative:
    it uses common explicit attributes and otherwise leaves raw units unchanged.
    """
    for key in ("uV_per_sample_unit", "microvolts_per_sample_unit", "uv_per_sample_unit"):
        if key in attrs:
            return key, safe_float(attrs[key]), "uV"
    for key in ("volts_per_sample_unit", "V_per_sample_unit"):
        if key in attrs:
            return key, safe_float(attrs[key]), "V"
    return None, None, None
