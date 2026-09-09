from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _quality_counts(quality: dict[str, Any] | None) -> tuple[int, int, int]:
    if not quality:
        return 0, 0, 0
    return (
        len(quality.get("errors", []) or []),
        len(quality.get("warnings", []) or []),
        len(quality.get("infos", []) or []),
    )


def manifest_row_from_result(
    *,
    recording_id: str,
    source_file: str,
    output_dir: str | Path,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result = result or {}
    recording = result.get("recording")
    quality = result.get("quality") or (recording.quality.to_dict() if recording is not None else {})
    n_errors, n_warnings, n_infos = _quality_counts(quality)
    return {
        "recording_id": recording_id,
        "source_file": str(source_file),
        "output_dir": str(output_dir),
        "status": status,
        "adapter": result.get("adapter"),
        "n_samples": getattr(recording, "n_samples", None),
        "n_channels": getattr(recording, "n_channels", None),
        "sampling_rate": getattr(recording, "sampling_rate", None),
        "duration_seconds": (getattr(recording, "n_samples", None) / getattr(recording, "sampling_rate", 1)) if getattr(recording, "sampling_rate", None) else None,
        "warnings_count": n_warnings,
        "errors_count": n_errors,
        "infos_count": n_infos,
        "error_message": error,
    }


def write_manifest(rows: list[dict[str, Any]], output_dir: str | Path, filename: str = "dataset_manifest.csv") -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)
