from __future__ import annotations

import json
import zipfile
from urllib.parse import quote
from pathlib import Path
from typing import Any

import pandas as pd

from neuro_importer_analysis import run_comparative_analysis
from .static_dataset_builder import write_speedmouse_dataset


def build_neuromouse_comparison_pack(
    group_a_dirs: list[str | Path],
    group_b_dirs: list[str | Path],
    *,
    output_dir: str | Path,
    comparison_name: str = "neuromouse_comparison",
    sampling_rate: float | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    neuromouse_dir = output_dir / "neuromouse"
    group_a_out = neuromouse_dir / "group_A"
    group_b_out = neuromouse_dir / "group_B"
    group_a_out.mkdir(parents=True, exist_ok=True)
    group_b_out.mkdir(parents=True, exist_ok=True)

    datasets = []
    for i, p in enumerate(group_a_dirs):
        dsid = f"A_{i+1}_{Path(p).name}"
        out = group_a_out / dsid
        paths = write_speedmouse_dataset(p, out, dataset_id=dsid, sampling_rate=sampling_rate)
        data_url = f"/api/file?path={quote(paths['data_json'])}"
        datasets.append({"group": "A", "dataset_id": dsid, "recording_dir": str(Path(p)), **paths, "data_json_url": data_url})
    for i, p in enumerate(group_b_dirs):
        dsid = f"B_{i+1}_{Path(p).name}"
        out = group_b_out / dsid
        paths = write_speedmouse_dataset(p, out, dataset_id=dsid, sampling_rate=sampling_rate)
        data_url = f"/api/file?path={quote(paths['data_json'])}"
        datasets.append({"group": "B", "dataset_id": dsid, "recording_dir": str(Path(p)), **paths, "data_json_url": data_url})

    comparison_result = run_comparative_analysis(
        group_a_dirs,
        group_b_dirs,
        output_dir=output_dir / "comparison_metrics",
        comparison_name=comparison_name,
        sampling_rate=sampling_rate,
    )

    manifest = {
        "schema": "neuromouse.comparison.v1",
        "comparison_name": comparison_name,
        "datasets": datasets,
        "comparison_result": comparison_result,
        "compatibility_note": "Feature-level comparisons are safe across variable electrode counts. Same-channel views require matching channel names/layouts.",
    }
    manifest_path = neuromouse_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = output_dir / "neuromouse_sessions.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in datasets:
            data_path = Path(item["data_json"])
            arcname = f"{item['group']}/{item['dataset_id']}/data.json"
            zf.write(data_path, arcname)
        zf.write(manifest_path, "comparison_manifest.json")
        for key in ("summary_csv", "features_csv", "report_html"):
            p = comparison_result.get(key)
            if p and Path(p).exists():
                zf.write(p, f"comparison_metrics/{Path(p).name}")

    return {
        "neuromouse_dir": str(neuromouse_dir),
        "comparison_manifest": str(manifest_path),
        "sessions_zip": str(zip_path),
        "datasets": datasets,
        "comparison_result": comparison_result,
    }
