from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from neuro_importer.batch.dataset_scanner import scan_dataset
from neuro_importer.batch.manifest_builder import manifest_row_from_result, write_manifest
from neuro_importer.config import load_project_config


def safe_recording_id(relative_path: str, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(relative_path).with_suffix("").as_posix())
    stem = stem.strip("._-") or f"recording_{index:05d}"
    return f"{index:05d}_{stem.replace('/', '__')}"


class BatchConverter:
    """Dataset-level continuous-signal batch converter."""

    def __init__(self, pipeline: Any | None = None) -> None:
        if pipeline is None:
            from neuro_importer.pipeline import NeuroImportPipeline
            pipeline = NeuroImportPipeline()
        self.pipeline = pipeline

    def convert_dataset(
        self,
        input_root: str | Path,
        output_root: str | Path,
        *,
        config_path: str | Path | None = None,
        continue_on_error: bool = True,
        subject: str | int | float | None = None,
        session: str | int | float | None = None,
    ) -> dict[str, Any]:
        config = load_project_config(config_path)
        out_root = Path(output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        scan = scan_dataset(input_root, config)
        rows: list[dict[str, Any]] = []

        for i, row in scan.reset_index(drop=True).iterrows():
            source_file = row["source_file"]
            recording_id = safe_recording_id(row["relative_path"], int(i))
            rec_out = out_root / recording_id
            try:
                result = self.pipeline.convert(
                    source_file,
                    output_dir=rec_out,
                    subject=subject,
                    session=session,
                    include_aux=bool(config["conversion"].get("include_aux", False)),
                    export_neural_signal=bool(config["conversion"].get("export_neural_signal", True)),
                    min_confidence=float(config["conversion"].get("min_confidence", 0.5)),
                    write_tree_report=bool(config["conversion"].get("write_tree_report", True)),
                    preprocess_config=config.get("preprocessing", {}),
                    qc_config=config.get("qc", {}),
                    window_config=config.get("windowing", {}),
                    export_config=config.get("export", {}),
                    unit_config=config.get("units", {}),
                )
                rows.append(manifest_row_from_result(
                    recording_id=recording_id,
                    source_file=source_file,
                    output_dir=rec_out,
                    status="converted",
                    result=result,
                ))
            except Exception as exc:
                rows.append(manifest_row_from_result(
                    recording_id=recording_id,
                    source_file=source_file,
                    output_dir=rec_out,
                    status="failed",
                    error=str(exc),
                ))
                if not continue_on_error:
                    raise

        manifest_path = write_manifest(rows, out_root)
        scan_path = out_root / "dataset_scan.csv"
        scan.to_csv(scan_path, index=False)
        config_path_out = out_root / "effective_config.json"
        config_path_out.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return {
            "input_root": str(input_root),
            "output_root": str(out_root),
            "n_scanned": int(len(scan)),
            "n_converted": int(sum(r["status"] == "converted" for r in rows)),
            "n_failed": int(sum(r["status"] == "failed" for r in rows)),
            "manifest": str(manifest_path),
            "dataset_scan": str(scan_path),
            "effective_config": str(config_path_out),
            "rows": rows,
        }
