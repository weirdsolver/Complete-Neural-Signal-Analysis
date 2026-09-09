from __future__ import annotations

from pathlib import Path
from typing import Any

from neuro_importer.adapters import (
    CorticalLabsCL1Adapter,
    DSampMatAdapter,
    EEGLABAdapter,
    FieldTripAdapter,
    FinalSparkLiveMEAAdapter,
    GenericHDF5ContinuousAdapter,
    GenericMatAdapter,
    MNERawAdapter,
    MaxwellHDMEAAdapter,
    NWBContinuousAdapter,
    NumpyArrayAdapter,
    TableAdapter,
)
from neuro_importer.adapters.base import BaseAdapter
from neuro_importer.config import deep_merge, load_project_config
from neuro_importer.mapping import load_mapping, MappingSpec
from neuro_importer.adapters.mapping_adapter import MappingAdapter
from neuro_importer.units import UnitCalibration, calibrate_recording
from neuro_importer.provenance import sha256_file, stable_json_hash
from neuro_importer.progress import ProgressCallback, ProgressReporter
from neuro_importer.plugins import load_plugin_adapters
from neuro_importer.failure import write_failure_artifacts
from neuro_importer.detect import detect_file_type
from neuro_importer.export import CanonicalExporter, NeuralSignalExporter
from neuro_importer.inspect import write_file_tree_report
from neuro_importer.preprocess import apply_preprocessing
from neuro_importer.qc import write_qc_report
from neuro_importer.readers import EDFReader, HDF5Reader, MatReader, MNEReader, NumpyReader, TableReader
from neuro_importer.validate import validate_recording
from neuro_importer.windows import TensorExporter


class NeuroImportPipeline:
    """High-level continuous neural-signal import pipeline.

    v0.4 keeps the core invariant from v0.3: continuous signal first,
    no spike adapters, no stimulus/event alignment, no block labels.

    New v0.4 responsibilities:
    - optional project config
    - optional preprocessing on a copy of the raw Recording
    - optional QC reports
    - optional fixed-window tensor export
    - batch conversion through neuro_importer.batch
    """

    def __init__(self, adapters: list[BaseAdapter] | None = None, *, load_plugins: bool = True) -> None:
        self.mat_reader = MatReader()
        self.table_reader = TableReader()
        self.numpy_reader = NumpyReader()
        self.hdf5_reader = HDF5Reader()
        self.edf_reader = EDFReader()
        self.mne_reader = MNEReader()
        base_adapters = adapters or [
            DSampMatAdapter(),
            EEGLABAdapter(),
            FieldTripAdapter(),
            MNERawAdapter(),
            FinalSparkLiveMEAAdapter(),
            CorticalLabsCL1Adapter(),
            MaxwellHDMEAAdapter(),
            NWBContinuousAdapter(),
            GenericHDF5ContinuousAdapter(),
            TableAdapter(),
            NumpyArrayAdapter(),
            GenericMatAdapter(),
        ]
        plugin_adapters = load_plugin_adapters() if load_plugins else []
        self.adapters = [*plugin_adapters, *base_adapters]

    def read_raw(self, path: str | Path) -> Any:
        path = Path(path)
        file_type = detect_file_type(path)
        if file_type == "mat":
            return self.mat_reader.read(path)
        if file_type in {"table", "excel"}:
            return self.table_reader.read(path)
        if file_type == "numpy":
            return self.numpy_reader.read(path)
        if file_type == "hdf5":
            return self.hdf5_reader.read(path)
        if file_type == "edf":
            return self.edf_reader.read(path)
        if file_type == "mne":
            return self.mne_reader.read(path)
        raise ValueError(f"No reader implemented yet for file type: {file_type} ({path.suffix})")

    def inspect(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        raw = self.read_raw(path)
        scores = [adapter.score(raw) for adapter in self.adapters]
        scores = sorted(scores, key=lambda x: x.confidence, reverse=True)
        result: dict[str, Any] = {
            "path": str(path),
            "file_type": detect_file_type(path),
            "adapter_scores": [s.__dict__ for s in scores],
            "best_adapter": scores[0].name if scores and scores[0].confidence > 0 else None,
            "best_confidence": scores[0].confidence if scores else 0,
        }
        if isinstance(raw, dict) and raw.get("__file_type__") == "hdf5":
            try:
                from neuro_importer.inspect import summarize_hdf5_dict
                result["hdf5_tree"] = [node.__dict__ for node in summarize_hdf5_dict(raw)]
            except Exception as exc:
                result["hdf5_tree_error"] = str(exc)
        return result

    def choose_adapter(self, raw: Any, min_confidence: float = 0.5, mapping: MappingSpec | None = None) -> BaseAdapter:
        if mapping is not None:
            return MappingAdapter(mapping)

        scored = [(adapter, adapter.score(raw)) for adapter in self.adapters]
        scored.sort(key=lambda pair: pair[1].confidence, reverse=True)
        if not scored or scored[0][1].confidence < min_confidence:
            best = scored[0][1] if scored else None
            raise ValueError(
                "No adapter was confident enough to convert this file. "
                f"Best score: {best.confidence if best else 0}. "
                "Run inspect and add a mapping/adapter for this schema."
            )
        return scored[0][0]


    def generate_mapping_template(
        self,
        path: str | Path,
        *,
        output_dir: str | Path,
        sampling_rate: float | None = None,
    ) -> dict[str, str]:
        path = Path(path)
        raw = self.read_raw(path)
        return write_failure_artifacts(path, raw, output_dir, error=None, sampling_rate=sampling_rate)

    def convert(
        self,
        path: str | Path,
        *,
        output_dir: str | Path,
        subject: int | float | str | None = None,
        session: int | float | str | None = None,
        include_aux: bool = False,
        export_neural_signal: bool = False,
        min_confidence: float = 0.5,
        write_tree_report: bool = True,
        config_path: str | Path | None = None,
        preprocess_config: dict[str, Any] | None = None,
        qc_config: dict[str, Any] | None = None,
        window_config: dict[str, Any] | None = None,
        mapping_path: str | Path | None = None,
        export_config: dict[str, Any] | None = None,
        unit_config: dict[str, Any] | None = None,
        write_failure_report: bool = True,
        progress_callback: ProgressCallback | None = None,
        **adapter_kwargs: Any,
    ) -> dict[str, Any]:
        path = Path(path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        progress = ProgressReporter(progress_callback, total_steps=9)
        progress.running("Prepare configuration", 1, 3, message="Preparing configuration...")
        project_config = load_project_config(config_path)
        if preprocess_config is not None:
            project_config = deep_merge(project_config, {"preprocessing": preprocess_config})
        if qc_config is not None:
            project_config = deep_merge(project_config, {"qc": qc_config})
        if window_config is not None:
            project_config = deep_merge(project_config, {"windowing": window_config})
        if export_config is not None:
            project_config = deep_merge(project_config, {"export": export_config})
        if unit_config is not None:
            project_config = deep_merge(project_config, {"units": unit_config})
        progress.complete("Prepare configuration", 1, 8, message="Configuration ready.")

        progress.running("Read source file", 2, 10, message=f"Reading source file: {path.name}...")
        raw = self.read_raw(path)
        progress.complete("Read source file", 2, 20, message="Source file loaded.")

        progress.running("Select adapter", 3, 22, message="Selecting best adapter...")
        mapping = load_mapping(mapping_path) if mapping_path is not None else None
        try:
            adapter = self.choose_adapter(raw, min_confidence=min_confidence, mapping=mapping)
        except Exception as exc:
            progress.fail("Select adapter", 3, 22, message="Adapter selection failed.", detail=str(exc))
            if write_failure_report:
                failure_paths = write_failure_artifacts(path, raw, output_dir, error=str(exc), sampling_rate=adapter_kwargs.get("sampling_rate"))
                raise ValueError(f"No adapter was confident enough. Failure report written: {failure_paths}") from exc
            raise
        progress.complete("Select adapter", 3, 30, message=f"Adapter selected: {adapter.name}.")

        progress.running("Extract neural signal", 4, 32, message="Extracting continuous neural signal...")
        recording = adapter.convert(
            raw,
            source_path=path,
            subject=subject,
            session=session,
            include_aux=include_aux,
            **adapter_kwargs,
        )
        progress.complete("Extract neural signal", 4, 50, message="Continuous neural signal extracted.")

        progress.running("Validate and calibrate", 5, 52, message="Validating signal, time, and channels...")
        recording = validate_recording(recording)
        units_cfg = project_config.get("units", {}) or {}
        calibration = UnitCalibration.from_values(
            original_units=units_cfg.get("original_units"),
            target_units=units_cfg.get("target_units"),
            scale_factor=units_cfg.get("scale_factor"),
            offset=units_cfg.get("offset"),
        ) if any(k in units_cfg for k in ("original_units", "target_units", "scale_factor", "offset")) else None
        recording = calibrate_recording(recording, calibration)
        progress.complete("Validate and calibrate", 5, 60, message="Validation and unit calibration complete.")
        recording.metadata.setdefault("project_config", project_config)
        provenance_v05 = {
            "software_version": "0.5.5",
            "source_sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            "config_sha256": stable_json_hash(project_config),
            "mapping_sha256": sha256_file(mapping_path) if mapping_path is not None else None,
            "adapter_used": adapter.name,
            "continuous_signal_only": True,
        }
        recording.metadata.setdefault("provenance_v05", provenance_v05)

        exported: dict[str, Any] = {}
        export_cfg = project_config.get("export", {}) or {}
        exporter_kwargs = {
            "export_format": export_cfg.get("format", export_cfg.get("export_format", "npy")),
            "save_signal_csv": bool(export_cfg.get("save_signal_csv", False)),
            "csv_max_mb": export_cfg.get("csv_max_mb", 250.0),
            "compression": export_cfg.get("compression", "gzip"),
        }
        canonical_paths: dict[str, str]
        neural_paths: dict[str, str]
        final_recording = recording

        preprocessing_enabled = bool(project_config.get("preprocessing", {}).get("enabled", False))
        if preprocessing_enabled:
            progress.running("Preprocess signal", 6, 62, message="Writing raw signal and applying preprocessing...")
            raw_dir = output_dir / "raw"
            processed_dir = output_dir / "processed"
            exported["raw_paths"] = CanonicalExporter().export(recording, raw_dir, **exporter_kwargs)
            final_recording = apply_preprocessing(recording, project_config.get("preprocessing", {}))
            final_recording = validate_recording(final_recording)
            progress.complete("Preprocess signal", 6, 70, message="Preprocessing complete.")
            progress.running("Export canonical files", 7, 72, message="Exporting processed canonical files...")
            canonical_paths = CanonicalExporter().export(final_recording, processed_dir, **exporter_kwargs)
            neural_paths = NeuralSignalExporter().export(final_recording, processed_dir) if export_neural_signal else {}
            if write_tree_report and isinstance(raw, dict) and raw.get("__file_type__") == "hdf5":
                exported["raw_paths"]["file_tree_report"] = write_file_tree_report(raw, raw_dir)
                canonical_paths["file_tree_report"] = write_file_tree_report(raw, processed_dir)
            progress.complete("Export canonical files", 7, 82, message="Canonical files exported.")
        else:
            progress.complete("Preprocess signal", 6, 70, message="Preprocessing skipped.")
            progress.running("Export canonical files", 7, 72, message="Exporting signal.npy and canonical files...")
            canonical_paths = CanonicalExporter().export(final_recording, output_dir, **exporter_kwargs)
            if write_tree_report and isinstance(raw, dict) and raw.get("__file_type__") == "hdf5":
                canonical_paths["file_tree_report"] = write_file_tree_report(raw, output_dir)
            neural_paths = NeuralSignalExporter().export(final_recording, output_dir) if export_neural_signal else {}
            progress.complete("Export canonical files", 7, 82, message="Canonical files exported.")

        progress.running("Write QC and optional windows", 8, 84, message="Writing QC report and optional windows...")
        qc_paths: dict[str, str] = {}
        if bool(project_config.get("qc", {}).get("enabled", True)):
            qc_dir = (output_dir / "processed") if preprocessing_enabled else output_dir
            qc_paths = write_qc_report(final_recording, qc_dir, project_config.get("qc", {}))

        window_paths: dict[str, str] = {}
        if bool(project_config.get("windowing", {}).get("enabled", False)):
            win_dir = output_dir / "windows"
            window_paths = TensorExporter().export(final_recording, win_dir, project_config.get("windowing", {}))
            final_recording.quality.add_info("Window/tensor export completed.")
        progress.complete("Write QC and optional windows", 8, 95, message="QC report and optional outputs complete.")

        progress.complete("Complete", 9, 100, message="Conversion complete.")
        exported.update({
            "canonical_paths": canonical_paths,
            "neural_paths": neural_paths,
            "qc_paths": qc_paths,
            "window_paths": window_paths,
        })
        return {
            "adapter": adapter.name,
            "recording": final_recording,
            "raw_recording": recording,
            **exported,
            "quality": final_recording.quality.to_dict(),
        }
