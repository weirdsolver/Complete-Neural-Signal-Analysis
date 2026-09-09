from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neuro_importer.batch import BatchConverter, scan_dataset
from neuro_importer.config import load_project_config, write_default_config
from neuro_importer.pipeline import NeuroImportPipeline


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.size <= 64:
                return value.tolist()
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k not in {"recording", "raw_recording"}}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _window_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if not getattr(args, "make_windows", False) and args.window_seconds is None and args.window_samples is None:
        return None
    cfg: dict[str, Any] = {"enabled": True}
    if args.window_seconds is not None:
        cfg["window_seconds"] = args.window_seconds
    if args.step_seconds is not None:
        cfg["step_seconds"] = args.step_seconds
    if args.window_samples is not None:
        cfg["window_samples"] = args.window_samples
    if args.step_samples is not None:
        cfg["step_samples"] = args.step_samples
    return cfg


def _preprocess_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if not getattr(args, "preprocess", False):
        return None
    cfg: dict[str, Any] = {"enabled": True}
    if getattr(args, "demean", False):
        cfg["demean"] = True
    if getattr(args, "detrend", False):
        cfg["detrend"] = True
    if getattr(args, "notch_hz", None) is not None:
        cfg["notch_hz"] = args.notch_hz
    if getattr(args, "bandpass_hz", None) is not None:
        parts = [float(x.strip()) for x in str(args.bandpass_hz).replace(",", "-").split("-") if x.strip()]
        if len(parts) != 2:
            raise ValueError("--bandpass-hz must look like 1-100")
        cfg["bandpass_hz"] = parts
    if getattr(args, "downsample_to_hz", None) is not None:
        cfg["downsample_to_hz"] = args.downsample_to_hz
    if getattr(args, "normalization", None) is not None:
        cfg["normalization"] = args.normalization
    return cfg


def _export_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    cfg: dict[str, Any] = {}
    if getattr(args, "export_format", None) is not None:
        cfg["format"] = args.export_format
    if getattr(args, "no_signal_csv", False):
        cfg["save_signal_csv"] = False
    if getattr(args, "csv_max_mb", None) is not None:
        cfg["csv_max_mb"] = args.csv_max_mb
    if getattr(args, "compression", None) is not None:
        cfg["compression"] = args.compression
    return cfg or None


def _unit_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    cfg: dict[str, Any] = {}
    for key in ("original_units", "target_units", "scale_factor", "offset"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg or None


def add_common_convert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Optional project YAML config")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--include-aux", action="store_true", help="Keep auxiliary channels such as BIP/RESP/AUX/EOG/EMG/ECG")
    parser.add_argument("--no-neural-export", action="store_true", help="Only write canonical files")
    parser.add_argument("--sampling-rate", type=float, default=None, help="Override/provide sampling rate for formats that lack it")
    parser.add_argument("--signal-path", default=None, help="For generic MAT/HDF5 adapters: exact candidate path to use as signal")
    parser.add_argument("--time-column", default=None, help="For TableAdapter: column to use as time")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum adapter confidence required for conversion")
    parser.add_argument("--no-tree-report", action="store_true", help="Skip HDF5 file_tree_report.json output")
    parser.add_argument("--no-scaling", action="store_true", help="Do not apply explicit voltage scaling attributes when adapters find them")
    parser.add_argument("--no-qc", action="store_true", help="Skip QC report generation")
    parser.add_argument("--mapping", default=None, help="Manual mapping YAML for unknown files")

    parser.add_argument("--original-units", default=None, help="Original signal units, e.g. adc_counts, microvolts, millivolts, volts")
    parser.add_argument("--target-units", default=None, help="Target units to write in metadata, e.g. microvolts")
    parser.add_argument("--scale-factor", type=float, default=None, help="Apply signal = signal * scale_factor before unit conversion")
    parser.add_argument("--offset", type=float, default=None, help="Apply signal = signal + offset after scale factor")

    parser.add_argument("--export-format", choices=["npy", "memmap", "hdf5", "zarr"], default=None, help="Signal export backend")
    parser.add_argument("--no-signal-csv", action="store_true", help="Skip signal.csv even for small files")
    parser.add_argument("--csv-max-mb", type=float, default=None, help="Skip signal.csv above this in-memory signal size")
    parser.add_argument("--compression", default=None, help="Compression for hdf5 export, e.g. gzip or lzf")

    parser.add_argument("--preprocess", action="store_true", help="Enable preprocessing using CLI flags/defaults")
    parser.add_argument("--demean", action="store_true")
    parser.add_argument("--detrend", action="store_true")
    parser.add_argument("--notch-hz", type=float, default=None)
    parser.add_argument("--bandpass-hz", default=None, help="Bandpass like 1-100")
    parser.add_argument("--downsample-to-hz", type=float, default=None)
    parser.add_argument("--normalization", choices=["zscore", "robust", "minmax"], default=None)
    parser.add_argument("--make-windows", action="store_true", help="Export fixed windows/tensors")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--step-seconds", type=float, default=None)
    parser.add_argument("--window-samples", type=int, default=None)
    parser.add_argument("--step-samples", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuous neural signal importer")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect a file and score available adapters")
    inspect.add_argument("path")

    mapping = sub.add_parser("generate-mapping", help="Generate a mapping YAML template for an unknown file")
    mapping.add_argument("path")
    mapping.add_argument("--output", "-o", required=True, help="Output directory for failure_report.json and mapping_template.yaml")
    mapping.add_argument("--sampling-rate", type=float, default=None)

    scan = sub.add_parser("scan", help="Scan a dataset folder for supported neural signal files")
    scan.add_argument("input_root")
    scan.add_argument("--config", default=None)
    scan.add_argument("--output", "-o", default=None, help="Optional CSV path for the scan table")

    default_config = sub.add_parser("write-default-config", help="Write a default project YAML config")
    default_config.add_argument("path")

    convert = sub.add_parser("convert", help="Convert a supported file to canonical continuous neural-signal outputs")
    convert.add_argument("path")
    convert.add_argument("--output", "-o", required=True)
    add_common_convert_args(convert)

    batch = sub.add_parser("batch", help="Batch-convert a folder of supported continuous neural signal files")
    batch.add_argument("input_root")
    batch.add_argument("--output", "-o", required=True)
    batch.add_argument("--config", default=None, help="Optional project YAML config")
    batch.add_argument("--stop-on-error", action="store_true")
    batch.add_argument("--subject", default=None)
    batch.add_argument("--session", default=None)

    gui = sub.add_parser("gui", help="Launch optional desktop frontend")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "gui":
        from neuro_importer_frontend.desktop_app import main as gui_main
        gui_main()
        return

    if args.command == "write-default-config":
        path = write_default_config(args.path)
        print(json.dumps({"default_config": path}, indent=2))
        return

    if args.command == "scan":
        config = load_project_config(args.config)
        df = scan_dataset(args.input_root, config)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(json.dumps({"n_files": int(len(df)), "scan_csv": str(out)}, indent=2))
        else:
            print(df.to_json(orient="records", indent=2))
        return

    pipeline = NeuroImportPipeline()

    if args.command == "inspect":
        result = pipeline.inspect(args.path)
        print(json.dumps(_json_safe(result), indent=2))
        return

    if args.command == "generate-mapping":
        result = pipeline.generate_mapping_template(args.path, output_dir=args.output, sampling_rate=args.sampling_rate)
        print(json.dumps(_json_safe(result), indent=2))
        return

    if args.command == "convert":
        qc_cfg = {"enabled": not args.no_qc}
        preprocess_cfg = _preprocess_override_from_args(args)
        window_cfg = _window_override_from_args(args)
        export_cfg = _export_override_from_args(args)
        unit_cfg = _unit_override_from_args(args)
        result = pipeline.convert(
            args.path,
            output_dir=args.output,
            config_path=args.config,
            subject=args.subject,
            session=args.session,
            include_aux=args.include_aux,
            export_neural_signal=not args.no_neural_export,
            sampling_rate=args.sampling_rate,
            signal_path=args.signal_path,
            time_column=args.time_column,
            min_confidence=args.min_confidence,
            write_tree_report=not args.no_tree_report,
            apply_scaling=not args.no_scaling,
            preprocess_config=preprocess_cfg,
            qc_config=qc_cfg,
            window_config=window_cfg,
            export_config=export_cfg,
            unit_config=unit_cfg,
            mapping_path=args.mapping,
        )
        printable = {k: v for k, v in result.items() if k not in {"recording", "raw_recording"}}
        print(json.dumps(_json_safe(printable), indent=2))
        return

    if args.command == "batch":
        result = BatchConverter(pipeline).convert_dataset(
            args.input_root,
            args.output,
            config_path=args.config,
            continue_on_error=not args.stop_on_error,
            subject=args.subject,
            session=args.session,
        )
        print(json.dumps(_json_safe(result), indent=2))
        return


if __name__ == "__main__":
    main()
