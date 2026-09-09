from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neuro_importer.core.recording import Recording
from neuro_importer.export.large_file_exporter import array_size_mb, save_signal_array


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.size <= 10000:
                return value.tolist()
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class CanonicalExporter:
    """Write canonical continuous neural-signal outputs.

    v0.5 adds large-file aware export controls. Defaults preserve the older
    v0.4 behavior: signal.npy plus signal.csv.
    """

    def export(
        self,
        recording: Recording,
        output_dir: str | Path,
        *,
        export_format: str = "npy",
        save_signal_csv: bool = True,
        csv_max_mb: float | None = 250.0,
        compression: str | None = "gzip",
    ) -> dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {
            "time": out / "time.npy",
            "channels": out / "channels.csv",
            "metadata": out / "metadata.json",
            "quality_report": out / "quality_report.json",
            "provenance": out / "provenance.json",
            "export_report": out / "export_report.json",
        }

        signal_path, signal_info = save_signal_array(recording.signal, out, export_format=export_format, compression=compression)
        result: dict[str, str] = {"signal": signal_path}
        np.save(paths["time"], recording.effective_time())
        recording.channels.to_csv(paths["channels"], index=False)
        if recording.electrodes is not None:
            paths["electrodes"] = out / "electrodes.csv"
            recording.electrodes.to_csv(paths["electrodes"], index=False)

        should_save_csv = bool(save_signal_csv)
        if should_save_csv and csv_max_mb is not None and array_size_mb(recording.signal) > float(csv_max_mb):
            should_save_csv = False
            recording.quality.add_warning(
                f"Skipped signal.csv because signal size is {array_size_mb(recording.signal):.2f} MB > csv_max_mb={csv_max_mb}."
            )
        if should_save_csv:
            paths["signal_csv"] = out / "signal.csv"
            recording.to_dataframe(time_column="Time", time_last=False).to_csv(paths["signal_csv"], index=False)

        export_report = {
            "signal": signal_info,
            "saved_signal_csv": should_save_csv,
            "csv_max_mb": csv_max_mb,
        }
        paths["metadata"].write_text(json.dumps(_json_safe(recording.metadata), indent=2), encoding="utf-8")
        paths["quality_report"].write_text(json.dumps(_json_safe(recording.quality.to_dict()), indent=2), encoding="utf-8")
        paths["provenance"].write_text(json.dumps(_json_safe(recording.provenance()), indent=2), encoding="utf-8")
        paths["export_report"].write_text(json.dumps(_json_safe(export_report), indent=2), encoding="utf-8")

        result.update({name: str(path) for name, path in paths.items()})
        return result
