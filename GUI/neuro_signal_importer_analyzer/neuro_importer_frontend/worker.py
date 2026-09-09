from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import QObject, Signal, Slot
except Exception:  # pragma: no cover
    class QObject:  # type: ignore
        pass
    class Signal:  # type: ignore
        def __init__(self, *a: Any, **k: Any): pass
    def Slot(*a: Any, **k: Any):
        def deco(fn: Any) -> Any:
            return fn
        return deco


class PipelineWorker(QObject):
    progress = Signal(str)          # compatibility/simple detail text
    progress_percent = Signal(int)  # overall percentage
    progress_event = Signal(dict)   # v0.5.5 structured heartbeat event
    finished = Signal(dict)
    failed = Signal(str)
    done = Signal()

    def __init__(self, *, inputs: list[str], output_dir: str, options: dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.output_dir = output_dir
        self.options = options

    def _emit_file_event(self, event: Any, *, item_index: int, total_items: int) -> None:
        """Forward a backend ProgressEvent as an overall GUI heartbeat."""
        try:
            data = event.to_dict()
        except Exception:
            data = dict(event) if isinstance(event, dict) else {"message": str(event), "percent": 0, "status": "heartbeat", "stage": "Pipeline"}
        file_percent = float(data.get("percent", 0) or 0)
        overall = int((((item_index - 1) + file_percent / 100.0) / max(total_items, 1)) * 100)
        data["item_index"] = item_index
        data["total_items"] = total_items
        data["overall_percent"] = max(0, min(100, overall))
        self.progress_event.emit(data)
        self.progress_percent.emit(data["overall_percent"])

    @Slot()
    def run(self) -> None:
        try:
            from neuro_importer.pipeline import NeuroImportPipeline
            from neuro_importer.batch import BatchConverter

            pipeline = NeuroImportPipeline()
            out = Path(self.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            results: dict[str, Any] = {"output_dir": str(out), "items": []}
            total = max(len(self.inputs), 1)
            self.progress_percent.emit(0)
            self.progress_event.emit({
                "stage": "Start",
                "status": "running",
                "step_index": 0,
                "total_steps": 9,
                "percent": 0,
                "overall_percent": 0,
                "message": "Starting conversion...",
            })

            for idx, item in enumerate(self.inputs, start=1):
                p = Path(item)
                self.progress.emit(f"Processing {idx}/{total}: {p}...")
                self.progress_event.emit({
                    "stage": "Start file",
                    "status": "running",
                    "step_index": 0,
                    "total_steps": 9,
                    "percent": 0,
                    "overall_percent": int((idx - 1) / total * 100),
                    "item_index": idx,
                    "total_items": total,
                    "message": f"Processing {p.name}...",
                })

                if p.is_dir():
                    # Folder batch conversion has its own manifest-level flow.  We still show a clean stage.
                    batch_out = out / p.name
                    self.progress_event.emit({
                        "stage": "Batch scan",
                        "status": "running",
                        "step_index": 1,
                        "total_steps": 3,
                        "percent": 10,
                        "overall_percent": int(((idx - 1) + 0.1) / total * 100),
                        "item_index": idx,
                        "total_items": total,
                        "message": f"Scanning folder {p.name}...",
                    })
                    res = BatchConverter(pipeline).convert_dataset(
                        p,
                        batch_out,
                        config_path=self.options.get("config_path") or None,
                        continue_on_error=True,
                    )
                    self.progress_event.emit({
                        "stage": "Batch complete",
                        "status": "complete",
                        "step_index": 3,
                        "total_steps": 3,
                        "percent": 100,
                        "overall_percent": int(idx / total * 100),
                        "item_index": idx,
                        "total_items": total,
                        "message": f"Folder {p.name} complete.",
                    })
                    results["items"].append({"input": str(p), "mode": "batch", "result": res})
                else:
                    rec_out = out / (p.stem + "_converted")
                    callback = lambda event, i=idx, t=total: self._emit_file_event(event, item_index=i, total_items=t)
                    res = pipeline.convert(
                        p,
                        output_dir=rec_out,
                        config_path=self.options.get("config_path") or None,
                        mapping_path=self.options.get("mapping_path") or None,
                        sampling_rate=self.options.get("sampling_rate"),
                        signal_path=self.options.get("signal_path") or None,
                        include_aux=bool(self.options.get("include_aux", False)),
                        preprocess_config=self.options.get("preprocess_config"),
                        window_config=self.options.get("window_config"),
                        export_config=self.options.get("export_config"),
                        unit_config=self.options.get("unit_config"),
                        export_neural_signal=bool(self.options.get("export_neural_signal", False)),
                        progress_callback=callback,
                    )
                    slim = {k: v for k, v in res.items() if k not in {"recording", "raw_recording"}}
                    results["items"].append({"input": str(p), "mode": "single", "result": slim})

            self.progress_percent.emit(100)
            self.progress_event.emit({
                "stage": "Complete",
                "status": "complete",
                "step_index": 9,
                "total_steps": 9,
                "percent": 100,
                "overall_percent": 100,
                "message": "Conversion complete.",
            })
            self.progress.emit("Conversion complete.")
            self.finished.emit(results)
        except Exception as exc:
            self.progress_event.emit({
                "stage": "Failure",
                "status": "failed",
                "step_index": 0,
                "total_steps": 0,
                "percent": 0,
                "overall_percent": 0,
                "message": str(exc),
            })
            self.failed.emit(str(exc))
        finally:
            self.done.emit()
