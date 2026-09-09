from __future__ import annotations

import datetime as _dt
import json
import os
import queue
import shutil
import zipfile
import subprocess
import sys
import threading
import time
from urllib.parse import quote
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from neuro_importer.pipeline import NeuroImportPipeline
from neuro_importer.progress import ProgressEvent
from neuro_importer_analysis import run_comparative_analysis
from neuro_importer_neuromouse import write_speedmouse_dataset, build_speedmouse_comparison_pack


APP_VERSION = "0.11.33"

SUPPORTED_UPLOAD_SUFFIXES = {
    # Archive uploads; extracted safely before primary-file discovery
    ".zip",
    # Primary signal/header files
    ".mat", ".h5", ".hdf5", ".nwb", ".npy", ".npz", ".csv", ".tsv", ".xlsx", ".xls",
    ".edf", ".bdf", ".set", ".vhdr", ".fif",
    # Sidecars that should be uploaded/copied but not converted as standalone recordings
    ".fdt", ".eeg", ".vmrk", ".json", ".txt",
}

PRIMARY_SIGNAL_SUFFIXES = {
    ".mat", ".h5", ".hdf5", ".nwb", ".npy", ".npz", ".csv", ".tsv", ".xlsx", ".xls",
    ".edf", ".bdf", ".set", ".vhdr", ".fif",
}

BIDS_SIDECAR_NAME_ENDINGS = (
    "_channels.tsv", "_electrodes.tsv", "_events.tsv", "_coordsystem.json",
    "_eeg.json", "_ieeg.json", "_meg.json",
)


ARCHIVE_SUFFIXES = {".zip"}


def _safe_extract_zip(zip_path: Path) -> dict[str, Any]:
    """Safely extract a user-uploaded zip archive next to the upload.

    Zip files are containers, not neural signal recordings. This helper extracts
    them into a deterministic sibling folder and returns the extracted files so
    normal primary-file discovery can find .mat/.edf/.set/etc. entries inside.
    It rejects path traversal entries and skips directory entries.
    """
    zip_path = Path(zip_path)
    extract_root = zip_path.with_suffix("")
    extract_root = zip_path.parent / f"{extract_root.name}_archive_contents"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    extracted_files: list[str] = []
    skipped_entries: list[dict[str, str]] = []
    root_resolved = extract_root.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir():
                continue
            # Normalize archive paths and block absolute/path-traversal members.
            parts = [part for part in Path(name.replace("\\", "/")).parts if part not in ("", ".")]
            if not parts or any(part == ".." for part in parts):
                skipped_entries.append({"entry": name, "reason": "unsafe_archive_path"})
                continue
            target = extract_root.joinpath(*parts)
            try:
                resolved = target.resolve()
                if root_resolved not in resolved.parents and resolved != root_resolved:
                    skipped_entries.append({"entry": name, "reason": "unsafe_archive_path"})
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_files.append(str(target))
            except Exception as exc:
                skipped_entries.append({"entry": name, "reason": f"extract_failed:{exc!r}"})
    return {
        "archive": str(zip_path),
        "extract_dir": str(extract_root),
        "extracted_files": extracted_files,
        "skipped_entries": skipped_entries,
    }


def is_primary_signal_file(path: str | Path) -> bool:
    """Return True when a path should be converted as a recording.

    Uploads often include mandatory sidecars such as EEGLAB .fdt or BIDS
    *_channels.tsv files. Those files must remain next to the primary file,
    but converting them as standalone recordings causes false failures.
    """
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".fif.gz"):
        return True
    if name.endswith(BIDS_SIDECAR_NAME_ENDINGS):
        return False
    if p.suffix.lower() in {".fdt", ".eeg", ".vmrk", ".json"}:
        return False
    return p.suffix.lower() in PRIMARY_SIGNAL_SUFFIXES


def _matching_primary_for_sidecar(path: Path) -> Path | None:
    """Return a nearby primary/header file for a known sidecar when possible."""
    suffix = path.suffix.lower()
    if suffix == ".fdt":
        candidate = path.with_suffix(".set")
        return candidate if candidate.exists() else None
    if suffix in {".eeg", ".vmrk"}:
        candidate = path.with_suffix(".vhdr")
        return candidate if candidate.exists() else None
    return None


def discover_primary_signal_files(input_paths: list[str | Path]) -> dict[str, Any]:
    """Classify requested inputs into primary conversion files and sidecars.

    This is intentionally defensive. Even if the browser upload layer or a user
    passes sidecars such as .fdt directly, the job manager must not try to
    convert those sidecars as independent recordings. If a matching primary
    header is next to the sidecar, we promote that header instead.
    """
    primary: list[Path] = []
    skipped: list[dict[str, str]] = []
    promoted: list[dict[str, str]] = []
    archives: list[dict[str, Any]] = []
    requested = [str(Path(p)) for p in input_paths]

    def classify_path(path: Path, *, from_archive: str | None = None) -> None:
        if path.is_dir():
            for child in path.rglob("*"):
                if not child.is_file():
                    continue
                if is_primary_signal_file(child):
                    primary.append(child)
                else:
                    item = {"path": str(child), "reason": "sidecar_or_non_primary_in_folder"}
                    if from_archive:
                        item["archive"] = from_archive
                    skipped.append(item)
            return

        if not path.is_file():
            skipped.append({"path": str(path), "reason": "missing_or_not_a_file"})
            return

        if path.suffix.lower() in ARCHIVE_SUFFIXES:
            try:
                archive_info = _safe_extract_zip(path)
                archives.append(archive_info)
                classify_path(Path(archive_info["extract_dir"]), from_archive=str(path))
            except Exception as exc:
                skipped.append({"path": str(path), "reason": f"archive_extract_failed:{exc!r}"})
            return

        if is_primary_signal_file(path):
            primary.append(path)
            return

        match = _matching_primary_for_sidecar(path)
        if match and match.exists():
            primary.append(match)
            promoted.append({"sidecar": str(path), "primary": str(match), "reason": "matched_sidecar_to_primary_header"})
            skipped.append({"path": str(path), "reason": "sidecar_promoted_to_primary_header"})
        else:
            skipped.append({"path": str(path), "reason": "sidecar_or_non_primary"})

    for raw in input_paths:
        classify_path(Path(raw))

    # Deduplicate by resolved path while preserving deterministic order.
    deduped: dict[str, Path] = {}
    for p in primary:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        deduped[key] = p
    primary_files = sorted(deduped.values(), key=lambda x: str(x).lower())

    return {
        "requested_paths": requested,
        "primary_files": primary_files,
        "skipped": skipped,
        "promoted": promoted,
        "archives": archives,
    }



@dataclass
class JobRecord:
    job_id: str
    kind: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress_percent: int = 0
    current_step: str = "Queued"
    output_dir: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    raw_log_path: str | None = None
    raw_jsonl_path: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class JobManager:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, JobRecord] = {}
        self.event_queues: dict[str, list[queue.Queue]] = {}
        self.lock = threading.RLock()
        self.live_processes: dict[str, list[subprocess.Popen]] = {}

    def create_job(self, kind: str, output_dir: str | Path | None = None) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        job_output = Path(output_dir).expanduser().resolve() if output_dir else self.workspace / "outputs" / job_id
        job_output.mkdir(parents=True, exist_ok=True)
        raw_log_path = job_output / "job_log.txt"
        raw_jsonl_path = job_output / "job_log.jsonl"
        record = JobRecord(
            job_id=job_id,
            kind=kind,
            output_dir=str(job_output),
            raw_log_path=str(raw_log_path),
            raw_jsonl_path=str(raw_jsonl_path),
        )
        with self.lock:
            self.jobs[job_id] = record
            self.event_queues[job_id] = []
        self._init_raw_logs(record)
        self.emit(job_id, "state", {"status": "queued", "message": "Job queued.", "percent": 0})
        return record

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.event_queues.setdefault(job_id, []).append(q)
            record = self.jobs.get(job_id)
            if record:
                for event in record.events[-25:]:
                    q.put(event)
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self.lock:
            queues = self.event_queues.get(job_id, [])
            if q in queues:
                queues.remove(q)

    def _json_safe(self, value: Any) -> Any:
        """Return a JSON-serializable copy suitable for raw job logs."""
        try:
            json.dumps(value)
            return value
        except TypeError:
            pass
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        return repr(value)

    def _init_raw_logs(self, record: JobRecord) -> None:
        """Create the per-job human-readable and JSONL raw backend logs."""
        if not record.raw_log_path or not record.raw_jsonl_path:
            return
        try:
            created = _dt.datetime.fromtimestamp(record.created_at).isoformat(timespec="seconds")
            header = (
                "# Neuro Signal raw job log\n"
                f"job_id: {record.job_id}\n"
                f"kind: {record.kind}\n"
                f"created_at: {created}\n"
                f"output_dir: {record.output_dir}\n"
                "\n"
            )
            Path(record.raw_log_path).write_text(header, encoding="utf-8")
            Path(record.raw_jsonl_path).write_text("", encoding="utf-8")
        except Exception:
            # Logging must never prevent the actual analysis job from starting.
            pass

    def _append_raw_log(self, record: JobRecord, payload: dict[str, Any]) -> None:
        """Append one event to job_log.txt and job_log.jsonl.

        The log is a backend/pipeline trace, not a dump of neural samples.
        It records steps, adapter decisions, paths, generated NeuroMouse URLs,
        warnings, and errors so users can debug a completed job later.
        """
        try:
            safe_payload = self._json_safe(payload)
            iso = _dt.datetime.fromtimestamp(float(payload.get("time", time.time()))).isoformat(timespec="seconds")
            status = payload.get("status", "")
            percent = payload.get("percent", "")
            step = payload.get("step") or payload.get("stage") or payload.get("type", "event")
            message = payload.get("message", "")
            line = f"[{iso}] {payload.get('type', 'event')}"
            if status:
                line += f" status={status}"
            if percent != "":
                line += f" percent={percent}"
            if step:
                line += f" step={step}"
            if message:
                line += f" message={message}"
            if payload.get("output_dir"):
                line += f" output_dir={payload.get('output_dir')}"
            if payload.get("error"):
                line += f" error={payload.get('error')}"
            if payload.get("detail") not in (None, ""):
                line += f" detail={payload.get('detail')}"
            line += "\n"
            extra_lines: list[str] = []
            if payload.get("app_version"):
                extra_lines.append(f"  app_version: {payload.get('app_version')}\n")
            if payload.get("requested_paths"):
                extra_lines.append("  requested_paths:\n")
                for item in payload.get("requested_paths") or []:
                    extra_lines.append(f"    - {item}\n")
            if payload.get("primary_files"):
                extra_lines.append("  primary_files_selected_for_conversion:\n")
                for item in payload.get("primary_files") or []:
                    extra_lines.append(f"    - {item}\n")
            if payload.get("skipped"):
                extra_lines.append("  skipped_sidecars_or_non_primary_files:\n")
                for item in payload.get("skipped") or []:
                    if isinstance(item, dict):
                        extra_lines.append(f"    - {item.get('path')} reason={item.get('reason')}\n")
                    else:
                        extra_lines.append(f"    - {item}\n")
            if payload.get("promoted"):
                extra_lines.append("  sidecars_promoted_to_primary_headers:\n")
                for item in payload.get("promoted") or []:
                    if isinstance(item, dict):
                        extra_lines.append(f"    - sidecar={item.get('sidecar')} primary={item.get('primary')} reason={item.get('reason')}\n")
                    else:
                        extra_lines.append(f"    - {item}\n")
            if payload.get("archives"):
                extra_lines.append("  archives_extracted:\n")
                for item in payload.get("archives") or []:
                    if isinstance(item, dict):
                        extra_lines.append(f"    - archive={item.get('archive')} extract_dir={item.get('extract_dir')} extracted_files={len(item.get('extracted_files') or [])} skipped_entries={len(item.get('skipped_entries') or [])}\n")
                    else:
                        extra_lines.append(f"    - {item}\n")
            if record.raw_log_path:
                with Path(record.raw_log_path).open("a", encoding="utf-8") as f:
                    f.write(line)
                    if extra_lines:
                        f.writelines(extra_lines)
            if record.raw_jsonl_path:
                with Path(record.raw_jsonl_path).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(safe_payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            # Raw logging is diagnostic only. Never let it break conversion/analysis.
            pass

    def _coerce_progress_percent(self, data: dict[str, Any]) -> int | None:
        """Return a safe 0-100 integer progress value for job events.

        Some backend progress events provide an explicit percent, while others
        only provide step_index/total_steps or item_index/total_items.  This
        normalizer keeps the browser progress bar honest: running events stay
        within 0-99, terminal events are exactly 100, and bad values cannot
        raise or push the bar outside its bounds.
        """
        status = str(data.get("status") or "").lower()
        if status in {"complete", "completed", "success", "finished", "failed", "stopped"}:
            return 100

        percent = data.get("percent")
        if percent is None:
            for idx_key, total_key in (("step_index", "total_steps"), ("item_index", "total_items")):
                idx = data.get(idx_key)
                total = data.get(total_key)
                try:
                    idx_f = float(idx)
                    total_f = float(total)
                except (TypeError, ValueError):
                    continue
                if total_f > 0:
                    # step_index/item_index may be zero- or one-based. Treat 0 as
                    # the start of work and total as the last completed unit.
                    idx_f = max(0.0, min(idx_f, total_f))
                    percent = 100.0 * idx_f / total_f
                    break

        if percent is None:
            return None

        try:
            value = int(round(float(percent)))
        except (TypeError, ValueError, OverflowError):
            return None

        value = max(0, min(100, value))
        if status in {"running", "queued"}:
            value = min(value, 99)
        return value

    def emit(self, job_id: str, event_type: str, data: dict[str, Any]) -> None:
        safe_data = dict(data)
        normalized_percent = self._coerce_progress_percent(safe_data)
        if normalized_percent is not None:
            safe_data["percent"] = normalized_percent
        payload = {"type": event_type, "job_id": job_id, "time": time.time(), **safe_data}
        with self.lock:
            record = self.jobs.get(job_id)
            if record:
                record.events.append(payload)
                record.updated_at = time.time()

                # Only state events are allowed to change the job's lifecycle.
                # Progress callbacks can report per-step `status=complete` and
                # `percent=100`; treating those as job completion made the
                # browser loader close early or jump backward during multi-file
                # conversions.  Raw-log warnings likewise must not become the
                # overall job status.
                if event_type == "state" and "status" in safe_data:
                    record.status = str(safe_data["status"])
                elif (
                    event_type == "progress"
                    and record.status == "queued"
                    and str(safe_data.get("status") or "").lower() == "running"
                ):
                    record.status = "running"

                if "percent" in safe_data and safe_data["percent"] is not None:
                    incoming_percent = int(safe_data["percent"])
                    if event_type == "state" or incoming_percent >= record.progress_percent:
                        record.progress_percent = incoming_percent

                if "step" in safe_data and safe_data["step"]:
                    record.current_step = str(safe_data["step"])
                if "output_dir" in safe_data and safe_data["output_dir"]:
                    record.output_dir = str(safe_data["output_dir"])
                if "result" in safe_data:
                    record.result = safe_data["result"]
                if event_type == "state" and "error" in safe_data:
                    record.error = str(safe_data["error"])
                self._append_raw_log(record, payload)
            for q in self.event_queues.get(job_id, []):
                q.put(payload)

    def get(self, job_id: str) -> JobRecord | None:
        with self.lock:
            return self.jobs.get(job_id)

    @staticmethod
    def _conversion_step_percent(step: str, status: str, raw_percent: Any) -> int:
        """Return a per-file conversion percent that never uses sub-step 100s as job 100.

        The converter emits events such as `Read source file: running 10` and
        then `Read source file: complete 100`.  That 100 means the *sub-step* is
        done, not the whole conversion job.  Map sub-step completion to the next
        stable conversion milestone so the global loader can remain monotonic.
        """
        text = str(step or "").lower()
        status_text = str(status or "").lower()
        is_step_complete = status_text in {"complete", "completed", "success", "finished"}

        complete_milestones = [
            ("prepare configuration", 8),
            ("read source file", 20),
            ("select adapter", 30),
            ("extract neural signal", 48),
            ("validate and calibrate", 65),
            ("preprocess signal", 70),
            ("export canonical files", 82),
            ("write qc", 95),
            ("complete", 100),
        ]
        if is_step_complete:
            for key, value in complete_milestones:
                if key in text:
                    return value

        try:
            value = int(round(float(raw_percent)))
        except (TypeError, ValueError, OverflowError):
            value = 0
        return max(0, min(100, value))

    @staticmethod
    def _map_percent_to_range(percent: int | float | None, start: int, end: int) -> int:
        if percent is None:
            percent = 0
        try:
            local = max(0.0, min(100.0, float(percent)))
        except (TypeError, ValueError, OverflowError):
            local = 0.0
        if end <= start:
            return int(max(0, min(99, start)))
        mapped = start + (end - start) * (local / 100.0)
        return int(max(0, min(99, round(mapped))))

    def _progress_callback(self, job_id: str, *, percent_start: int = 0, percent_end: int = 99) -> Callable[[ProgressEvent], None]:
        def callback(event: ProgressEvent) -> None:
            """Normalize backend progress events for the HTML job stream.

            Converter progress is local to one input file.  The web progress bar
            is global to the whole job, so local per-step percentages are mapped
            into an overall file slice and sub-step `complete 100` events are not
            allowed to mark the job as finished.
            """
            try:
                if hasattr(event, "to_dict"):
                    data = event.to_dict()
                elif isinstance(event, dict):
                    data = dict(event)
                else:
                    data = {
                        "stage": getattr(event, "stage", None)
                        or getattr(event, "step", None)
                        or getattr(event, "step_name", None)
                        or "Pipeline progress",
                        "status": getattr(event, "status", "running"),
                        "step_index": getattr(event, "step_index", None),
                        "total_steps": getattr(event, "total_steps", None),
                        "percent": getattr(event, "percent", None),
                        "message": getattr(event, "message", ""),
                        "detail": getattr(event, "detail", None),
                    }

                step = (
                    data.get("step")
                    or data.get("stage")
                    or data.get("step_name")
                    or data.get("message")
                    or "Pipeline progress"
                )
                local_percent = self._conversion_step_percent(step, data.get("status", "running"), data.get("percent"))
                global_percent = self._map_percent_to_range(local_percent, percent_start, percent_end)

                progress_payload = {
                    "status": "running",
                    "step_status": data.get("status", "running"),
                    "step": step,
                    "stage": data.get("stage", step),
                    "step_index": data.get("step_index"),
                    "total_steps": data.get("total_steps"),
                    "percent": global_percent,
                    "local_percent": local_percent,
                    "message": data.get("message") or str(step),
                    "detail": data.get("detail"),
                    "item_index": data.get("item_index"),
                    "total_items": data.get("total_items"),
                    "percent_range_start": percent_start,
                    "percent_range_end": percent_end,
                }

                self.emit(job_id, "progress", progress_payload)
            except Exception as exc:
                # Do not let a progress/UI reporting bug kill data conversion.
                self.emit(job_id, "progress", {
                    "status": "running",
                    "step": "Progress update skipped",
                    "percent": percent_start,
                    "message": f"A non-fatal progress update error was ignored: {exc!r}",
                })
        return callback


    def _speedmouse_dataset_api_url(self, job_id: str) -> str:
        return f"/api/jobs/{job_id}/neuromouse/data.json"

    def _speedmouse_job_url(self, job_id: str) -> str:
        return f"/neuromouse-job/{job_id}/?backend=1&force_backend=1&t={int(time.time())}"

    def find_latest_neuromouse_dataset(self) -> dict[str, Any] | None:
        """Find the newest backend-generated NeuroMouse data.json.

        Works even after a server restart by scanning the workspace outputs.
        """
        candidates: list[Path] = []
        outputs_root = self.workspace / "outputs"
        if outputs_root.exists():
            candidates.extend(outputs_root.rglob("neuromouse/**/data.json"))
            # Legacy packages wrote the same generated dataset under speedmouse/.
            candidates.extend(outputs_root.rglob("speedmouse/**/data.json"))
        # Also include active/custom-output jobs tracked in memory.
        with self.lock:
            records = list(self.jobs.values())
        for record in records:
            if record.output_dir:
                root = Path(record.output_dir)
                if root.exists():
                    candidates.extend(root.rglob("neuromouse/**/data.json"))
                    candidates.extend(root.rglob("speedmouse/**/data.json"))
        unique: dict[str, Path] = {str(p.resolve()): p for p in candidates if p.exists()}
        if not unique:
            return None
        newest = max(unique.values(), key=lambda p: p.stat().st_mtime)
        job_id = None
        try:
            parts = newest.resolve().parts
            if "outputs" in parts:
                idx = parts.index("outputs")
                if idx + 1 < len(parts):
                    job_id = parts[idx + 1]
        except Exception:
            job_id = None
        if not job_id:
            # Fall back to an in-memory job whose output_dir contains this file.
            with self.lock:
                for record in self.jobs.values():
                    try:
                        if record.output_dir and newest.resolve().is_relative_to(Path(record.output_dir).resolve()):
                            job_id = record.job_id
                            break
                    except Exception:
                        pass
        job_dataset_url = f"/api/jobs/{job_id}/neuromouse/data.json" if job_id else f"/api/file?path={quote(str(newest), safe='')}"
        # For the user-facing "latest" button and plain /neuromouse/ startup,
        # use a stable endpoint that cannot point at a stale browser-stored job id.
        # The job-specific URL is still returned for diagnostics.
        dataset_url = "/api/neuromouse/latest/data.json"
        neuromouse_url = f"/neuromouse-latest/?backend=1&force_backend=1&t={int(time.time())}"
        return {
            "job_id": job_id,
            "data_json": str(newest),
            "dataset_url": dataset_url,
            "job_dataset_url": job_dataset_url,
            "neuromouse_url": neuromouse_url,
            "mtime": newest.stat().st_mtime,
        }

    def _build_speedmouse_dataset_for_converted(
        self,
        job_id: str,
        converted_dir: Path,
        speedmouse_dir: Path,
        dataset_id: str,
        options: dict[str, Any],
        *,
        input_path: Path | None = None,
        percent: int = 96,
    ) -> dict[str, Any]:
        """Build a NeuroMouse data.json from a converted recording.

        This is intentionally used after normal conversion too. Users often
        click Convert first and then open NeuroMouse; without this step plain
        NeuroMouse can only show its bundled demo data. The generated data.json
        is the bridge from our canonical signal.npy output into NeuroMouse.
        """
        self.emit(job_id, "state", {
            "status": "running",
            "step": f"Build NeuroMouse dataset {dataset_id}",
            "percent": percent,
            "message": "Building full NeuroMouse data.json and advanced plot objects from converted signal.npy.",
        })
        sm_paths = write_speedmouse_dataset(
            converted_dir,
            speedmouse_dir,
            dataset_id=dataset_id,
            sampling_rate=options.get("sampling_rate") if options.get("sampling_rate") not in (None, "") else None,
            max_analysis_samples=int(options.get("neuromouse_max_analysis_samples") or options.get("speedmouse_max_analysis_samples") or 240000),
            max_windows=int(options.get("neuromouse_max_windows") or options.get("speedmouse_max_windows") or 600),
        )
        dataset_info = {
            "input_path": str(input_path) if input_path else None,
            "converted_dir": str(converted_dir),
            **sm_paths,
            "neuromouse_url": self._speedmouse_job_url(job_id),
            "neuromouse_dataset_url": self._speedmouse_dataset_api_url(job_id),
            "source": "generated_from_uploaded_or_converted_data",
        }
        self.emit(job_id, "raw_log", {
            "status": "running",
            "step": "NeuroMouse backend data.json generated",
            "percent": percent,
            "message": "Generated NeuroMouse data.json with PSD, geometry, chronomap, Kuramoto, PLV/network, Higuchi FD, and TDA views from converted uploaded data.",
            "dataset": dataset_info,
            "neuromouse_url": dataset_info["neuromouse_url"],
            "neuromouse_dataset_url": dataset_info["neuromouse_dataset_url"],
        })
        return dataset_info

    def start_convert_job(
        self,
        input_paths: list[str | Path],
        *,
        output_dir: str | Path | None = None,
        options: dict[str, Any] | None = None,
    ) -> JobRecord:
        record = self.create_job("convert", output_dir=output_dir)
        thread = threading.Thread(
            target=self._run_convert_job,
            args=(record.job_id, [str(Path(p)) for p in input_paths], options or {}),
            daemon=True,
        )
        thread.start()
        return record

    def _convert_one(
        self,
        pipeline: NeuroImportPipeline,
        job_id: str,
        input_path: Path,
        out_dir: Path,
        options: dict[str, Any],
        *,
        percent_start: int = 3,
        percent_end: int = 85,
    ) -> dict[str, Any]:
        self.emit(job_id, "progress", {"status": "running", "step": f"Converting {input_path.name}", "percent": percent_start, "message": f"Converting {input_path.name}"})
        preprocess_cfg = None
        if options.get("preprocess"):
            preprocess_cfg = {
                "enabled": True,
                "demean": bool(options.get("demean", False)),
                "detrend": bool(options.get("detrend", False)),
                "normalization": options.get("normalization") or None,
            }
        window_cfg = None
        if options.get("make_windows"):
            window_cfg = {
                "enabled": True,
                "window_seconds": float(options.get("window_seconds") or 2.0),
                "step_seconds": float(options.get("step_seconds") or 1.0),
            }
        export_cfg = {
            "format": options.get("export_format", "npy"),
            "save_signal_csv": bool(options.get("save_signal_csv", False)),
            "csv_max_mb": float(options.get("csv_max_mb", 250.0)),
        }
        unit_cfg = {}
        for key in ("original_units", "target_units", "scale_factor", "offset"):
            if options.get(key) not in (None, ""):
                unit_cfg[key] = options[key]
        adapter_kwargs: dict[str, Any] = {}
        if options.get("sampling_rate") not in (None, ""):
            adapter_kwargs["sampling_rate"] = float(options["sampling_rate"])
        if options.get("signal_path"):
            adapter_kwargs["signal_path"] = options["signal_path"]
        if options.get("orientation") and options["orientation"] != "auto":
            adapter_kwargs["orientation"] = options["orientation"]

        result = pipeline.convert(
            input_path,
            output_dir=out_dir,
            include_aux=bool(options.get("include_aux", False)),
            preprocess_config=preprocess_cfg,
            window_config=window_cfg,
            export_config=export_cfg,
            unit_config=unit_cfg or None,
            mapping_path=options.get("mapping_path") or None,
            progress_callback=self._progress_callback(job_id, percent_start=percent_start, percent_end=percent_end),
            **adapter_kwargs,
        )
        return {
            "input_path": str(input_path),
            "output_dir": str(out_dir),
            "adapter": result.get("adapter"),
            "canonical_paths": result.get("canonical_paths", {}),
            "qc_paths": result.get("qc_paths", {}),
            "window_paths": result.get("window_paths", {}),
        }

    def _run_convert_job(self, job_id: str, input_paths: list[str], options: dict[str, Any]) -> None:
        out_root = Path(self.jobs[job_id].output_dir or self.workspace / "outputs" / job_id)
        out_root.mkdir(parents=True, exist_ok=True)
        pipeline = NeuroImportPipeline()
        results = []
        try:
            discovery = discover_primary_signal_files(input_paths)
            primary_files: list[Path] = discovery["primary_files"]
            self.emit(job_id, "state", {"status": "running", "step": "Starting conversion", "percent": 1, "message": "Starting conversion.", "app_version": APP_VERSION, "input_paths": input_paths, "options": options})
            self.emit(job_id, "raw_log", {
                "status": "running",
                "step": "Input discovery / sidecar filtering",
                "percent": 2,
                "message": f"Found {len(primary_files)} primary conversion file(s); skipped {len(discovery['skipped'])} sidecar/non-primary file(s).",
                "app_version": APP_VERSION,
                "requested_paths": discovery["requested_paths"],
                "primary_files": [str(p) for p in primary_files],
                "skipped": discovery["skipped"],
                "promoted": discovery["promoted"],
                "archives": discovery.get("archives", []),
            })
            if not primary_files:
                raise ValueError("No primary neural signal files found. Upload/select a primary file such as .set, .edf, .vhdr, .zip archives containing .mat/.edf/.set files, or primary files such as .mat, .h5, .nwb, .npy, or .csv. Sidecars such as .fdt are copied but not converted directly.")
            speedmouse_root = out_root / "neuromouse"
            speedmouse_root.mkdir(parents=True, exist_ok=True)
            speedmouse_datasets: list[dict[str, Any]] = []
            for file_index, file_path in enumerate(primary_files):
                safe_name = file_path.stem.replace(" ", "_")[:80]
                out_dir = out_root / f"{file_index+1:03d}_{safe_name}"
                total_files = max(1, len(primary_files))
                conv_start = 3 + int(round((file_index / total_files) * 82))
                conv_end = 3 + int(round(((file_index + 1) / total_files) * 82))
                conv = self._convert_one(
                    pipeline,
                    job_id,
                    file_path,
                    out_dir,
                    options,
                    percent_start=conv_start,
                    percent_end=conv_end,
                )
                results.append(conv)
                try:
                    sm_dir = speedmouse_root / f"{file_index+1:03d}_{safe_name}"
                    sm_info = self._build_speedmouse_dataset_for_converted(
                        job_id,
                        out_dir,
                        sm_dir,
                        safe_name,
                        options,
                        input_path=file_path,
                        percent=min(98, 86 + file_index),
                    )
                    speedmouse_datasets.append(sm_info)
                except Exception as sm_exc:
                    self.emit(job_id, "raw_log", {
                        "status": "warning",
                        "step": "NeuroMouse data.json generation skipped",
                        "percent": 96,
                        "message": f"Conversion succeeded, but NeuroMouse data.json generation failed for {file_path.name}: {sm_exc!r}",
                        "error": repr(sm_exc),
                    })
            manifest_path = out_root / "webapp_conversion_manifest.json"
            result_payload = {
                "output_dir": str(out_root),
                "manifest": str(manifest_path),
                "converted": results,
                "neuromouse_datasets": speedmouse_datasets,
                "primary_neuromouse_url": speedmouse_datasets[0]["neuromouse_url"] if speedmouse_datasets else None,
                "primary_neuromouse_dataset_url": speedmouse_datasets[0]["neuromouse_dataset_url"] if speedmouse_datasets else None,
                "input_discovery": {**discovery, "primary_files": [str(p) for p in primary_files]},
                "app_version": APP_VERSION,
            }
            manifest_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            self.emit(job_id, "state", {
                "status": "complete",
                "step": "Conversion complete",
                "percent": 100,
                "message": "Conversion complete. NeuroMouse data.json was generated from the converted data when possible.",
                "output_dir": str(out_root),
                "result": result_payload,
            })
        except Exception as exc:
            self.emit(job_id, "state", {"status": "failed", "step": "Conversion failed", "percent": 100, "message": "Conversion failed.", "error": repr(exc), "output_dir": str(out_root), "app_version": APP_VERSION})

    def start_compare_job(
        self,
        group_a_dirs: list[str | Path],
        group_b_dirs: list[str | Path],
        *,
        output_dir: str | Path | None = None,
        options: dict[str, Any] | None = None,
    ) -> JobRecord:
        record = self.create_job("compare", output_dir=output_dir)
        thread = threading.Thread(
            target=self._run_compare_job,
            args=(record.job_id, [str(Path(p)) for p in group_a_dirs], [str(Path(p)) for p in group_b_dirs], options or {}),
            daemon=True,
        )
        thread.start()
        return record

    def _run_compare_job(self, job_id: str, group_a_dirs: list[str], group_b_dirs: list[str], options: dict[str, Any]) -> None:
        out_root = Path(self.jobs[job_id].output_dir or self.workspace / "outputs" / job_id)
        out_root.mkdir(parents=True, exist_ok=True)
        try:
            self.emit(job_id, "state", {"status": "running", "step": "Extracting features", "percent": 10, "message": "Extracting feature summaries from Group A and Group B."})
            result = run_comparative_analysis(group_a_dirs, group_b_dirs, output_dir=out_root, comparison_name=options.get("comparison_name", "comparison"), sampling_rate=options.get("sampling_rate"))
            self.emit(job_id, "state", {"status": "complete", "step": "Comparative analysis complete", "percent": 100, "message": "Comparative analysis complete.", "output_dir": str(out_root), "result": result})
        except Exception as exc:
            self.emit(job_id, "state", {"status": "failed", "step": "Comparative analysis failed", "percent": 100, "message": "Comparative analysis failed.", "error": repr(exc), "output_dir": str(out_root)})

    def start_live_job(
        self,
        source: str | Path,
        *,
        channels_csv: str | Path | None = None,
        metadata_json: str | Path | None = None,
        fs: float | None = None,
        channel_profile: str = "auto",
        output_dir: str | Path | None = None,
    ) -> JobRecord:
        record = self.create_job("live", output_dir=output_dir)
        thread = threading.Thread(
            target=self._run_live_job,
            args=(record.job_id, str(Path(source)), str(channels_csv) if channels_csv else None, str(metadata_json) if metadata_json else None, fs, channel_profile),
            daemon=True,
        )
        thread.start()
        return record

    def _run_live_job(self, job_id: str, source: str, channels_csv: str | None, metadata_json: str | None, fs: float | None, channel_profile: str) -> None:
        try:
            self.emit(job_id, "state", {"status": "running", "step": "Starting live replay backend", "percent": 10, "message": "Launching variable-electrode live replay backend."})
            cmd = [sys.executable, "-m", "neuro_importer_live.run_variable_backend", "--source", source, "--channel-profile", channel_profile]
            if channels_csv:
                cmd += ["--channels-csv", channels_csv]
            if metadata_json:
                cmd += ["--metadata-json", metadata_json]
            if fs:
                cmd += ["--fs", str(fs)]
            self.emit(job_id, "raw_log", {"status": "running", "step": "Live replay command", "percent": 20, "message": "Starting live replay subprocess.", "command": cmd})
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with self.lock:
                self.live_processes[job_id] = [proc]
            self.emit(job_id, "state", {"status": "running", "step": "Live replay running", "percent": 100, "message": "Live replay backend is running. Open the raw and spectral browser views.", "result": {"raw_url": "http://127.0.0.1:8765", "spectral_url": "http://127.0.0.1:8766"}})
        except Exception as exc:
            self.emit(job_id, "state", {"status": "failed", "step": "Live replay failed", "percent": 100, "message": "Live replay failed.", "error": repr(exc)})


    def start_speedmouse_analyze_job(
        self,
        input_paths: list[str | Path],
        *,
        output_dir: str | Path | None = None,
        options: dict[str, Any] | None = None,
    ) -> JobRecord:
        record = self.create_job("neuromouse_analyze", output_dir=output_dir)
        thread = threading.Thread(
            target=self._run_speedmouse_analyze_job,
            args=(record.job_id, [str(Path(p)) for p in input_paths], options or {}),
            daemon=True,
        )
        thread.start()
        return record

    def _run_speedmouse_analyze_job(self, job_id: str, input_paths: list[str], options: dict[str, Any]) -> None:
        out_root = Path(self.jobs[job_id].output_dir or self.workspace / "outputs" / job_id)
        converted_root = out_root / "converted"
        speedmouse_root = out_root / "neuromouse"
        converted_root.mkdir(parents=True, exist_ok=True)
        speedmouse_root.mkdir(parents=True, exist_ok=True)
        pipeline = NeuroImportPipeline()
        converted: list[dict[str, Any]] = []
        speedmouse_datasets: list[dict[str, Any]] = []
        try:
            self.emit(job_id, "state", {"status": "running", "step": "Analyze for NeuroMouse", "percent": 1, "message": "Starting NeuroMouse analysis pipeline.", "app_version": APP_VERSION})
            discovery = discover_primary_signal_files(input_paths)
            files: list[Path] = discovery["primary_files"]
            self.emit(job_id, "raw_log", {
                "status": "running",
                "step": "NeuroMouse input discovery / sidecar filtering",
                "percent": 2,
                "message": f"Discovered {len(files)} primary file(s) for NeuroMouse analysis; skipped {len(discovery['skipped'])} sidecar/non-primary file(s).",
                "app_version": APP_VERSION,
                "requested_paths": discovery["requested_paths"],
                "primary_files": [str(p) for p in files],
                "skipped": discovery["skipped"],
                "promoted": discovery["promoted"],
                "archives": discovery.get("archives", []),
            })
            if not files:
                raise ValueError("No supported primary input files found for NeuroMouse analysis. Upload/select .set, .edf, .vhdr, .zip archives containing .mat/.edf/.set files, or primary files such as .mat, .h5, .nwb, .npy, or .csv. Sidecars such as .fdt are copied but not analyzed directly.")
            total = len(files)
            for i, file_path in enumerate(files):
                base_percent = int((i / total) * 70)
                safe_name = file_path.stem.replace(" ", "_")[:80]
                out_dir = converted_root / f"{i+1:03d}_{safe_name}"
                self.emit(job_id, "state", {"status": "running", "step": f"Convert {file_path.name}", "percent": max(3, base_percent), "message": f"Converting {file_path.name}."})
                conv = self._convert_one(pipeline, job_id, file_path, out_dir, options)
                converted.append(conv)
                sm_dir = speedmouse_root / f"{i+1:03d}_{safe_name}"
                self.emit(job_id, "state", {"status": "running", "step": f"Build NeuroMouse dataset {file_path.name}", "percent": min(95, base_percent + 70), "message": "Computing Welch/centroid/geometry arrays for NeuroMouse."})
                dataset_info = self._build_speedmouse_dataset_for_converted(
                    job_id,
                    out_dir,
                    sm_dir,
                    safe_name,
                    options,
                    input_path=file_path,
                    percent=min(99, base_percent + 90),
                )
                speedmouse_datasets.append(dataset_info)
            manifest_path = out_root / "neuromouse_job_manifest.json"
            result = {
                "output_dir": str(out_root),
                "converted": converted,
                "neuromouse_datasets": speedmouse_datasets,
                "primary_neuromouse_url": speedmouse_datasets[0]["neuromouse_url"] if speedmouse_datasets else None,
                "primary_neuromouse_dataset_url": speedmouse_datasets[0].get("neuromouse_dataset_url") if speedmouse_datasets else None,
            }
            manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["manifest"] = str(manifest_path)
            self.emit(job_id, "state", {"status": "complete", "step": "NeuroMouse analysis complete", "percent": 100, "message": "NeuroMouse analysis complete.", "output_dir": str(out_root), "result": result})
        except Exception as exc:
            self.emit(job_id, "state", {"status": "failed", "step": "NeuroMouse analysis failed", "percent": 100, "message": "NeuroMouse analysis failed.", "error": repr(exc), "output_dir": str(out_root)})

    def start_speedmouse_from_converted_job(
        self,
        recording_dirs: list[str | Path],
        *,
        output_dir: str | Path | None = None,
        options: dict[str, Any] | None = None,
    ) -> JobRecord:
        record = self.create_job("neuromouse_from_converted", output_dir=output_dir)
        thread = threading.Thread(
            target=self._run_speedmouse_from_converted_job,
            args=(record.job_id, [str(Path(p)) for p in recording_dirs], options or {}),
            daemon=True,
        )
        thread.start()
        return record

    def _run_speedmouse_from_converted_job(self, job_id: str, recording_dirs: list[str], options: dict[str, Any]) -> None:
        out_root = Path(self.jobs[job_id].output_dir or self.workspace / "outputs" / job_id)
        speedmouse_root = out_root / "neuromouse"
        speedmouse_root.mkdir(parents=True, exist_ok=True)
        datasets = []
        try:
            for i, rec in enumerate(recording_dirs):
                p = Path(rec)
                safe_name = p.name.replace(" ", "_")[:80]
                self.emit(job_id, "state", {"status": "running", "step": f"Build NeuroMouse dataset {safe_name}", "percent": int((i / max(1, len(recording_dirs))) * 90), "message": f"Building NeuroMouse data.json for {safe_name}."})
                sm_dir = speedmouse_root / f"{i+1:03d}_{safe_name}"
                info = self._build_speedmouse_dataset_for_converted(
                    job_id,
                    p,
                    sm_dir,
                    safe_name,
                    options,
                    percent=min(99, int((i / max(1, len(recording_dirs))) * 90) + 10),
                )
                info["recording_dir"] = str(p)
                datasets.append(info)
            result = {
                "output_dir": str(out_root),
                "neuromouse_datasets": datasets,
                "primary_neuromouse_url": datasets[0]["neuromouse_url"] if datasets else None,
                "primary_neuromouse_dataset_url": datasets[0].get("neuromouse_dataset_url") if datasets else None,
            }
            (out_root / "neuromouse_from_converted_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            self.emit(job_id, "state", {"status": "complete", "step": "NeuroMouse dataset complete", "percent": 100, "message": "NeuroMouse dataset complete.", "output_dir": str(out_root), "result": result})
        except Exception as exc:
            self.emit(job_id, "state", {"status": "failed", "step": "NeuroMouse dataset failed", "percent": 100, "message": "NeuroMouse dataset failed.", "error": repr(exc), "output_dir": str(out_root)})

    def start_speedmouse_compare_job(
        self,
        group_a_dirs: list[str | Path],
        group_b_dirs: list[str | Path],
        *,
        output_dir: str | Path | None = None,
        options: dict[str, Any] | None = None,
    ) -> JobRecord:
        record = self.create_job("neuromouse_compare", output_dir=output_dir)
        thread = threading.Thread(
            target=self._run_speedmouse_compare_job,
            args=(record.job_id, [str(Path(p)) for p in group_a_dirs], [str(Path(p)) for p in group_b_dirs], options or {}),
            daemon=True,
        )
        thread.start()
        return record

    def _run_speedmouse_compare_job(self, job_id: str, group_a_dirs: list[str], group_b_dirs: list[str], options: dict[str, Any]) -> None:
        out_root = Path(self.jobs[job_id].output_dir or self.workspace / "outputs" / job_id)
        out_root.mkdir(parents=True, exist_ok=True)
        try:
            self.emit(job_id, "state", {"status": "running", "step": "Build NeuroMouse comparison pack", "percent": 5, "message": "Building NeuroMouse data.json files and comparison manifest."})
            result = build_speedmouse_comparison_pack(
                group_a_dirs,
                group_b_dirs,
                output_dir=out_root,
                comparison_name=options.get("comparison_name", "neuromouse_comparison"),
                sampling_rate=options.get("sampling_rate") if options.get("sampling_rate") not in (None, "") else None,
            )
            result["neuromouse_comparison_url"] = f"/neuromouse/?comparison={quote('/api/file?path=' + result['comparison_manifest'], safe='/?:=&%')}"
            self.emit(job_id, "state", {"status": "complete", "step": "NeuroMouse comparison complete", "percent": 100, "message": "NeuroMouse comparison complete.", "output_dir": str(out_root), "result": result})
        except Exception as exc:
            self.emit(job_id, "state", {"status": "failed", "step": "NeuroMouse comparison failed", "percent": 100, "message": "NeuroMouse comparison failed.", "error": repr(exc), "output_dir": str(out_root)})

    def stop_live_job(self, job_id: str) -> None:
        with self.lock:
            procs = self.live_processes.pop(job_id, [])
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        self.emit(job_id, "state", {"status": "stopped", "step": "Live replay stopped", "percent": 100, "message": "Live replay stopped."})
