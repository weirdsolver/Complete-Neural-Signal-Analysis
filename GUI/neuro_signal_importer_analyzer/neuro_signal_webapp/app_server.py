from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import webbrowser
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .job_manager import APP_VERSION, JobManager, SUPPORTED_UPLOAD_SUFFIXES, discover_primary_signal_files, is_primary_signal_file

PACKAGE_DIR = Path(__file__).parent
STATIC_DIR = PACKAGE_DIR / "static"
NEUROMOUSE_DIR = PACKAGE_DIR / "neuromouse"
DEFAULT_WORKSPACE = Path(os.environ.get("NEURO_SIGNAL_APP_WORKSPACE", str(Path.home() / "neuro_signal_app_workspace")))
manager = JobManager(DEFAULT_WORKSPACE)

app = FastAPI(title="Neuro Signal App", version=APP_VERSION)


@app.middleware("http")
async def no_cache_for_local_frontend(request, call_next):
    """Prevent stale NeuroMouse/launcher JavaScript from surviving updates."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith(("/neuromouse", "/speedmouse", "/static")) or path.endswith((".js", ".css", ".html", ".json")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/neuromouse", StaticFiles(directory=NEUROMOUSE_DIR, html=True), name="neuromouse")


@app.get("/speedmouse")
@app.get("/speedmouse/")
def speedmouse_legacy_redirect() -> RedirectResponse:
    return RedirectResponse(url="/neuromouse/", status_code=307)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/advanced-analysis")
def advanced_analysis_page() -> RedirectResponse:
    return RedirectResponse(url="/#higuchiDirectCard", status_code=307)


@app.get("/higuchi")
@app.get("/higuchi-fractal-dimension")
def higuchi_page() -> RedirectResponse:
    return RedirectResponse(url="/#higuchiDirectCard", status_code=307)


@app.get("/embedded-fractal-dimension")
@app.get("/embedded-fd")
def embedded_fd_page() -> RedirectResponse:
    return RedirectResponse(url="/#embeddedFdDirectCard", status_code=307)


@app.get("/dimension-saturation-profiling")
@app.get("/saturation-profiling")
def saturation_profiling_page() -> RedirectResponse:
    return RedirectResponse(url="/#saturationDirectCard", status_code=307)


@app.get("/wavelet-hurst")
@app.get("/manual-wavelet-hurst")
def wavelet_hurst_page() -> RedirectResponse:
    return RedirectResponse(url="/#waveletDirectCard", status_code=307)


@app.get("/manual-expert-mfdfa")
@app.get("/mfdfa")
def mfdfa_page() -> RedirectResponse:
    return RedirectResponse(url="/#mfdfaDirectCard", status_code=307)


@app.get("/manual-mfdfa-spectrum")
@app.get("/mfdfa-spectrum")
def mfdfa_spectrum_page() -> RedirectResponse:
    return RedirectResponse(url="/#mfdfaSpectrumDirectCard", status_code=307)


@app.get("/manual-mfdfa-shuffle-surrogate")
@app.get("/mfdfa-shuffle")
def mfdfa_shuffle_page() -> RedirectResponse:
    return RedirectResponse(url="/#mfdfaShuffleDirectCard", status_code=307)


@app.get("/manual-mfdfa-iaaft-surrogate")
@app.get("/mfdfa-iaaft")
def mfdfa_iaaft_page() -> RedirectResponse:
    return RedirectResponse(url="/#mfdfaIaaftDirectCard", status_code=307)


@app.get("/wavelet-leader-multifractal")
@app.get("/wavelet-leader")
def wavelet_leader_page() -> RedirectResponse:
    return RedirectResponse(url="/#waveletLeaderDirectCard", status_code=307)


@app.get("/arnold-kuramoto")
@app.get("/arnold-tongues-kuramoto")
def arnold_kuramoto_page() -> RedirectResponse:
    return RedirectResponse(url="/#arnoldKuramotoDirectCard", status_code=307)


@app.get("/circle-map-arnold-tongues")
@app.get("/circle-map")
def circle_map_arnold_page() -> RedirectResponse:
    return RedirectResponse(url="/#circleMapArnoldDirectCard", status_code=307)


@app.get("/circle-map-converged-density")
@app.get("/circle-density")
def circle_map_density_page() -> RedirectResponse:
    return RedirectResponse(url="/#circleMapDensityDirectCard", status_code=307)


@app.get("/mfdfa-plot-viewer")
@app.get("/mfdfa-viewer")
def mfdfa_viewer_page() -> RedirectResponse:
    return RedirectResponse(url="/#mfdfaViewerDirectCard", status_code=307)


@app.get("/api/neuromouse/advanced-plots")
def advanced_plot_summary() -> JSONResponse:
    from neuro_importer_neuromouse.advanced_plot_renderer import load_best_dataset, plot_summary_payload

    data, info = load_best_dataset(manager, NEUROMOUSE_DIR)
    return JSONResponse(plot_summary_payload(data, info))



@app.get("/neuromouse-job/{job_id}/", response_class=HTMLResponse)
def neuromouse_job_page(job_id: str) -> HTMLResponse:
    """Serve original NeuroMouse forced to load this backend job's data.json.

    This route prevents accidental fallback to /neuromouse/data/data.json. If
    the job data is missing, NeuroMouse will show a load error instead of
    silently showing the bundled demo dataset.
    """
    index_path = NEUROMOUSE_DIR / "index.html"
    html = index_path.read_text(encoding="utf-8")
    dataset_url = f"/api/neuromouse/job-or-latest/{job_id}/data.json"
    bootstrap = f"""
    <base href="/neuromouse/">
    <script>
      // Hard bind this NeuroMouse page to the backend-generated dataset.
      // This guard exists because the original NeuroMouse workbench falls back
      // to its bundled demo data when it does not receive a dataset URL.
      // A job page must never silently show the bundled demo dataset.
      window.NEURO_SIGNAL_BACKEND_DATASET = {{
        backend: true,
        forceBackend: true,
        disableDemoFallback: true,
        jobId: {json.dumps(job_id)},
        datasetUrl: {json.dumps(dataset_url)},
        source: "neuro_signal_backend_job"
      }};
      try {{
        window.localStorage.setItem("NEURO_SIGNAL_LAST_BACKEND_DATASET_URL", {json.dumps(dataset_url)});
        window.localStorage.setItem("NEURO_SIGNAL_LAST_BACKEND_JOB_ID", {json.dumps(job_id)});
        window.localStorage.setItem("NEURO_SIGNAL_LAST_NEUROMOUSE_URL", {json.dumps(f"/neuromouse-job/{job_id}/")});
      }} catch (e) {{}}
      (function () {{
        const backendDatasetUrl = {json.dumps(dataset_url)};
        const originalFetch = window.fetch ? window.fetch.bind(window) : null;
        if (!originalFetch) return;
        window.fetch = function (input, init) {{
          try {{
            const rawUrl = typeof input === "string" ? input : (input && input.url ? input.url : "");
            const resolved = rawUrl ? new URL(rawUrl, window.location.href) : null;
            const path = resolved ? resolved.pathname : "";
            const isDemoData = path.endsWith("/neuromouse/data/data.json") || path.endsWith("/speedmouse/data/data.json") || path.endsWith("/data/data.json") || rawUrl === "data/data.json" || rawUrl === "./data/data.json";
            if (isDemoData) {{
              const forced = backendDatasetUrl + (backendDatasetUrl.includes("?") ? "&" : "?") + "forced_backend=1&t=" + Date.now();
              return originalFetch(forced, init);
            }}
          }} catch (e) {{}}
          return originalFetch(input, init);
        }};
      }}());
    </script>
    """
    if "<head>" in html:
        html = html.replace("<head>", "<head>" + bootstrap, 1)
    else:
        html = bootstrap + html
    return HTMLResponse(html)


@app.get("/speedmouse-job/{job_id}/", response_class=HTMLResponse)
def speedmouse_job_page_legacy(job_id: str) -> HTMLResponse:
    return neuromouse_job_page(job_id)





def _paths_from_neuromouse_result_payload(payload: dict[str, Any]) -> list[Path]:
    """Return candidate NeuroMouse data.json paths referenced by a job result/manifest.

    The server only keeps JobRecord objects in memory. After a restart, a
    /neuromouse-job/<job_id>/ page can still exist in browser history or local
    storage, but /api/jobs/<job_id>/neuromouse/data.json used to 404 because
    the in-memory record was gone. This helper lets the API recover by reading
    saved manifests and filesystem outputs.
    """
    paths: list[Path] = []
    for key in ("neuromouse_datasets", "speedmouse_datasets"):
        datasets = payload.get(key) or []
        if isinstance(datasets, list):
            for item in datasets:
                if isinstance(item, dict):
                    for path_key in ("data_json", "neuromouse_data_json", "speedmouse_data_json"):
                        value = item.get(path_key)
                        if value:
                            paths.append(Path(str(value)).expanduser())
    for key in ("data_json", "neuromouse_data_json", "speedmouse_data_json"):
        value = payload.get(key)
        if value:
            paths.append(Path(str(value)).expanduser())
    result = payload.get("result")
    if isinstance(result, dict):
        paths.extend(_paths_from_neuromouse_result_payload(result))
    return paths


def _find_neuromouse_data_json_for_job(job_id: str) -> tuple[Path | None, dict[str, Any]]:
    """Find a generated NeuroMouse data.json for a job, even after restart."""
    debug: dict[str, Any] = {"job_id": job_id, "checked": [], "reason": None}
    candidates: list[Path] = []

    record = manager.get(job_id)
    roots: list[Path] = []
    if record:
        debug["record_in_memory"] = True
        if record.output_dir:
            roots.append(Path(record.output_dir).expanduser())
        if record.result:
            candidates.extend(_paths_from_neuromouse_result_payload(record.result))
    else:
        debug["record_in_memory"] = False

    # Persisted default workspace output root. This is the important restart
    # fallback: the browser may know the job id even when the Python process no
    # longer has the in-memory JobRecord.
    roots.append(manager.workspace / "outputs" / job_id)
    roots.append(DEFAULT_WORKSPACE / "outputs" / job_id)

    # Deduplicate roots while preserving order.
    unique_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.expanduser().resolve())
        except Exception:
            resolved = str(root)
        if resolved not in seen_roots:
            seen_roots.add(resolved)
            unique_roots.append(root)

    for root in unique_roots:
        debug["checked"].append(str(root))
        if not root.exists():
            continue

        # Prefer manifest-declared order, which preserves the primary dataset
        # the Results tab originally intended to open.
        for manifest_name in (
            "webapp_conversion_manifest.json",
            "neuromouse_job_manifest.json",
            "neuromouse_from_converted_manifest.json",
            "speedmouse_job_manifest.json",
        ):
            manifest = root / manifest_name
            if manifest.exists():
                try:
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    manifest_candidates = _paths_from_neuromouse_result_payload(payload)
                    for c in manifest_candidates:
                        if not c.is_absolute():
                            c = root / c
                        candidates.append(c)
                except Exception as exc:
                    debug.setdefault("manifest_errors", []).append({"manifest": str(manifest), "error": repr(exc)})

        # Filesystem fallback for older/newer layouts and legacy names.
        preferred_patterns = (
            "neuromouse/data.json",
            "speedmouse/data.json",
            "neuromouse/*/data.json",
            "speedmouse/*/data.json",
            "**/neuromouse/data.json",
            "**/speedmouse/data.json",
            "**/data.json",
        )
        for pattern in preferred_patterns:
            candidates.extend(root.glob(pattern))

    # Deduplicate candidate files. Keep manifest order first.
    seen: set[str] = set()
    existing: list[Path] = []
    for candidate in candidates:
        try:
            c = candidate.expanduser().resolve()
        except Exception:
            continue
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.name == "data.json":
            existing.append(c)

    if existing:
        # Manifest candidates appear first. If we only found files by glob, use
        # the newest so "latest generated dataset" errors recover sensibly.
        selected = existing[0]
        debug["selected"] = str(selected)
        debug["candidate_count"] = len(existing)
        return selected, debug

    debug["reason"] = "No generated NeuroMouse data.json found on disk for this job id."
    return None, debug


@app.get("/api/neuromouse/latest")
def latest_neuromouse_dataset() -> JSONResponse:
    """Return the newest backend-generated NeuroMouse dataset.

    This is used by the launcher and by plain /neuromouse/ so users do not
    accidentally see the bundled demo after they have just converted data.
    """
    info = manager.find_latest_neuromouse_dataset()
    if not info:
        return JSONResponse({"ok": False, "error": "No backend-generated NeuroMouse dataset found yet."}, status_code=404)
    return JSONResponse({"ok": True, **info})


@app.get("/api/neuromouse/latest/data.json")
def latest_neuromouse_data_json() -> FileResponse:
    """Serve the newest generated NeuroMouse data.json directly.

    This route is intentionally stable: unlike /api/jobs/<job_id>/..., it does
    not depend on a browser-stored job id. It prevents a stale localStorage URL
    from causing NeuroMouse startup 404s after conversion or server restart.
    """
    info = manager.find_latest_neuromouse_dataset()
    if not info:
        return JSONResponse({"error": "No backend-generated NeuroMouse dataset found yet."}, status_code=404)  # type: ignore[return-value]
    p = Path(str(info["data_json"])).expanduser()
    if not p.exists():
        return JSONResponse({"error": "Latest NeuroMouse data.json path no longer exists.", "path": str(p)}, status_code=404)  # type: ignore[return-value]
    return FileResponse(p, media_type="application/json")


@app.get("/api/neuromouse/job-or-latest/{job_id}/data.json")
def job_or_latest_neuromouse_data_json(job_id: str) -> FileResponse:
    """Serve a job's data.json, but fall back to the latest generated dataset.

    NeuroMouse should never become a blank/dead page merely because the browser
    held an old job URL. If the requested job output is missing, the user still
    gets the newest generated backend dataset instead of the bundled demo or a
    startup 404.
    """
    p, debug = _find_neuromouse_data_json_for_job(job_id)
    if p:
        return FileResponse(p, media_type="application/json")
    latest = manager.find_latest_neuromouse_dataset()
    if latest:
        latest_path = Path(str(latest["data_json"])).expanduser()
        if latest_path.exists():
            return FileResponse(latest_path, media_type="application/json")
    return JSONResponse({
        "error": "No generated NeuroMouse data.json found for requested job or latest fallback.",
        "job_id": job_id,
        "debug": debug,
        "hint": "Re-run Convert or Analyze in NeuroMouse, then open Open Latest NeuroMouse Dataset from Results.",
    }, status_code=404)  # type: ignore[return-value]


@app.get("/neuromouse-latest/", response_class=HTMLResponse)
def neuromouse_latest_page() -> HTMLResponse:
    """Serve NeuroMouse hard-bound to the newest generated backend dataset."""
    index_path = NEUROMOUSE_DIR / "index.html"
    html = index_path.read_text(encoding="utf-8")
    dataset_url = "/api/neuromouse/latest/data.json"
    bootstrap = f"""
    <base href="/neuromouse/">
    <script>
      window.NEURO_SIGNAL_BACKEND_DATASET = {{
        backend: true,
        forceBackend: true,
        disableDemoFallback: true,
        jobId: "latest",
        datasetUrl: {json.dumps(dataset_url)},
        source: "neuro_signal_backend_latest"
      }};
      try {{
        window.localStorage.setItem("NEURO_SIGNAL_LAST_BACKEND_DATASET_URL", {json.dumps(dataset_url)});
        window.localStorage.setItem("NEURO_SIGNAL_LAST_NEUROMOUSE_URL", "/neuromouse-latest/");
      }} catch (e) {{}}
      (function () {{
        const backendDatasetUrl = {json.dumps(dataset_url)};
        const originalFetch = window.fetch ? window.fetch.bind(window) : null;
        if (!originalFetch) return;
        window.fetch = function (input, init) {{
          try {{
            const rawUrl = typeof input === "string" ? input : (input && input.url ? input.url : "");
            const resolved = rawUrl ? new URL(rawUrl, window.location.href) : null;
            const path = resolved ? resolved.pathname : "";
            const isDemoData = path.endsWith("/neuromouse/data/data.json") || path.endsWith("/speedmouse/data/data.json") || path.endsWith("/data/data.json") || rawUrl === "data/data.json" || rawUrl === "./data/data.json";
            if (isDemoData) {{
              const forced = backendDatasetUrl + (backendDatasetUrl.includes("?") ? "&" : "?") + "forced_backend=1&t=" + Date.now();
              return originalFetch(forced, init);
            }}
          }} catch (e) {{}}
          return originalFetch(input, init);
        }};
      }}());
    </script>
    """
    if "<head>" in html:
        html = html.replace("<head>", "<head>" + bootstrap, 1)
    else:
        html = bootstrap + html
    return HTMLResponse(html)




def _recording_dir_has_signal(path: str | Path) -> bool:
    root = Path(path).expanduser()
    return any((root / rel).exists() for rel in ("signal.npy", "processed/signal.npy", "raw/signal.npy"))


def _latest_converted_recording_dir() -> str | None:
    """Return the newest converted recording folder with canonical signal.npy."""
    latest = manager.find_latest_neuromouse_dataset()
    if latest:
        data_path = Path(str(latest.get("data_json") or "")).expanduser()
        if data_path.exists():
            try:
                data = json.loads(data_path.read_text(encoding="utf-8"))
                source_dir = data.get("meta", {}).get("source_recording_dir")
                if source_dir and _recording_dir_has_signal(source_dir):
                    return str(Path(source_dir).expanduser())
            except Exception:
                pass
    candidates: list[Path] = []
    outputs_root = manager.workspace / "outputs"
    if outputs_root.exists():
        candidates.extend(p.parent for p in outputs_root.rglob("signal.npy") if "neuromouse" not in p.parts and "speedmouse" not in p.parts)
    with manager.lock:
        records = list(manager.jobs.values())
    for record in records:
        if record.output_dir:
            root = Path(record.output_dir).expanduser()
            if root.exists():
                candidates.extend(p.parent for p in root.rglob("signal.npy") if "neuromouse" not in p.parts and "speedmouse" not in p.parts)
    unique = {str(path.resolve()): path for path in candidates if path.exists()}
    if not unique:
        return None
    newest = max(unique.values(), key=lambda path: (path / "signal.npy").stat().st_mtime)
    return str(newest)


@app.get("/api/advanced-methods")
def advanced_methods() -> JSONResponse:
    from neuro_importer_analysis import list_advanced_methods

    return JSONResponse({
        "ok": True,
        "methods": list_advanced_methods(),
        "latest_recording_dir": _latest_converted_recording_dir(),
    })


@app.get("/api/advanced-methods/latest-recording")
def advanced_methods_latest_recording() -> JSONResponse:
    latest = _latest_converted_recording_dir()
    if not latest:
        return JSONResponse({"ok": False, "error": "No converted recording folder with signal.npy found yet."}, status_code=404)
    return JSONResponse({"ok": True, "recording_dir": latest})


@app.post("/api/advanced-methods/run")
def advanced_methods_run(payload: dict[str, Any]) -> JSONResponse:
    from neuro_importer_analysis import run_advanced_method

    method_id = str(payload.get("method_id") or payload.get("method") or "").strip()
    if not method_id:
        return JSONResponse({"ok": False, "error": "Provide method_id."}, status_code=400)
    recording_dir = str(payload.get("recording_dir") or "").strip()
    if not recording_dir or recording_dir.lower() == "latest":
        latest = _latest_converted_recording_dir()
        if not latest:
            return JSONResponse({"ok": False, "error": "Provide a converted recording folder, or run Convert/Analyze first so a latest recording exists."}, status_code=400)
        recording_dir = latest
    p = Path(recording_dir).expanduser()
    if not p.exists():
        return JSONResponse({"ok": False, "error": f"Recording folder does not exist: {p}"}, status_code=404)
    try:
        result = run_advanced_method(method_id, p, payload.get("params") or {})
        out_dir = p / "advanced_methods"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{method_id}_result.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["saved_result_json"] = str(out_path)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": repr(exc), "method_id": method_id, "recording_dir": str(p)}, status_code=500)

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": APP_VERSION, "workspace": str(manager.workspace), "neuromouse": "/neuromouse/", "app_file": str(Path(__file__).resolve())}


def _parse_json_form(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _safe_upload_target(upload_dir: Path, uploaded_name: str | None) -> Path:
    """Preserve browser folder-upload relative paths without allowing traversal."""
    raw = uploaded_name or "uploaded_file"
    # Browser directory uploads usually send POSIX-style relative paths.
    parts = [part for part in PurePosixPath(raw).parts if part not in ("", ".", "..")]
    if not parts:
        parts = ["uploaded_file"]
    target = upload_dir.joinpath(*parts)
    resolved = target.resolve()
    root = upload_dir.resolve()
    if root not in resolved.parents and resolved != root:
        target = upload_dir / Path(raw).name
    return target


def _save_uploads(job_id: str, files: list[UploadFile]) -> list[str]:
    upload_dir = manager.workspace / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for up in files:
        target = _safe_upload_target(upload_dir, up.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            shutil.copyfileobj(up.file, f)
        saved.append(str(target))
    return saved


def _primary_upload_paths(saved_paths: list[str]) -> list[str]:
    """Return job input paths after light upload filtering.

    Sidecars such as .fdt are normally filtered here so old UI/API expectations
    stay stable. Archive uploads are different: a .zip is not itself a primary
    signal file, so it must be sent to the job manager, which safely extracts it
    in the background job and logs the extracted primary files.
    """
    if any(Path(p).suffix.lower() == ".zip" for p in saved_paths):
        return list(saved_paths)
    discovery = discover_primary_signal_files(saved_paths)
    return [str(p) for p in discovery["primary_files"]]


@app.post("/api/jobs/convert-upload")
def convert_upload(
    files: list[UploadFile] = File(...),
    options_json: str = Form("{}"),
    output_dir: str | None = Form(None),
) -> dict[str, Any]:
    # Make job first so uploaded files are placed under the job id.
    record = manager.create_job("upload_prepare", output_dir=output_dir or None)
    saved = _save_uploads(record.job_id, files)
    primary_paths = _primary_upload_paths(saved)
    # Reuse the job id by replacing the temporary record with a real conversion state.
    with manager.lock:
        manager.jobs.pop(record.job_id, None)
        manager.event_queues.pop(record.job_id, None)
    job = manager.start_convert_job(primary_paths, output_dir=output_dir or None, options=_parse_json_form(options_json))
    # Keep uploaded files copied into the upload folder for traceability; only primary_paths are converted.
    return {"job_id": job.job_id, "status": job.status, "saved_files": saved, "primary_paths": primary_paths, "output_dir": job.output_dir}


@app.post("/api/jobs/convert-paths")
def convert_paths(payload: dict[str, Any]) -> dict[str, Any]:
    paths = payload.get("paths") or []
    if not paths:
        return JSONResponse({"error": "No paths provided."}, status_code=400)  # type: ignore[return-value]
    job = manager.start_convert_job(paths, output_dir=payload.get("output_dir") or None, options=payload.get("options") or {})
    return {"job_id": job.job_id, "status": job.status, "output_dir": job.output_dir}


@app.post("/api/jobs/inspect-upload")
def inspect_upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    from neuro_importer.pipeline import NeuroImportPipeline
    temp_id = "inspect_" + os.urandom(4).hex()
    saved = _save_uploads(temp_id, files)
    pipe = NeuroImportPipeline()
    results = []
    for path in saved:
        try:
            results.append(pipe.inspect(path))
        except Exception as exc:
            results.append({"path": path, "error": repr(exc)})
    return {"results": results}


@app.post("/api/jobs/compare")
def compare(payload: dict[str, Any]) -> dict[str, Any]:
    group_a = payload.get("group_a") or []
    group_b = payload.get("group_b") or []
    if not group_a or not group_b:
        return JSONResponse({"error": "Provide group_a and group_b converted recording directories."}, status_code=400)  # type: ignore[return-value]
    job = manager.start_compare_job(group_a, group_b, output_dir=payload.get("output_dir") or None, options=payload.get("options") or {})
    return {"job_id": job.job_id, "status": job.status, "output_dir": job.output_dir}


@app.post("/api/jobs/live")
def live(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    if not source:
        return JSONResponse({"error": "Provide source signal.npy path."}, status_code=400)  # type: ignore[return-value]
    job = manager.start_live_job(
        source,
        channels_csv=payload.get("channels_csv") or None,
        metadata_json=payload.get("metadata_json") or None,
        fs=float(payload["fs"]) if payload.get("fs") else None,
        channel_profile=payload.get("channel_profile") or "auto",
        output_dir=payload.get("output_dir") or None,
    )
    return {"job_id": job.job_id, "status": job.status, "output_dir": job.output_dir}


@app.post("/api/jobs/{job_id}/stop-live")
def stop_live(job_id: str) -> dict[str, Any]:
    manager.stop_live_job(job_id)
    return {"ok": True, "job_id": job_id}


@app.post("/api/jobs/analyze-neuromouse-upload")
@app.post("/api/jobs/analyze-speedmouse-upload")
def analyze_neuromouse_upload(
    files: list[UploadFile] = File(...),
    options_json: str = Form("{}"),
    output_dir: str | None = Form(None),
) -> dict[str, Any]:
    record = manager.create_job("speedmouse_upload_prepare", output_dir=output_dir or None)
    saved = _save_uploads(record.job_id, files)
    primary_paths = _primary_upload_paths(saved)
    with manager.lock:
        manager.jobs.pop(record.job_id, None)
        manager.event_queues.pop(record.job_id, None)
    job = manager.start_speedmouse_analyze_job(primary_paths, output_dir=output_dir or None, options=_parse_json_form(options_json))
    return {"job_id": job.job_id, "status": job.status, "saved_files": saved, "primary_paths": primary_paths, "output_dir": job.output_dir}


@app.post("/api/jobs/analyze-neuromouse-paths")
@app.post("/api/jobs/analyze-speedmouse-paths")
def analyze_neuromouse_paths(payload: dict[str, Any]) -> dict[str, Any]:
    paths = payload.get("paths") or []
    if not paths:
        return JSONResponse({"error": "No paths provided."}, status_code=400)  # type: ignore[return-value]
    job = manager.start_speedmouse_analyze_job(paths, output_dir=payload.get("output_dir") or None, options=payload.get("options") or {})
    return {"job_id": job.job_id, "status": job.status, "output_dir": job.output_dir}


@app.post("/api/jobs/neuromouse-from-converted")
@app.post("/api/jobs/speedmouse-from-converted")
def neuromouse_from_converted(payload: dict[str, Any]) -> dict[str, Any]:
    recording_dirs = payload.get("recording_dirs") or []
    if not recording_dirs:
        return JSONResponse({"error": "Provide recording_dirs converted recording folders."}, status_code=400)  # type: ignore[return-value]
    job = manager.start_speedmouse_from_converted_job(recording_dirs, output_dir=payload.get("output_dir") or None, options=payload.get("options") or {})
    return {"job_id": job.job_id, "status": job.status, "output_dir": job.output_dir}


@app.post("/api/jobs/compare-neuromouse")
@app.post("/api/jobs/compare-speedmouse")
def compare_neuromouse(payload: dict[str, Any]) -> dict[str, Any]:
    group_a = payload.get("group_a") or []
    group_b = payload.get("group_b") or []
    if not group_a or not group_b:
        return JSONResponse({"error": "Provide group_a and group_b converted recording directories."}, status_code=400)  # type: ignore[return-value]
    job = manager.start_speedmouse_compare_job(group_a, group_b, output_dir=payload.get("output_dir") or None, options=payload.get("options") or {})
    return {"job_id": job.job_id, "status": job.status, "output_dir": job.output_dir}


@app.websocket("/ws/neuromouse/live")
@app.websocket("/ws/speedmouse/live")
async def neuromouse_live_ws(websocket: WebSocket) -> None:
    from neuro_importer_neuromouse.live_bridge import stream_speedmouse_samples
    await websocket.accept()
    qp = websocket.query_params
    source = qp.get("source")
    if not source:
        await websocket.send_json({"type": "error", "error": "Missing source query parameter."})
        await websocket.close()
        return
    try:
        await stream_speedmouse_samples(
            websocket,
            source=source,
            channels_csv=qp.get("channels_csv"),
            metadata_json=qp.get("metadata_json"),
            fs=float(qp["fs"]) if qp.get("fs") else None,
            chunk_samples=int(qp.get("chunk_samples") or 256),
            speed=float(qp.get("speed") or 1.0),
            loop=(qp.get("loop") or "false").lower() in {"1", "true", "yes"},
        )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "error": repr(exc)})
        finally:
            await websocket.close()




@app.get("/api/jobs/{job_id}/neuromouse/data.json")
@app.get("/api/jobs/{job_id}/speedmouse/data.json")
def job_neuromouse_data_json(job_id: str) -> FileResponse:
    """Serve the primary NeuroMouse data.json for a completed job.

    v0.11.0 important fix: this endpoint now works after the local server has
    restarted. Earlier versions only looked in the in-memory JobRecord table,
    so a valid browser link such as /neuromouse-job/<job_id>/ could fail with
    HTTP 404 even though outputs/<job_id>/neuromouse/.../data.json still existed
    on disk.
    """
    p, debug = _find_neuromouse_data_json_for_job(job_id)
    if p:
        return FileResponse(p, media_type="application/json")

    # v0.11.0: old browser tabs/localStorage may still request
    # /api/jobs/<stale_id>/neuromouse/data.json. Do not let that create a
    # blank NeuroMouse startup page when a newer generated dataset exists.
    latest = manager.find_latest_neuromouse_dataset()
    if latest:
        latest_path = Path(str(latest["data_json"])).expanduser()
        if latest_path.exists():
            return FileResponse(latest_path, media_type="application/json")

    return JSONResponse({
        "error": "No generated NeuroMouse data.json found for this job or latest fallback.",
        "job_id": job_id,
        "debug": debug,
        "hint": "Re-run Convert or Analyze in NeuroMouse, then open Open Latest NeuroMouse Dataset from the Results tab.",
    }, status_code=404)  # type: ignore[return-value]


@app.get("/api/jobs/{job_id}/neuromouse/manifest.json")
@app.get("/api/jobs/{job_id}/speedmouse/manifest.json")
def job_neuromouse_manifest_json(job_id: str) -> JSONResponse:
    """Return a small manifest used by NeuroMouse to show provenance."""
    record = manager.get(job_id)
    if record:
        result = record.result or {}
        return JSONResponse({
            "job_id": job_id,
            "status": record.status,
            "output_dir": record.output_dir,
            "result": result,
        })

    # Restart fallback: return saved manifest metadata if available.
    root = manager.workspace / "outputs" / job_id
    for manifest_name in ("webapp_conversion_manifest.json", "neuromouse_job_manifest.json", "neuromouse_from_converted_manifest.json"):
        manifest = root / manifest_name
        if manifest.exists():
            try:
                return JSONResponse({
                    "job_id": job_id,
                    "status": "persisted",
                    "output_dir": str(root),
                    "result": json.loads(manifest.read_text(encoding="utf-8")),
                })
            except Exception as exc:
                return JSONResponse({"error": f"Failed to read saved manifest: {exc!r}"}, status_code=500)
    data_json, debug = _find_neuromouse_data_json_for_job(job_id)
    if data_json:
        return JSONResponse({
            "job_id": job_id,
            "status": "persisted",
            "output_dir": str((manager.workspace / "outputs" / job_id)),
            "result": {
                "neuromouse_datasets": [{"data_json": str(data_json)}],
                "primary_neuromouse_dataset_url": f"/api/jobs/{job_id}/neuromouse/data.json",
                "primary_neuromouse_url": f"/neuromouse-job/{job_id}/",
            },
            "debug": debug,
        })
    return JSONResponse({"error": "Job not found and no saved manifest found.", "debug": debug}, status_code=404)



@app.get("/api/jobs/{job_id}/raw-log.txt")
def job_raw_log_txt(job_id: str) -> FileResponse:
    """Download or view the human-readable raw backend job log."""
    record = manager.get(job_id)
    if not record or not record.raw_log_path:
        return JSONResponse({"error": "Job or raw log not found."}, status_code=404)  # type: ignore[return-value]
    p = Path(record.raw_log_path).expanduser().resolve()
    if not p.exists():
        return JSONResponse({"error": f"Raw log path does not exist: {p}"}, status_code=404)  # type: ignore[return-value]
    return FileResponse(p, media_type="text/plain", filename=f"{job_id}_job_log.txt")


@app.get("/api/jobs/{job_id}/raw-log.jsonl")
def job_raw_log_jsonl(job_id: str) -> FileResponse:
    """Download the machine-readable JSONL backend job log."""
    record = manager.get(job_id)
    if not record or not record.raw_jsonl_path:
        return JSONResponse({"error": "Job or raw JSONL log not found."}, status_code=404)  # type: ignore[return-value]
    p = Path(record.raw_jsonl_path).expanduser().resolve()
    if not p.exists():
        return JSONResponse({"error": f"Raw JSONL log path does not exist: {p}"}, status_code=404)  # type: ignore[return-value]
    return FileResponse(p, media_type="application/x-ndjson", filename=f"{job_id}_job_log.jsonl")


@app.get("/api/jobs/{job_id}/raw-log")
def job_raw_log_summary(job_id: str) -> JSONResponse:
    """Return raw log paths and browser URLs for the Results tab."""
    record = manager.get(job_id)
    if not record:
        return JSONResponse({"error": "Job not found."}, status_code=404)
    return JSONResponse({
        "job_id": job_id,
        "raw_log_path": record.raw_log_path,
        "raw_jsonl_path": record.raw_jsonl_path,
        "raw_log_url": f"/api/jobs/{job_id}/raw-log.txt",
        "raw_jsonl_url": f"/api/jobs/{job_id}/raw-log.jsonl",
    })

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    record = manager.get(job_id)
    if not record:
        return JSONResponse({"error": "Job not found."}, status_code=404)  # type: ignore[return-value]
    return record.__dict__


@app.get("/api/open-output")
def open_output(path: str) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return JSONResponse({"error": f"Path does not exist: {p}"}, status_code=404)  # type: ignore[return-value]
    if os.name == "posix":
        os.system(f'xdg-open "{p}" >/dev/null 2>&1 &')
    else:
        webbrowser.open(p.as_uri() if p.is_file() else str(p))
    return {"ok": True, "path": str(p)}


@app.get("/api/file")
def get_file(path: str) -> FileResponse:
    p = Path(path).expanduser().resolve()
    return FileResponse(p)


@app.websocket("/api/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    q = manager.subscribe(job_id)
    try:
        while True:
            try:
                event = await asyncio.to_thread(q.get, True, 0.5)
                await websocket.send_json(event)
            except queue.Empty:
                record = manager.get(job_id)
                if record and record.status in {"complete", "failed", "stopped"}:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "job_id": job_id,
                        "status": record.status,
                        "percent": record.progress_percent,
                        "step": record.current_step,
                        "output_dir": record.output_dir,
                        "result": record.result,
                    })
                    return
                await websocket.send_json({"type": "heartbeat", "job_id": job_id})
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(job_id, q)



@app.get("/live/raw")
def live_raw_view() -> FileResponse:
    from neuro_importer_live import __file__ as live_init
    return FileResponse(Path(live_init).parent / "raw_visualizer_variable.html")


@app.get("/live/spectral")
def live_spectral_view() -> FileResponse:
    from neuro_importer_live import __file__ as live_init
    return FileResponse(Path(live_init).parent / "spectral_visualizer_variable.html")


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Launch the local HTML Neuro Signal App.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args()

    global manager
    if args.workspace:
        manager = JobManager(args.workspace)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print(f"Neuro Signal App v{APP_VERSION} running at {url}")
    print(f"Workspace: {manager.workspace}")
    uvicorn.run("neuro_signal_webapp.app_server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
