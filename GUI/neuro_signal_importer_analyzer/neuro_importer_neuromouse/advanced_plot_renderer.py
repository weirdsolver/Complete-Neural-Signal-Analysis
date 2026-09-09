from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any


PALETTE = {
    "bg": "#02070a",
    "panel": "#061114",
    "ink": "#dfffff",
    "muted": "#8fb9bd",
    "cyan": "#00e5e5",
    "green": "#00ff99",
    "line": "rgba(127,255,255,0.24)",
    "soft": "rgba(0,229,229,0.18)",
}


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _num_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [_safe_num(v) for v in values]


def _matrix(values: Any) -> list[list[float]]:
    if not isinstance(values, list):
        return []
    out: list[list[float]] = []
    for row in values:
        if isinstance(row, list):
            out.append([_safe_num(v) for v in row])
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    f = _safe_num(value, float("nan"))
    if not math.isfinite(f):
        return "—"
    if abs(f) >= 100:
        return f"{f:.2f}"
    return f"{f:.{digits}f}".rstrip("0").rstrip(".")


def _svg_frame(width: int, height: int, title: str, body: str, subtitle: str = "") -> str:
    subtitle_svg = ""
    if subtitle:
        subtitle_svg = f'<text x="24" y="48" fill="{PALETTE["muted"]}" font-size="12" font-family="ui-monospace, Menlo, monospace">{_esc(subtitle[:130])}</text>'
    return f'''
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" rx="16" fill="{PALETTE['bg']}" />
  <rect x="10" y="10" width="{width-20}" height="{height-20}" rx="14" fill="none" stroke="rgba(127,255,255,0.20)" />
  <text x="24" y="28" fill="{PALETTE['ink']}" font-size="16" font-family="ui-monospace, Menlo, monospace" font-weight="700">{_esc(title)}</text>
  {subtitle_svg}
  {body}
</svg>'''


def _empty_svg(title: str, message: str) -> str:
    body = f'''
  <text x="28" y="90" fill="{PALETTE['muted']}" font-size="14" font-family="ui-monospace, Menlo, monospace">{_esc(message)}</text>
  <text x="28" y="118" fill="{PALETTE['green']}" font-size="13" font-family="ui-monospace, Menlo, monospace">The backend page itself rendered, but this data object was absent.</text>
'''
    return _svg_frame(760, 360, title, body)


def polar_chronomap_svg(data: dict[str, Any]) -> str:
    pc = data.get("polar_chronomap") or {}
    posterior = _num_list(pc.get("posterior_alpha"))
    frontal = _num_list(pc.get("frontal_alpha"))
    balance = _num_list(pc.get("balance"))
    if not posterior:
        return _empty_svg("Polar Alpha Chronomap", "Missing polar_chronomap.posterior_alpha.")
    width, height = 760, 420
    cx, cy = width / 2, height / 2 + 18
    radius = 148
    max_val = max(max(posterior), max(frontal or [0]), 1e-9)
    n = len(posterior)
    spokes: list[str] = []
    for i, val in enumerate(posterior):
        angle = (i / max(1, n)) * math.tau - math.pi / 2
        r = max(5.0, (val / max_val) * radius)
        b = balance[i] if i < len(balance) else 0.0
        color = PALETTE["green"] if b >= 0 else PALETTE["cyan"]
        opacity = min(0.95, max(0.25, 0.35 + abs(b) * 3.0))
        x = cx + math.cos(angle) * r
        y = cy + math.sin(angle) * r
        spokes.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-opacity="{opacity:.3f}" stroke-width="2" />')
    rings = "".join(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius * rr:.1f}" fill="none" stroke="rgba(127,255,255,0.16)" />' for rr in (0.25, 0.5, 0.75, 1.0))
    mean_post = sum(posterior) / len(posterior)
    mean_front = sum(frontal) / len(frontal) if frontal else 0.0
    body = f'''
  {rings}
  {''.join(spokes)}
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{PALETTE['ink']}" />
  <text x="24" y="380" fill="{PALETTE['ink']}" font-size="13" font-family="ui-monospace, Menlo, monospace">frames={n} · posterior mean={_fmt(mean_post)} · frontal mean={_fmt(mean_front)} · posterior channels={len(pc.get('posterior_channels') or [])}</text>
  <text x="24" y="400" fill="{PALETTE['muted']}" font-size="12" font-family="ui-monospace, Menlo, monospace">green = posterior-dominant balance · cyan = frontal-dominant balance</text>
'''
    return _svg_frame(width, height, "Polar Alpha Chronomap", body, "Original NeuroMouse polar chronomap semantics, server-rendered from data.json.")


def kuramoto_svg(data: dict[str, Any], frame_index: int | None = None) -> str:
    k = data.get("kuramoto") or {}
    phases = _matrix(k.get("channel_phases"))
    r_values = _num_list(k.get("order_parameter_r"))
    psi_values = _num_list(k.get("mean_phase_psi"))
    if not phases or not phases[0]:
        return _empty_svg("Kuramoto Animation", "Missing kuramoto.channel_phases.")
    n_frames = min([len(row) for row in phases if row] or [0])
    if n_frames <= 0:
        return _empty_svg("Kuramoto Animation", "Kuramoto phase arrays are empty.")
    idx = frame_index if frame_index is not None else n_frames // 3
    idx = max(0, min(n_frames - 1, int(idx)))
    width, height = 760, 420
    cx, cy = width / 2, height / 2 + 18
    radius = 138
    points: list[str] = []
    for i, row in enumerate(phases[:96]):
        phase = row[idx]
        x = cx + math.cos(phase) * radius
        y = cy + math.sin(phase) * radius
        color = PALETTE["green"] if i % 2 == 0 else PALETTE["cyan"]
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" fill-opacity="0.86" />')
    rr = r_values[idx] if idx < len(r_values) else 0.0
    psi = psi_values[idx] if idx < len(psi_values) else 0.0
    vx = cx + math.cos(psi) * radius * rr
    vy = cy + math.sin(psi) * radius * rr
    body = f'''
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="none" stroke="rgba(127,255,255,0.28)" stroke-width="2" />
  <line x1="{cx:.1f}" y1="{cy:.1f}" x2="{vx:.1f}" y2="{vy:.1f}" stroke="{PALETTE['green']}" stroke-width="5" stroke-linecap="round" />
  {''.join(points)}
  <text x="24" y="380" fill="{PALETTE['ink']}" font-size="13" font-family="ui-monospace, Menlo, monospace">frame={idx+1}/{n_frames} · channels={len(phases)} · order parameter r={_fmt(rr, 3)} · mean phase ψ={_fmt(psi, 3)}</text>
  <text x="24" y="400" fill="{PALETTE['muted']}" font-size="12" font-family="ui-monospace, Menlo, monospace">This is a guaranteed static frame from the original Kuramoto animation data object.</text>
'''
    return _svg_frame(width, height, "Kuramoto Animation", body, "Alpha/broadband channel phase oscillators and mean phase vector.")


def channel_network_svg(data: dict[str, Any]) -> str:
    cn = data.get("channel_network") or {}
    matrix = _matrix(cn.get("composite_correlation") or (cn.get("per_metric") or {}).get("signal_correlation"))
    channels = cn.get("channels") or (data.get("meta") or {}).get("channels") or []
    if not matrix:
        return _empty_svg("Channel Network", "Missing channel_network.composite_correlation.")
    n = min(len(matrix), len(channels) if isinstance(channels, list) else len(matrix), 64)
    if n <= 0:
        return _empty_svg("Channel Network", "Channel network matrix is empty.")
    threshold = _safe_num(cn.get("threshold_strong"), 0.70)
    threshold = min(0.98, max(0.20, threshold))
    width, height = 760, 470
    cx, cy = width / 2, height / 2 + 32
    radius = 158
    coords: list[tuple[float, float, float]] = []
    for i in range(n):
        a = (i / n) * math.tau - math.pi / 2
        coords.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius, a))
    edges: list[str] = []
    strong = 0
    for i in range(n):
        for j in range(i + 1, n):
            v = abs(matrix[i][j]) if j < len(matrix[i]) else 0.0
            if v < threshold:
                continue
            strong += 1
            x1, y1, _ = coords[i]
            x2, y2, _ = coords[j]
            opacity = min(0.9, max(0.25, v))
            width_px = 1.0 + v * 2.8
            edges.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{PALETTE["green"]}" stroke-opacity="{opacity:.3f}" stroke-width="{width_px:.2f}" />')
    nodes: list[str] = []
    for i, (x, y, a) in enumerate(coords):
        degree = sum(1 for j in range(n) if i != j and j < len(matrix[i]) and abs(matrix[i][j]) >= threshold)
        r = 5.0 + min(7.0, degree * 0.7)
        nodes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{PALETTE["cyan"]}" stroke="{PALETTE["ink"]}" stroke-width="0.8" />')
        if n <= 36:
            label = _esc((channels[i] if isinstance(channels, list) and i < len(channels) else str(i))[:10])
            tx = cx + math.cos(a) * (radius + 28)
            ty = cy + math.sin(a) * (radius + 28)
            anchor = "end" if tx < cx - 4 else "start" if tx > cx + 4 else "middle"
            nodes.append(f'<text x="{tx:.1f}" y="{ty:.1f}" fill="{PALETTE["muted"]}" font-size="10" text-anchor="{anchor}" font-family="ui-monospace, Menlo, monospace">{label}</text>')
    body = f'''
  {''.join(edges)}
  {''.join(nodes)}
  <text x="24" y="428" fill="{PALETTE['ink']}" font-size="13" font-family="ui-monospace, Menlo, monospace">channels={n} · strong threshold={_fmt(threshold, 2)} · strong edges={strong}</text>
  <text x="24" y="448" fill="{PALETTE['muted']}" font-size="12" font-family="ui-monospace, Menlo, monospace">Edges use the original NeuroMouse channel_network composite matrix.</text>
'''
    return _svg_frame(width, height, "Channel Network", body, "Correlation / PLV-style graph over channels.")


def tda_svg(data: dict[str, Any]) -> str:
    tda = data.get("tda") or {}
    h0 = _matrix(tda.get("h0"))
    h1 = _matrix(tda.get("h1"))
    pairs = [(b, d, "H0") for b, d, *_ in h0 if math.isfinite(b) and math.isfinite(d)] + [(b, d, "H1") for b, d, *_ in h1 if math.isfinite(b) and math.isfinite(d)]
    if not pairs:
        return _empty_svg("TDA View", "Missing computed tda.h0 / tda.h1 persistence pairs.")
    width, height = 760, 470
    left, top, sw, sh = 62, 78, 300, 285
    min_b = min([b for b, _, _ in pairs] + [0.0])
    max_d = max([d for _, d, _ in pairs] + [1.0])
    span = max(1e-9, max_d - min_b)

    def sx(x: float) -> float:
        return left + ((x - min_b) / span) * sw

    def sy(y: float) -> float:
        return top + sh - ((y - min_b) / span) * sh

    points: list[str] = []
    for b, d, typ in pairs[:300]:
        color = PALETTE["green"] if typ == "H1" else PALETTE["cyan"]
        points.append(f'<circle cx="{sx(b):.1f}" cy="{sy(d):.1f}" r="4" fill="{color}" fill-opacity="0.85" />')
    bar_left = 430
    bar_top = 84
    bar_w = 282
    rows = pairs[:80]
    row_gap = min(7.0, 270 / max(1, len(rows)))
    bars: list[str] = []
    for i, (b, d, typ) in enumerate(rows):
        y = bar_top + i * row_gap
        color = PALETTE["green"] if typ == "H1" else PALETTE["cyan"]
        x1 = bar_left + ((b - min_b) / span) * bar_w
        x2 = bar_left + ((d - min_b) / span) * bar_w
        bars.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{max(x1+2,x2):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2" stroke-opacity="0.82" />')
    max_life = max((d - b) for b, d, _ in pairs)
    body = f'''
  <rect x="{left}" y="{top}" width="{sw}" height="{sh}" fill="none" stroke="rgba(127,255,255,0.24)" />
  <line x1="{sx(min_b):.1f}" y1="{sy(min_b):.1f}" x2="{sx(max_d):.1f}" y2="{sy(max_d):.1f}" stroke="rgba(255,255,255,0.25)" />
  {''.join(points)}
  <text x="{left}" y="{top-10}" fill="{PALETTE['muted']}" font-size="11" font-family="ui-monospace, Menlo, monospace">persistence scatter</text>
  <text x="{bar_left}" y="{top-10}" fill="{PALETTE['muted']}" font-size="11" font-family="ui-monospace, Menlo, monospace">barcode</text>
  {''.join(bars)}
  <text x="24" y="428" fill="{PALETTE['ink']}" font-size="13" font-family="ui-monospace, Menlo, monospace">status={_esc(tda.get('status') or 'unknown')} · H0={len(h0)} · H1={len(h1)} · max lifetime={_fmt(max_life)}</text>
  <text x="24" y="448" fill="{PALETTE['muted']}" font-size="12" font-family="ui-monospace, Menlo, monospace">Server-rendered from original NeuroMouse TDA persistence fields.</text>
'''
    return _svg_frame(width, height, "TDA View", body, "Persistence scatter and barcode view.")


def synthetic_advanced_dataset() -> dict[str, Any]:
    channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    frames = 96
    posterior = [0.18 + 0.06 * math.sin(i / frames * math.tau * 2.0) for i in range(frames)]
    frontal = [0.12 + 0.04 * math.cos(i / frames * math.tau * 1.5) for i in range(frames)]
    balance = [p - f for p, f in zip(posterior, frontal)]
    phases = [[(i * 0.42 + j * 0.08 + 0.45 * math.sin(j / 9.0)) % math.tau for j in range(frames)] for i in range(len(channels))]
    order = [abs(sum(math.cos(row[j]) for row in phases) / len(phases)) for j in range(frames)]
    psi = [math.atan2(sum(math.sin(row[j]) for row in phases), sum(math.cos(row[j]) for row in phases)) for j in range(frames)]
    matrix = []
    for i in range(len(channels)):
        row = []
        for j in range(len(channels)):
            row.append(1.0 if i == j else max(0.0, 0.82 - abs(i-j)*0.08 + 0.07*math.sin(i+j)))
        matrix.append(row)
    h0 = [[0.0, 0.15 + i*0.04] for i in range(len(channels)-1)]
    h1 = [[0.32, 0.78], [0.42, 0.95]]
    return {
        "meta": {"dataset_id": "synthetic_builtin_advanced_plot_demo", "channels": channels, "n_channels": len(channels)},
        "geometry": {"time": [i * 0.5 for i in range(frames)]},
        "polar_chronomap": {"time": [i * 0.5 for i in range(frames)], "posterior_alpha": posterior, "frontal_alpha": frontal, "balance": balance, "posterior_channels": ["P3", "P4", "O1", "O2"], "frontal_channels": ["Fp1", "Fp2", "F3", "F4"]},
        "kuramoto": {"time": [i * 0.5 for i in range(frames)], "channel_phases": phases, "channels": channels, "order_parameter_r": order, "mean_phase_psi": psi},
        "channel_network": {"channels": channels, "composite_correlation": matrix, "threshold_strong": 0.66, "threshold_moderate": 0.45},
        "tda": {"status": "computed", "h0": h0, "h1": h1, "point_cloud": [[math.sin(i), math.cos(i)] for i in range(len(channels))]},
    }


def load_best_dataset(manager: Any, neuromouse_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    try:
        info = manager.find_latest_neuromouse_dataset()
        if info:
            p = Path(str(info.get("data_json") or "")).expanduser()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    avail = availability(data)
                    if all(avail.values()):
                        return data, {"source": "latest_backend_neuromouse_analysis", "path": str(p), **info}
                    missing = [key for key, ok in avail.items() if not ok]
                    errors.append(f"latest backend data lacks required advanced objects: {', '.join(missing)} at {p}")
            errors.append(f"latest path missing: {p}")
        else:
            errors.append("no latest backend NeuroMouse dataset")
    except Exception as exc:
        errors.append(f"latest backend load failed: {exc!r}")

    demo = neuromouse_dir / "data" / "data.json"
    try:
        if demo.exists():
            data = json.loads(demo.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data, {"source": "bundled_original_neuromouse_demo", "path": str(demo), "errors": errors}
        errors.append(f"demo path missing: {demo}")
    except Exception as exc:
        errors.append(f"bundled demo load failed: {exc!r}")

    return synthetic_advanced_dataset(), {"source": "synthetic_builtin_advanced_plot_demo", "path": "generated in advanced_plot_renderer.py", "errors": errors}


def availability(data: dict[str, Any]) -> dict[str, bool]:
    return {
        "polar_chronomap": bool((data.get("polar_chronomap") or {}).get("posterior_alpha")),
        "kuramoto": bool((data.get("kuramoto") or {}).get("channel_phases")),
        "channel_network": bool((data.get("channel_network") or {}).get("composite_correlation")),
        "tda": bool((data.get("tda") or {}).get("status") == "computed" and ((data.get("tda") or {}).get("h0") or (data.get("tda") or {}).get("h1"))),
    }


def render_advanced_analysis_page(data: dict[str, Any], info: dict[str, Any], *, app_version: str = "") -> str:
    meta = data.get("meta") or {}
    channels = meta.get("channels") if isinstance(meta.get("channels"), list) else []
    n_channels = len(channels) or meta.get("n_channels") or len((data.get("channel_network") or {}).get("channels") or [])
    frames = len((data.get("geometry") or {}).get("time") or (data.get("kuramoto") or {}).get("time") or (data.get("polar_chronomap") or {}).get("time") or [])
    avail = availability(data)
    status_items = "".join(f'<span class="pill {"ok" if ok else "warn"}">{_esc(key)}: {"ready" if ok else "fallback/missing"}</span>' for key, ok in avail.items())
    source = _esc(info.get("source", "unknown"))
    path = _esc(info.get("path", ""))
    title_version = f" v{_esc(app_version)}" if app_version else ""
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Advanced Analysis{title_version} — guaranteed NeuroMouse plots</title>
  <style>
    :root {{ color-scheme: dark; --bg:#02070a; --panel:#061114; --line:rgba(127,255,255,0.22); --ink:#dfffff; --muted:#8fb9bd; --green:#00ff99; --cyan:#00e5e5; --warn:#ffd166; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top left, rgba(0,255,153,.10), transparent 32rem), var(--bg); color:var(--ink); font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding:22px 28px; border-bottom:1px solid var(--line); display:flex; gap:16px; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; }}
    h1 {{ margin:0 0 6px; font-size:26px; }}
    h1 span {{ color:var(--green); font-size:16px; }}
    h2 {{ margin:0 0 10px; color:var(--green); }}
    p {{ color:var(--muted); line-height:1.48; }}
    main {{ padding:20px; max-width:1580px; margin:0 auto; }}
    .card {{ background:linear-gradient(180deg, rgba(6,17,20,.98), rgba(3,8,10,.98)); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 12px 36px rgba(0,0,0,.22); }}
    .hero {{ margin-bottom:18px; border-color:rgba(0,255,153,.38); }}
    .button-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }}
    a.button {{ color:#00110b; background:var(--green); text-decoration:none; padding:10px 14px; border-radius:10px; font-weight:700; }}
    a.button.secondary {{ color:var(--ink); background:transparent; border:1px solid var(--line); }}
    .pills {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
    .pill {{ border:1px solid var(--line); border-radius:999px; padding:7px 10px; color:var(--muted); font-family:ui-monospace, Menlo, monospace; font-size:12px; }}
    .pill.ok {{ border-color:rgba(0,255,153,.45); color:var(--green); }}
    .pill.warn {{ border-color:rgba(255,209,102,.5); color:var(--warn); }}
    .grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:16px; }}
    .plot-card svg {{ width:100%; height:auto; display:block; border-radius:14px; }}
    .plot-card {{ min-height:420px; }}
    .meta {{ font-family:ui-monospace, Menlo, monospace; color:var(--muted); font-size:12px; overflow-wrap:anywhere; }}
    .note {{ margin-top:16px; padding:12px; border:1px solid rgba(0,229,229,.25); border-radius:12px; color:var(--muted); background:rgba(0,229,229,.04); }}
    @media (max-width: 1000px) {{ .grid {{ grid-template-columns:1fr; }} header {{ padding:18px; }} main {{ padding:14px; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Advanced Analysis <span>{title_version or ''} guaranteed plot page</span></h1>
      <p>This page is server-rendered. It does not depend on tabs, localStorage, hidden NeuroMouse panels, or frontend JavaScript to show the four requested plots.</p>
    </div>
    <div class="meta">source: {source}<br/>path: {path}</div>
  </header>
  <main>
    <section class="card hero">
      <h2>Original NeuroMouse advanced plots, rendered by a reliable backend method</h2>
      <p>The backend loads the newest generated NeuroMouse <code>data.json</code>. If none exists, it uses the bundled original NeuroMouse demo. If even that fails, it uses a built-in synthetic NeuroMouse-compatible dataset so these plot sections still appear.</p>
      <div class="pills">
        <span class="pill ok">Polar Alpha Chronomap</span>
        <span class="pill ok">Kuramoto Animation</span>
        <span class="pill ok">Channel Network</span>
        <span class="pill ok">TDA View</span>
      </div>
      <div class="pills">{status_items}</div>
      <p class="meta">dataset={_esc(meta.get('dataset_id') or 'unknown')} · channels={_esc(n_channels)} · frames={_esc(frames)}</p>
      <div class="button-row">
        <a class="button" href="/advanced-analysis?refresh=1">Reload plots</a>
        <a class="button secondary" href="/">Back to launcher</a>
        <a class="button secondary" href="/neuromouse/">Open original NeuroMouse workbench</a>
      </div>
    </section>

    <section class="grid" aria-label="Guaranteed NeuroMouse advanced plots">
      <article class="card plot-card"><h2>Polar Alpha Chronomap</h2>{polar_chronomap_svg(data)}</article>
      <article class="card plot-card"><h2>Kuramoto Animation</h2>{kuramoto_svg(data)}</article>
      <article class="card plot-card"><h2>Channel Network</h2>{channel_network_svg(data)}</article>
      <article class="card plot-card"><h2>TDA View</h2>{tda_svg(data)}</article>
    </section>
    <section class="note">
      These plots use the same NeuroMouse data objects and visual meanings as the original workbench. The rendering is intentionally backend-side SVG so the sections are visible even when the original conditional browser panels would hide themselves.
    </section>
  </main>
</body>
</html>'''


def plot_summary_payload(data: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    meta = data.get("meta") or {}
    return {
        "ok": True,
        "source": info.get("source"),
        "path": info.get("path"),
        "dataset_id": meta.get("dataset_id"),
        "n_channels": meta.get("n_channels") or len(meta.get("channels") or []),
        "availability": availability(data),
        "plot_page_url": "/advanced-analysis",
    }
