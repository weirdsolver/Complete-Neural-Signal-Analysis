from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from neuro_importer.core.recording import Recording
from neuro_importer.export.canonical_exporter import _json_safe
from neuro_importer.qc.channel_qc import channel_qc
from neuro_importer.qc.signal_qc import signal_qc


def write_qc_report(recording: Recording, output_dir: str | Path, config: dict[str, Any] | None = None) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config or {}
    summary = signal_qc(recording, cfg)
    channels = channel_qc(recording, cfg)

    json_path = out / "qc_report.json"
    csv_path = out / "qc_channels.csv"
    html_path = out / "qc_report.html"
    json_path.write_text(json.dumps(_json_safe({"summary": summary}), indent=2), encoding="utf-8")
    channels.to_csv(csv_path, index=False)

    warnings = "".join(f"<li>{html.escape(str(w))}</li>" for w in summary.get("warnings", [])) or "<li>No QC warnings.</li>"
    top_table = channels.sort_values("std", ascending=False).head(20).to_html(index=False, escape=True)
    flat_table = channels[channels["flat"]].head(50).to_html(index=False, escape=True)
    if flat_table.strip() == "":
        flat_table = "<p>No flat channels detected.</p>"

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>Neuro signal QC report</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; }}
th {{ background: #f4f4f4; }}
code, pre {{ background: #f8f8f8; padding: 0.2rem 0.35rem; }}
</style>
</head>
<body>
<h1>Neuro signal QC report</h1>
<h2>Summary</h2>
<pre>{html.escape(json.dumps(_json_safe(summary), indent=2))}</pre>
<h2>Warnings</h2>
<ul>{warnings}</ul>
<h2>Highest-variance channels</h2>
{top_table}
<h2>Flat channels</h2>
{flat_table}
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")

    for warning in summary.get("warnings", []):
        recording.quality.add_warning(str(warning))
    recording.quality.add_info("QC report generated.")
    return {"qc_report_json": str(json_path), "qc_channels": str(csv_path), "qc_report_html": str(html_path)}
