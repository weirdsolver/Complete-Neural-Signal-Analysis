from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    return df.head(max_rows).to_html(index=False, escape=True, classes="data-table")


def write_comparison_report(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = html.escape((metadata or {}).get("comparison_name", "Neural Signal Comparative Analysis"))
    meta_json = html.escape(json.dumps(metadata or {}, indent=2))
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #061010; color: #dfffff; }}
    h1, h2 {{ color: #00ffff; }}
    .card {{ border: 1px solid #00aaaa; border-radius: 10px; padding: 16px; margin: 16px 0; background: #081818; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #155; padding: 6px 8px; text-align: left; }}
    th {{ background: #022; color: #8fffff; }}
    pre {{ white-space: pre-wrap; background: #020707; padding: 12px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="card"><h2>Comparison Metrics</h2>{_table(comparison_df)}</div>
  <div class="card"><h2>Dataset Summaries</h2>{_table(summary_df)}</div>
  <div class="card"><h2>Metadata</h2><pre>{meta_json}</pre></div>
</body>
</html>
"""
    output_path.write_text(body)
    return str(output_path)
