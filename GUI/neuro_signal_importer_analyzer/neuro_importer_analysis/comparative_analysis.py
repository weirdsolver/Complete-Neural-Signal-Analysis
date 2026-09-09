from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .feature_extractor import extract_features_from_recording_dir
from .report_builder import write_comparison_report


def _summary_row(summary: dict[str, Any], group: str) -> dict[str, Any]:
    row = dict(summary)
    row["group"] = group
    return row


def run_comparative_analysis(
    group_a_dirs: list[str | Path],
    group_b_dirs: list[str | Path],
    *,
    output_dir: str | Path,
    comparison_name: str = "comparison",
    sampling_rate: float | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for i, path in enumerate(group_a_dirs):
        dsid = f"A_{i+1}_{Path(path).name}"
        features, summary = extract_features_from_recording_dir(path, dataset_id=dsid, sampling_rate=sampling_rate)
        features["group"] = "A"
        all_features.append(features)
        summaries.append(_summary_row(summary, "A"))

    for i, path in enumerate(group_b_dirs):
        dsid = f"B_{i+1}_{Path(path).name}"
        features, summary = extract_features_from_recording_dir(path, dataset_id=dsid, sampling_rate=sampling_rate)
        features["group"] = "B"
        all_features.append(features)
        summaries.append(_summary_row(summary, "B"))

    features_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)

    metric_cols = [
        "global_rms_mean",
        "global_variance_mean",
        "global_centroid_mean_hz",
        "global_alpha_power_mean",
        "global_beta_power_mean",
    ]
    comparison_rows = []
    for metric in metric_cols:
        if metric not in summary_df:
            continue
        a = pd.to_numeric(summary_df.loc[summary_df["group"] == "A", metric], errors="coerce")
        b = pd.to_numeric(summary_df.loc[summary_df["group"] == "B", metric], errors="coerce")
        comparison_rows.append({
            "metric": metric,
            "group_a_mean": float(a.mean()) if len(a.dropna()) else None,
            "group_b_mean": float(b.mean()) if len(b.dropna()) else None,
            "difference_b_minus_a": float(b.mean() - a.mean()) if len(a.dropna()) and len(b.dropna()) else None,
            "n_group_a": int(len(a.dropna())),
            "n_group_b": int(len(b.dropna())),
        })
    comparison_df = pd.DataFrame(comparison_rows)

    features_path = output_dir / "channel_features.csv"
    summary_path = output_dir / "dataset_summaries.csv"
    comparison_path = output_dir / "comparison_metrics.csv"
    metadata_path = output_dir / "comparison_metadata.json"
    report_path = output_dir / "comparative_report.html"

    features_df.to_csv(features_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    metadata = {
        "comparison_name": comparison_name,
        "group_a_dirs": [str(Path(p)) for p in group_a_dirs],
        "group_b_dirs": [str(Path(p)) for p in group_b_dirs],
        "continuous_signal_only": True,
        "comparison_mode": "feature_level",
        "note": "Feature-level comparison is safe across variable channel counts. Same-channel comparisons should only be used when channel manifests match.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    write_comparison_report(summary_df, comparison_df, report_path, metadata=metadata)

    return {
        "output_dir": str(output_dir),
        "features_csv": str(features_path),
        "summaries_csv": str(summary_path),
        "comparison_csv": str(comparison_path),
        "metadata_json": str(metadata_path),
        "report_html": str(report_path),
        "metadata": metadata,
    }
