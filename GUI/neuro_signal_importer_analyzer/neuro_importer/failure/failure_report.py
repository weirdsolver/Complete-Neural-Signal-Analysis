from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from neuro_importer.detect.continuous_signal_detector import find_continuous_signal_candidates, find_vector_by_hints, LABEL_HINTS, TIME_HINTS
from neuro_importer.detect.hdf5_signature_detector import collect_attrs
from neuro_importer.mapping import MappingSpec, write_mapping


def build_failure_report(path: str | Path, raw: Any, error: str | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        'source_path': str(path),
        'error': error,
        'message': 'The file was readable, but automatic conversion was not confident enough.',
        'possible_next_steps': [
            'Run inspect to review adapter scores.',
            'Generate a mapping YAML and specify the signal path/sampling rate.',
            'Add a custom adapter if this file family will recur often.',
        ],
    }
    if isinstance(raw, dict):
        candidates = []
        for sig_path, value, shape, score in find_continuous_signal_candidates(raw)[:20]:
            candidates.append({'path': sig_path, 'shape': list(shape), 'score': float(score)})
        report['continuous_signal_candidates'] = candidates
        label_path, _ = find_vector_by_hints(raw, LABEL_HINTS)
        time_path, _ = find_vector_by_hints(raw, TIME_HINTS)
        attrs = collect_attrs(raw)
        report['candidate_label_path'] = label_path
        report['candidate_time_path'] = time_path
        report['available_attrs'] = {str(k): str(v) for k, v in list(attrs.items())[:100]}
    return report


def suggest_mapping_from_report(report: dict[str, Any], *, sampling_rate: float | None = None) -> MappingSpec:
    candidates = report.get('continuous_signal_candidates') or []
    signal_path = candidates[0]['path'] if candidates else None
    return MappingSpec(
        signal_path=signal_path,
        sampling_rate=sampling_rate,
        time_path=report.get('candidate_time_path'),
        channel_names_path=report.get('candidate_label_path'),
        orientation='auto',
        original_units=None,
        target_units='microvolts',
        scale_factor=None,
        offset=None,
        metadata={'generated_from_failure_report': True, 'source_path': report.get('source_path')},
    )


def write_failure_artifacts(path: str | Path, raw: Any, output_dir: str | Path, *, error: str | None = None, sampling_rate: float | None = None) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_failure_report(path, raw, error)
    report_path = out / 'failure_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    spec = suggest_mapping_from_report(report, sampling_rate=sampling_rate)
    mapping_path = out / 'mapping_template.yaml'
    write_mapping(mapping_path, spec)
    return {'failure_report': str(report_path), 'mapping_template': str(mapping_path)}
