from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MappingSpec:
    signal_path: str | None = None
    signal_columns: list[str] | None = None
    sampling_rate: float | None = None
    sampling_rate_path: str | None = None
    time_path: str | None = None
    time_column: str | None = None
    channel_names_path: str | None = None
    orientation: str = 'auto'  # auto | samples_by_channels | channels_by_samples
    original_units: str | None = None
    target_units: str | None = None
    scale_factor: float | None = None
    offset: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'MappingSpec':
        return cls(
            signal_path=d.get('signal_path'),
            signal_columns=d.get('signal_columns'),
            sampling_rate=d.get('sampling_rate'),
            sampling_rate_path=d.get('sampling_rate_path'),
            time_path=d.get('time_path'),
            time_column=d.get('time_column'),
            channel_names_path=d.get('channel_names_path'),
            orientation=d.get('orientation', 'auto'),
            original_units=d.get('original_units'),
            target_units=d.get('target_units'),
            scale_factor=d.get('scale_factor'),
            offset=d.get('offset'),
            metadata=d.get('metadata') or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'signal_path': self.signal_path,
            'signal_columns': self.signal_columns,
            'sampling_rate': self.sampling_rate,
            'sampling_rate_path': self.sampling_rate_path,
            'time_path': self.time_path,
            'time_column': self.time_column,
            'channel_names_path': self.channel_names_path,
            'orientation': self.orientation,
            'original_units': self.original_units,
            'target_units': self.target_units,
            'scale_factor': self.scale_factor,
            'offset': self.offset,
            'metadata': self.metadata,
        }


def load_mapping(path: str | Path) -> MappingSpec:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Mapping file must contain a YAML mapping/object: {path}')
    return MappingSpec.from_dict(data)


def write_mapping(path: str | Path, spec: MappingSpec) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding='utf-8')
    return str(p)
