from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from neuro_importer.core.recording import Recording


@dataclass
class AdapterScore:
    name: str
    confidence: float
    reasons: list[str]


class BaseAdapter(Protocol):
    name: str

    def score(self, raw: Any) -> AdapterScore:
        ...

    def convert(self, raw: Any, *, source_path: str | Path | None = None, **kwargs: Any) -> Recording:
        ...
