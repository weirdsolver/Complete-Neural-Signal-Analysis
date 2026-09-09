from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class BaseReader(Protocol):
    def read(self, path: str | Path) -> Any:
        ...
