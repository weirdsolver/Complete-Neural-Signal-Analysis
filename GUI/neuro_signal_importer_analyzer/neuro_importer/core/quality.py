from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityReport:
    """Human-readable import/conversion report.

    This object is deliberately simple so it can be exported to JSON and read by
    a scientist or engineer after conversion.
    """

    adapter: str | None = None
    confidence: float = 0.0
    infos: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def add_info(self, message: str) -> None:
        self.infos.append(str(message))

    def add_warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def add_error(self, message: str) -> None:
        self.errors.append(str(message))

    def add_assumption(self, message: str) -> None:
        self.assumptions.append(str(message))

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "confidence": self.confidence,
            "ok": self.ok,
            "infos": self.infos,
            "warnings": self.warnings,
            "errors": self.errors,
            "assumptions": self.assumptions,
        }
