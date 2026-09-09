from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

ProgressStatus = Literal["pending", "running", "complete", "failed", "heartbeat"]


@dataclass(slots=True)
class ProgressEvent:
    """Small backend-to-frontend progress heartbeat event.

    Percent is stage-based, not time-based.  The GUI should trust these events
    over thread-cleanup signals so it can show real sequential completion.
    """

    stage: str
    status: ProgressStatus
    step_index: int
    total_steps: int
    percent: int
    message: str = ""
    item_index: int | None = None
    total_items: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[ProgressEvent], None]


class ProgressReporter:
    """Convenience wrapper around an optional progress callback."""

    def __init__(self, callback: ProgressCallback | None, *, total_steps: int = 9) -> None:
        self.callback = callback
        self.total_steps = total_steps

    def emit(
        self,
        stage: str,
        *,
        status: ProgressStatus,
        step_index: int,
        percent: int,
        message: str | None = None,
        detail: str | None = None,
        item_index: int | None = None,
        total_items: int | None = None,
    ) -> None:
        if self.callback is None:
            return
        safe_percent = max(0, min(100, int(percent)))
        event = ProgressEvent(
            stage=stage,
            status=status,
            step_index=step_index,
            total_steps=self.total_steps,
            percent=safe_percent,
            message=message or stage,
            detail=detail,
            item_index=item_index,
            total_items=total_items,
        )
        self.callback(event)

    def running(self, stage: str, step_index: int, percent: int, **kwargs: Any) -> None:
        self.emit(stage, status="running", step_index=step_index, percent=percent, **kwargs)

    def complete(self, stage: str, step_index: int, percent: int, **kwargs: Any) -> None:
        self.emit(stage, status="complete", step_index=step_index, percent=percent, **kwargs)

    def fail(self, stage: str, step_index: int, percent: int, **kwargs: Any) -> None:
        self.emit(stage, status="failed", step_index=step_index, percent=percent, **kwargs)
