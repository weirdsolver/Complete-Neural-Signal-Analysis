from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class DropZone(QLabel):
    def __init__(self, on_paths: Callable[[list[str]], None]) -> None:
        super().__init__("Drop neural signal files or folders here\n\nOr use the buttons below")
        self.on_paths = on_paths
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)
        self.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 10px; padding: 24px; font-size: 16px; }"
        )

    def dragEnterEvent(self, event):  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # type: ignore[override]
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]
        if paths:
            self.on_paths(paths)
            event.acceptProposedAction()
