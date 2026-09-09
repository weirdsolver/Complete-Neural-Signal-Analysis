from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from neuro_importer_frontend.drop_zone import DropZone
from neuro_importer_frontend.worker import PipelineWorker


class MainWindow(QMainWindow):
    """Qt desktop frontend for the continuous neural-signal importer.

    v0.5.5 uses a backend heartbeat protocol. The GUI shows only the current
    stage and most recent completed stage by default, while keeping a compact
    detail area below.  Progress is based on sequential pipeline completion,
    not time guesses or Qt thread cleanup.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neuro Signal Importer v0.5.5")
        self.resize(1100, 760)
        self.inputs: list[str] = []
        self.output_dir = str(Path.home() / "neuro_importer_outputs")
        self.last_result: dict[str, Any] | None = None
        self.thread: QThread | None = None
        self.worker: PipelineWorker | None = None
        self._terminal_ui_shown = False
        self._failed = False
        self._recent_details: list[str] = []
        self._output_watch_timer = QTimer(self)
        self._output_watch_timer.setInterval(1000)
        self._output_watch_timer.timeout.connect(self.check_output_completion_from_disk)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_import_tab()
        self._build_options_tab()
        self._build_progress_tab()
        self._build_results_tab()

    def _build_import_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(DropZone(self.add_paths))

        buttons = QHBoxLayout()
        choose_files = QPushButton("Choose Files")
        choose_files.clicked.connect(self.choose_files)
        choose_folder = QPushButton("Choose Folder")
        choose_folder.clicked.connect(self.choose_folder)
        choose_output = QPushButton("Choose Output Folder")
        choose_output.clicked.connect(self.choose_output)
        self.output_label = QLabel(self.output_dir)
        buttons.addWidget(choose_files)
        buttons.addWidget(choose_folder)
        buttons.addWidget(choose_output)
        buttons.addWidget(self.output_label, 1)
        layout.addLayout(buttons)

        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(["Input", "Type"])
        self.file_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.file_table)

        actions = QHBoxLayout()
        self.inspect_btn = QPushButton("Inspect Selected")
        self.inspect_btn.clicked.connect(self.inspect_inputs)
        self.run_btn = QPushButton("Run Conversion")
        self.run_btn.clicked.connect(self.run_conversion)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_inputs)
        actions.addWidget(self.inspect_btn)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.clear_btn)
        layout.addLayout(actions)
        self.tabs.addTab(page, "Import")

    def _build_options_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)

        paths_box = QGroupBox("Optional mapping/config")
        form = QFormLayout(paths_box)
        self.config_path = QLineEdit()
        config_btn = QPushButton("Browse")
        config_btn.clicked.connect(lambda: self._pick_file(self.config_path, "Config YAML (*.yaml *.yml)"))
        row = QHBoxLayout(); row.addWidget(self.config_path); row.addWidget(config_btn)
        form.addRow("Project config", row)
        self.mapping_path = QLineEdit()
        mapping_btn = QPushButton("Browse")
        mapping_btn.clicked.connect(lambda: self._pick_file(self.mapping_path, "Mapping YAML (*.yaml *.yml)"))
        row2 = QHBoxLayout(); row2.addWidget(self.mapping_path); row2.addWidget(mapping_btn)
        form.addRow("Mapping YAML", row2)
        self.signal_path = QLineEdit()
        form.addRow("Signal path override", self.signal_path)
        self.sampling_rate = QLineEdit()
        form.addRow("Sampling rate override (Hz)", self.sampling_rate)
        layout.addWidget(paths_box)

        units_box = QGroupBox("Units/calibration")
        units = QFormLayout(units_box)
        self.original_units = QLineEdit()
        self.target_units = QLineEdit()
        self.scale_factor = QLineEdit()
        self.offset = QLineEdit()
        units.addRow("Original units", self.original_units)
        units.addRow("Target units", self.target_units)
        units.addRow("Scale factor", self.scale_factor)
        units.addRow("Offset", self.offset)
        layout.addWidget(units_box)

        proc_box = QGroupBox("Preprocessing and windows")
        proc = QFormLayout(proc_box)
        self.include_aux = QCheckBox("Keep auxiliary channels")
        proc.addRow(self.include_aux)
        self.preprocess = QCheckBox("Enable preprocessing")
        self.demean = QCheckBox("Demean")
        self.detrend = QCheckBox("Detrend")
        proc.addRow(self.preprocess)
        proc.addRow(self.demean)
        proc.addRow(self.detrend)
        self.normalization = QComboBox()
        self.normalization.addItems(["", "zscore", "robust", "minmax"])
        proc.addRow("Normalization", self.normalization)
        self.make_windows = QCheckBox("Create window/tensor output")
        self.window_seconds = QLineEdit()
        self.step_seconds = QLineEdit()
        proc.addRow(self.make_windows)
        proc.addRow("Window seconds", self.window_seconds)
        proc.addRow("Step seconds", self.step_seconds)
        layout.addWidget(proc_box)

        export_box = QGroupBox("Export")
        exp = QFormLayout(export_box)
        self.export_format = QComboBox()
        self.export_format.addItems(["npy", "memmap", "hdf5", "zarr"])
        self.no_signal_csv = QCheckBox("Skip signal.csv")
        self.no_signal_csv.setChecked(True)
        self.export_neural_signal = QCheckBox("Create notebook CSV exports: eeg_df.csv and neural_signal_with_time.csv (slow for large files)")
        self.export_neural_signal.setChecked(False)
        self.csv_max_mb = QLineEdit("250")
        exp.addRow("Signal array format", self.export_format)
        exp.addRow(self.no_signal_csv)
        exp.addRow(self.export_neural_signal)
        exp.addRow("CSV max MB", self.csv_max_mb)
        layout.addWidget(export_box)
        layout.addStretch(1)
        self.tabs.addTab(page, "Options")

    def _build_progress_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.progress_status = QLabel("Idle.")
        self.current_step_label = QLabel("Current step: —")
        self.last_completed_label = QLabel("Last completed: —")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Idle")
        self.progress.hide()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Compact heartbeat details will appear here.")
        layout.addWidget(self.progress_status)
        layout.addWidget(self.current_step_label)
        layout.addWidget(self.last_completed_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)
        self.tabs.addTab(page, "Progress")

    def _build_results_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.result_label = QLabel("No conversion has run yet.")
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        open_btn = QPushButton("Open Output Folder")
        open_btn.clicked.connect(self.open_output_folder)
        copy_btn = QPushButton("Copy Output Path to Log")
        copy_btn.clicked.connect(lambda: self._add_detail(self.output_dir))
        layout.addWidget(self.result_label)
        layout.addWidget(self.result_text)
        row = QHBoxLayout(); row.addWidget(open_btn); row.addWidget(copy_btn); row.addStretch(1)
        layout.addLayout(row)
        self.tabs.addTab(page, "Results")

    def add_paths(self, paths: list[str]) -> None:
        for p in paths:
            if p not in self.inputs:
                self.inputs.append(p)
        self.refresh_table()

    def refresh_table(self) -> None:
        self.file_table.setRowCount(len(self.inputs))
        for i, p in enumerate(self.inputs):
            path = Path(p)
            typ = "folder" if path.is_dir() else path.suffix.lower().lstrip('.') or "file"
            self.file_table.setItem(i, 0, QTableWidgetItem(str(path)))
            self.file_table.setItem(i, 1, QTableWidgetItem(typ))

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose neural signal files")
        self.add_paths(paths)

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose folder")
        if path:
            self.add_paths([path])

    def choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if path:
            self.output_dir = path
            self.output_label.setText(path)

    def _pick_file(self, target: QLineEdit, filter_text: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose file", filter=filter_text)
        if path:
            target.setText(path)

    def clear_inputs(self) -> None:
        self.inputs.clear()
        self.refresh_table()

    def _float_or_none(self, widget: QLineEdit) -> float | None:
        txt = widget.text().strip()
        if not txt:
            return None
        return float(txt)

    def build_options(self) -> dict[str, Any]:
        preprocess_config = None
        if self.preprocess.isChecked():
            preprocess_config = {"enabled": True, "demean": self.demean.isChecked(), "detrend": self.detrend.isChecked()}
            if self.normalization.currentText():
                preprocess_config["normalization"] = self.normalization.currentText()
        window_config = None
        if self.make_windows.isChecked():
            window_config = {"enabled": True}
            if self.window_seconds.text().strip():
                window_config["window_seconds"] = float(self.window_seconds.text())
            if self.step_seconds.text().strip():
                window_config["step_seconds"] = float(self.step_seconds.text())
        export_config = {"format": self.export_format.currentText(), "save_signal_csv": not self.no_signal_csv.isChecked()}
        if self.csv_max_mb.text().strip():
            export_config["csv_max_mb"] = float(self.csv_max_mb.text())
        unit_config = {}
        for key, widget in [
            ("original_units", self.original_units), ("target_units", self.target_units),
            ("scale_factor", self.scale_factor), ("offset", self.offset),
        ]:
            txt = widget.text().strip()
            if txt:
                unit_config[key] = float(txt) if key in {"scale_factor", "offset"} else txt
        return {
            "config_path": self.config_path.text().strip(),
            "mapping_path": self.mapping_path.text().strip(),
            "sampling_rate": self._float_or_none(self.sampling_rate),
            "signal_path": self.signal_path.text().strip(),
            "include_aux": self.include_aux.isChecked(),
            "preprocess_config": preprocess_config,
            "window_config": window_config,
            "export_config": export_config,
            "export_neural_signal": self.export_neural_signal.isChecked(),
            "unit_config": unit_config or None,
        }

    def _add_detail(self, message: str) -> None:
        self._recent_details.append(message)
        self._recent_details = self._recent_details[-8:]
        self.log.setPlainText("\n".join(self._recent_details))

    def set_running_state(self, running: bool) -> None:
        if running:
            self._terminal_ui_shown = False
            self._failed = False
            self._recent_details.clear()
            self.progress_status.setText("Conversion running")
            self.current_step_label.setText("Current step: Starting...")
            self.last_completed_label.setText("Last completed: —")
            self.progress.show()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("%p%")
            self.log.clear()
        else:
            self.ensure_terminal_ui_state()
        self.run_btn.setEnabled(not running)
        self.inspect_btn.setEnabled(not running)
        self.clear_btn.setEnabled(not running)

    def handle_worker_event(self, event: dict[str, Any]) -> None:
        status = str(event.get("status", "")).lower()
        stage = str(event.get("stage", "Pipeline"))
        message = str(event.get("message") or stage)
        percent = int(event.get("overall_percent", event.get("percent", 0)) or 0)
        item_index = event.get("item_index")
        total_items = event.get("total_items")
        item_prefix = f"File {item_index}/{total_items}: " if item_index and total_items else ""

        self.progress.show()
        self.progress.setRange(0, 100)
        self.progress.setValue(max(0, min(100, percent)))
        self.progress.setFormat("%p%")

        if status == "running":
            self.progress_status.setText(f"Conversion running — {percent}%")
            self.current_step_label.setText(f"Current step: {item_prefix}{message}")
        elif status == "complete":
            self.last_completed_label.setText(f"Last completed: ✓ {item_prefix}{stage}")
            if percent >= 100 or stage.lower() == "complete":
                self.mark_conversion_complete()
            else:
                self.progress_status.setText(f"Conversion running — {percent}%")
                self.current_step_label.setText("Current step: waiting for next stage...")
        elif status == "failed":
            self.progress_status.setText("Conversion failed")
            self.current_step_label.setText(f"Failed step: {item_prefix}{message}")
        else:
            self.progress_status.setText(f"Conversion running — {percent}%")
            self.current_step_label.setText(f"Current step: {item_prefix}{message}")

        symbol = "✓" if status == "complete" else "→" if status == "running" else "✗" if status == "failed" else "•"
        self._add_detail(f"{symbol} {item_prefix}{message}")

    def handle_worker_progress(self, message: str) -> None:
        # Compatibility for older worker messages; keep compact rather than appending forever.
        normalized = message.strip().lower()
        if normalized in {"done.", "done", "conversion complete", "conversion complete."}:
            self.mark_conversion_complete()
        else:
            self._add_detail(message)

    def update_progress_percent(self, value: int) -> None:
        safe_value = max(0, min(100, int(value)))
        self.progress.show()
        self.progress.setRange(0, 100)
        self.progress.setValue(safe_value)
        self.progress.setFormat("%p%")
        if safe_value >= 100:
            self.mark_conversion_complete()
        else:
            self.progress_status.setText(f"Conversion running — {safe_value}%")

    def mark_conversion_complete(self) -> None:
        self._terminal_ui_shown = True
        if self._output_watch_timer.isActive():
            self._output_watch_timer.stop()
        self.progress.show()
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setTextVisible(True)
        self.progress.setFormat("100% — Conversion complete")
        self.progress_status.setText("Conversion complete.")
        self.current_step_label.setText("Current step: Finished")
        self.last_completed_label.setText("Last completed: ✓ Complete")
        self.run_btn.setEnabled(True)
        self.inspect_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

    def ensure_terminal_ui_state(self) -> None:
        self.mark_conversion_complete()

    def expected_output_dirs(self) -> list[Path]:
        out = Path(self.output_dir)
        dirs: list[Path] = []
        for item in self.inputs:
            p = Path(item)
            if p.is_dir():
                dirs.append(out / p.name)
            else:
                dirs.append(out / (p.stem + "_converted"))
        return dirs

    def _dir_has_core_outputs(self, folder: Path) -> bool:
        if not folder.exists():
            return False
        signal_names = {"signal.npy", "signal_memmap.npy", "signal.h5"}
        has_signal = any((folder / name).exists() for name in signal_names) or any(folder.glob("signal.zarr*"))
        has_metadata = (folder / "metadata.json").exists()
        has_quality = (folder / "quality_report.json").exists()
        if has_signal and has_metadata and has_quality:
            return True
        for child in folder.rglob("metadata.json"):
            parent = child.parent
            if any((parent / name).exists() for name in signal_names) or any(parent.glob("signal.zarr*")):
                if (parent / "quality_report.json").exists():
                    return True
        return False

    def check_output_completion_from_disk(self) -> None:
        if self._terminal_ui_shown or not self.inputs:
            return
        expected = self.expected_output_dirs()
        if expected and all(self._dir_has_core_outputs(path) for path in expected):
            self._add_detail("Core converted files detected on disk. Conversion complete.")
            self.result_label.setText(f"Conversion complete. Finished files are located at: {self.output_dir}")
            self.result_text.setPlainText("Core converted files detected on disk. Open the output folder to view signal.npy, time.npy, channels.csv, metadata.json, quality_report.json, and provenance.json.")
            self.mark_conversion_complete()

    def inspect_inputs(self) -> None:
        if not self.inputs:
            QMessageBox.warning(self, "No inputs", "Drop or choose files/folders first.")
            return
        from neuro_importer.pipeline import NeuroImportPipeline
        pipe = NeuroImportPipeline()
        self._add_detail("Inspecting inputs...")
        for item in self.inputs:
            p = Path(item)
            if p.is_dir():
                self._add_detail(f"{p}: folder, use conversion for batch scan.")
                continue
            try:
                res = pipe.inspect(p)
                self.log.setPlainText(json.dumps(res, indent=2, default=str))
            except Exception as exc:
                self._add_detail(f"{p}: {exc}")
        self.tabs.setCurrentIndex(2)

    def run_conversion(self) -> None:
        if not self.inputs:
            QMessageBox.warning(self, "No inputs", "Drop or choose files/folders first.")
            return
        self.set_running_state(True)
        self.tabs.setCurrentIndex(2)
        self.thread = QThread()
        self.worker = PipelineWorker(inputs=self.inputs, output_dir=self.output_dir, options=self.build_options())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_event.connect(self.handle_worker_event)
        self.worker.progress.connect(self.handle_worker_progress)
        self.worker.progress_percent.connect(self.update_progress_percent)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.done.connect(self.thread.quit)
        self.worker.done.connect(self.on_worker_done)
        self.thread.finished.connect(self.thread.deleteLater)
        self._output_watch_timer.start()
        self.thread.start()

    def on_worker_done(self) -> None:
        if not self._terminal_ui_shown and not self._failed:
            self.mark_conversion_complete()

    def on_finished(self, result: dict[str, Any]) -> None:
        self.mark_conversion_complete()
        self.last_result = result
        self._add_detail(f"Conversion complete. Finished files are located at: {self.output_dir}")
        self.result_label.setText(f"Conversion complete. Finished files are located at: {self.output_dir}")
        self.result_text.setPlainText(json.dumps(result, indent=2, default=str))
        self.tabs.setCurrentIndex(3)

    def on_failed(self, message: str) -> None:
        self._failed = True
        self._terminal_ui_shown = True
        if self._output_watch_timer.isActive():
            self._output_watch_timer.stop()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Conversion failed")
        self.progress_status.setText("Conversion failed.")
        self.current_step_label.setText("Current step: failed")
        self.progress.show()
        self.run_btn.setEnabled(True)
        self.inspect_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self._add_detail("FAILED: " + message)
        QMessageBox.critical(self, "Conversion failed", message)

    def open_output_folder(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.output_dir)))
