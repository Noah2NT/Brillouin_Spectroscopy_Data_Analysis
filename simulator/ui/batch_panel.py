"""Batch generation: sweep the noise level, repeat each setting, write files.

Generation runs on a worker thread.  The core is pure numpy and never touches
Qt, so the worker only has to marshal progress back through signals.
"""

from __future__ import annotations

import os
import traceback

from PySide6 import QtCore, QtWidgets

from ..core import batch as batch_mod


class BatchWorker(QtCore.QObject):
    """Runs :func:`simulator.core.batch.run_batch` off the GUI thread."""

    progress = QtCore.Signal(int, int, float, str)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, drawing, library, instrument, spec, settings):
        super().__init__()
        self._args = (drawing, library, instrument, spec, settings)
        self._cancel = False

    @QtCore.Slot()
    def run(self):
        drawing, library, instrument, spec, settings = self._args
        try:
            result = batch_mod.run_batch(
                drawing, library, instrument, spec, settings,
                progress=lambda i, n, f, m: self.progress.emit(i, n, f, m),
                should_cancel=lambda: self._cancel,
            )
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    @QtCore.Slot()
    def cancel(self):
        self._cancel = True


class BatchPanel(QtWidgets.QWidget):
    """Define the sweep and run it."""

    request_context = QtCore.Signal()
    log_message = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._context = None

        f = QtWidgets.QFormLayout()

        self.exposures = QtWidgets.QLineEdit("1, 2, 5, 10, 20")
        self.exposures.setToolTip(
            "Comma-separated dwell times in seconds. This is the noise axis:\n"
            "shot-noise-limited precision improves as the square root of it.")
        f.addRow("Exposures (s)", self.exposures)

        self.repeats = QtWidgets.QSpinBox()
        self.repeats.setRange(1, 200)
        self.repeats.setValue(3)
        self.repeats.setToolTip(
            "Independent realisations per exposure, each with its own seed,\n"
            "so you get error bars rather than one lucky run.")
        f.addRow("Repeats per setting", self.repeats)

        self.steps = QtWidgets.QLineEdit("")
        self.steps.setPlaceholderText("blank = use the scan panel's step size")
        self.steps.setToolTip(
            "Optional second axis: comma-separated step sizes in microns, to\n"
            "test how algorithms cope with coarser sampling and heavier mixing.")
        f.addRow("Step sizes (um)", self.steps)

        self.name = QtWidgets.QLineEdit("sample")
        f.addRow("Name prefix", self.name)

        out_row = QtWidgets.QHBoxLayout()
        self.out_dir = QtWidgets.QLineEdit(os.path.join("outputs", "simulated"))
        browse = QtWidgets.QPushButton("...")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse)
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(browse)
        holder = QtWidgets.QWidget()
        holder.setLayout(out_row)
        f.addRow("Output folder", holder)

        self.crlf = QtWidgets.QCheckBox("Windows line endings (match data/csv)")
        f.addRow("", self.crlf)

        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#8892a4; font-size:11px;")

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QtWidgets.QLabel("Idle")
        self.status.setStyleSheet("color:#8892a4; font-size:11px;")

        self.run_btn = QtWidgets.QPushButton("Generate batch")
        self.run_btn.clicked.connect(self.start)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.run_btn, 1)
        btn_row.addWidget(self.cancel_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(f)
        layout.addWidget(self.summary)
        layout.addLayout(btn_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addStretch(1)

        for w in (self.exposures, self.steps):
            w.textChanged.connect(self._refresh_summary)
        self.repeats.valueChanged.connect(self._refresh_summary)
        self._refresh_summary()

    # -- helpers -------------------------------------------------------------

    def _browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Output folder", self.out_dir.text())
        if path:
            self.out_dir.setText(path)

    @staticmethod
    def _parse_list(text):
        out = []
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                value = float(part)
            except ValueError:
                continue
            if value > 0:
                out.append(value)
        return out

    def spec(self):
        return batch_mod.BatchSpec(
            exposures_s=self._parse_list(self.exposures.text()) or [10.0],
            repeats=self.repeats.value(),
            step_sizes_um=self._parse_list(self.steps.text()) or None,
            output_dir=self.out_dir.text().strip() or "outputs/simulated",
            drawing_name=self.name.text().strip() or "sample",
            seed_base=None,
            line_terminator="\r\n" if self.crlf.isChecked() else "\n",
        )

    def _refresh_summary(self):
        spec = self.spec()

    def set_context(self, drawing, library, instrument, settings):
        self._context = (drawing, library, instrument, settings)

    # -- run -----------------------------------------------------------------

    def start(self):
        self.request_context.emit()
        if self._context is None:
            QtWidgets.QMessageBox.warning(self, "Nothing to generate",
                                          "Create or open a sample first.")
            return
        if self._thread is not None:
            return

        drawing, library, instrument, settings = self._context
        spec = self.spec()
        problems = instrument.validate()
        if problems:
            QtWidgets.QMessageBox.warning(self, "Instrument settings",
                                          "\n".join(problems))
            return

        estimate = batch_mod.estimate_runtime(spec, settings)
        if estimate > 120:
            mins = estimate / 60.0
            if QtWidgets.QMessageBox.question(
                    self, "Long run",
                    f"{spec.n_jobs} scans of {settings.n_pixels} pixels will "
                    f"take roughly {mins:.0f} minutes. Continue?"
            ) != QtWidgets.QMessageBox.Yes:
                return

        self._thread = QtCore.QThread(self)
        self._worker = BatchWorker(drawing, library, instrument, spec, settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Starting...")

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Cancelling after the current scan...")

    def _on_progress(self, index, total, fraction, message):
        overall = (index + fraction) / max(total, 1)
        self.progress.setValue(int(overall * 1000))
        self.status.setText(f"[{index + 1}/{total}] {message}")

    def _on_finished(self, result):
        self._teardown()
        n = len(result["files"])
        if result["cancelled"]:
            self.status.setText(f"Cancelled after {n} file(s).")
            self.log_message.emit(f"Batch cancelled after {n} file(s).")
        else:
            self.progress.setValue(1000)
            self.status.setText(f"Wrote {n} file(s) and manifest.csv")
            self.log_message.emit(
                f"Batch complete: {n} CSV file(s) plus sidecars in "
                f"{os.path.abspath(os.path.dirname(result['manifest']))}")

    def _on_failed(self, message):
        self._teardown()
        self.status.setText("Failed")
        QtWidgets.QMessageBox.critical(self, "Batch failed", message)

    def _teardown(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
