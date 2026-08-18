"""Main window: canvas in the middle, palette left, controls right."""

from __future__ import annotations

import os
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import instrument as inst_mod
from ..core import writer
from ..core.drawing import CellDrawing, DEFAULT_SUPERSAMPLE
from ..core.materials import MaterialLibrary, DEFAULT_LIBRARY_PATH
from ..core.scan import ScanSettings, simulate_scan
from . import canvas_widget as cw
from .batch_panel import BatchPanel
from .canvas_widget import CanvasWidget
from .material_panel import MaterialPanel
from .preview_panel import PreviewPanel
from .scan_panel import ScanPanel

DARK_QSS = """
QWidget { background:#181d26; color:#c8d0dc; font-size:12px; }
QGroupBox { border:1px solid #2c3342; border-radius:4px; margin-top:8px; padding-top:8px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; color:#8892a4; }
QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QComboBox, QListWidget {
    background:#11151c; border:1px solid #2c3342; border-radius:3px; padding:3px; }
QPushButton { background:#232a36; border:1px solid #39424f; border-radius:3px; padding:5px 10px; }
QPushButton:hover { background:#2c3442; }
QPushButton:disabled { color:#5a6473; }
QPushButton:checked { background:#3d5a80; border-color:#5b83b8; }
QTabBar::tab { background:#1d232e; padding:6px 12px; border:1px solid #2c3342; }
QTabBar::tab:selected { background:#2b3444; }
QProgressBar { background:#11151c; border:1px solid #2c3342; border-radius:3px; text-align:center; }
QProgressBar::chunk { background:#3d5a80; }
QToolBar { background:#1d232e; border-bottom:1px solid #2c3342; spacing:3px; }
QDockWidget::title { background:#1d232e; padding:5px; }
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brillouin Sample Simulator")
        self.resize(1500, 940)

        self.library = MaterialLibrary.load()
        self.instrument = inst_mod.preset_narrow()
        self._drawing_path = None
        self._preview_pending = False

        self.canvas = CanvasWidget()
        self.canvas.set_palette(list(self.library))
        self.setCentralWidget(self.canvas)

        self.material_panel = MaterialPanel(self.library, self.instrument)
        self.scan_panel = ScanPanel(ScanSettings(), self.instrument)
        self.preview_panel = PreviewPanel()
        self.batch_panel = BatchPanel()

        self._build_docks()
        self._build_toolbar()
        self._build_menus()
        self._wire()

        self.statusBar().showMessage("Ready")
        self.new_sample(ask=False)

        # Preview recomputes on a short delay so painting stays responsive.
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(350)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._schedule_preview()

    # -- construction --------------------------------------------------------

    def _build_docks(self):
        left = QtWidgets.QDockWidget("Materials", self)
        left.setWidget(self._wrap(self.material_panel))
        left.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, left)
        left.setMinimumWidth(260)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._wrap(self.scan_panel), "Settings")
        tabs.addTab(self._wrap(self.preview_panel), "Preview")
        tabs.addTab(self._wrap(self.batch_panel), "Batch")
        self._right_tabs = tabs

        right = QtWidgets.QDockWidget("Simulation", self)
        right.setWidget(tabs)
        right.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, right)
        right.setMinimumWidth(430)

    @staticmethod
    def _wrap(widget):
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(widget)
        return holder

    def _build_toolbar(self):
        bar = self.addToolBar("Tools")
        bar.setMovable(False)

        self._tool_group = QtWidgets.QButtonGroup(self)
        self._tool_group.setExclusive(True)
        tools = [("Pencil", cw.PENCIL, "Freehand brush"),
                 ("Line", cw.LINE, "Straight line"),
                 ("Rect", cw.RECT, "Rectangle"),
                 ("Ellipse", cw.ELLIPSE, "Ellipse - the usual cell or nucleus shape"),
                 ("Polygon", cw.POLYGON, "Click points, double-click or Enter to close"),
                 ("Fill", cw.FILL, "Flood fill a connected region"),
                 ("Eraser", cw.ERASER, "Paint back to the background material"),
                 ("Picker", cw.PICKER, "Pick the material under the cursor")]
        for label, tool, tip in tools:
            b = QtWidgets.QToolButton()
            b.setText(label)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, t=tool: self._set_tool(t))
            self._tool_group.addButton(b)
            bar.addWidget(b)
            if tool == cw.PENCIL:
                b.setChecked(True)

        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel(" Size "))
        self.brush_spin = QtWidgets.QSpinBox()
        self.brush_spin.setRange(1, 400)
        self.brush_spin.setValue(self.canvas.brush_size)
        self.brush_spin.setToolTip("Brush and outline width, in canvas cells")
        self.brush_spin.valueChanged.connect(
            lambda v: setattr(self.canvas, "brush_size", v))
        bar.addWidget(self.brush_spin)

        self.filled_check = QtWidgets.QCheckBox("Filled")
        self.filled_check.setChecked(True)
        self.filled_check.toggled.connect(
            lambda v: setattr(self.canvas, "filled_shapes", v))
        bar.addWidget(self.filled_check)

        bar.addSeparator()
        self.grid_check = QtWidgets.QCheckBox("Scan grid")
        self.grid_check.setChecked(True)
        self.grid_check.setToolTip("Show which canvas cells fall in which scan pixel")
        self.grid_check.toggled.connect(self._toggle_grid)
        bar.addWidget(self.grid_check)

        for label, slot in [("Fit", self.canvas.fit_to_view),
                            ("Undo", self.canvas.undo),
                            ("Redo", self.canvas.redo)]:
            b = QtWidgets.QToolButton()
            b.setText(label)
            b.clicked.connect(slot)
            bar.addWidget(b)

    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        self._act(m, "&New sample...", "Ctrl+N", self.new_sample)
        self._act(m, "&Open sample...", "Ctrl+O", self.open_sample)
        self._act(m, "&Save sample", "Ctrl+S", self.save_sample)
        self._act(m, "Save sample &as...", "Ctrl+Shift+S", self.save_sample_as)
        m.addSeparator()
        self._act(m, "Generate a single scan...", "Ctrl+G", self.generate_single)
        m.addSeparator()
        self._act(m, "Save material library", "", self.save_library)
        m.addSeparator()
        self._act(m, "E&xit", "Ctrl+Q", self.close)

        e = self.menuBar().addMenu("&Edit")
        self._act(e, "&Undo", "Ctrl+Z", self.canvas.undo)
        self._act(e, "&Redo", "Ctrl+Y", self.canvas.redo)
        e.addSeparator()
        self._act(e, "Fill canvas with selected material", "",
                  self.fill_canvas)
        self._act(e, "Clear to background", "", self.clear_canvas)

        v = self.menuBar().addMenu("&View")
        self._act(v, "Fit to window", "Ctrl+0", self.canvas.fit_to_view)
        self._act(v, "Refresh preview", "F5", self._refresh_preview)

        h = self.menuBar().addMenu("&Help")
        self._act(h, "About", "", self.show_about)

    def _act(self, menu, text, shortcut, slot):
        action = QtGui.QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _wire(self):
        self.material_panel.material_selected.connect(self._on_material_selected)
        self.material_panel.library_changed.connect(self._on_library_changed)
        self.canvas.material_picked.connect(self.material_panel.select_index)
        self.canvas.drawing_changed.connect(self._schedule_preview)
        self.canvas.cursor_moved.connect(self._on_cursor)
        self.scan_panel.settings_changed.connect(self._on_settings_changed)
        self.scan_panel.instrument_changed.connect(self._schedule_preview)
        self.batch_panel.request_context.connect(self._provide_batch_context)
        self.batch_panel.log_message.connect(
            lambda msg: self.statusBar().showMessage(msg, 15000))

    # -- slots ---------------------------------------------------------------

    def _set_tool(self, tool):
        self.canvas.tool = tool

    def _toggle_grid(self, on):
        self.canvas.show_scan_grid = on
        self.canvas.update()

    def _on_material_selected(self, index):
        self.canvas.material_index = index

    def _on_library_changed(self):
        self.canvas.set_palette(list(self.library))
        self._schedule_preview()

    def _on_settings_changed(self):
        s = self.scan_panel.settings()
        self.canvas.set_scan_grid(s.grid_w, s.grid_h)
        self._schedule_preview()

    def _on_cursor(self, x_um, y_um, index):
        name = (self.library[index].name
                if 0 <= index < len(self.library) else "-")
        self.statusBar().showMessage(f"{x_um:.2f}, {y_um:.2f} um    {name}")

    def _schedule_preview(self):
        if hasattr(self, "_preview_timer"):
            self._preview_timer.start()

    def _refresh_preview(self):
        try:
            self.preview_panel.update_preview(
                self.canvas.drawing, self.library, self.instrument,
                self.scan_panel.settings())
        except Exception:
            self.statusBar().showMessage("Preview failed - see console", 8000)
            traceback.print_exc()

    def _provide_batch_context(self):
        self.batch_panel.set_context(
            self.canvas.drawing, self.library, self.instrument,
            self.scan_panel.settings())

    # -- file actions --------------------------------------------------------

    def new_sample(self, ask=True):
        s = self.scan_panel.settings()
        supersample = DEFAULT_SUPERSAMPLE
        if ask:
            value, ok = QtWidgets.QInputDialog.getInt(
                self, "New sample",
                "Canvas sub-cells per scan pixel.\n\n"
                "Higher means finer detail and more accurate sub-pixel mixing;\n"
                "8 is a good balance.",
                DEFAULT_SUPERSAMPLE, 1, 32)
            if not ok:
                return
            supersample = value
        drawing = CellDrawing.for_scan(s.grid_w, s.grid_h, s.step_um,
                                       supersample=supersample,
                                       material_names=list(self.library.names))
        self.canvas.set_drawing(drawing)
        self.canvas.set_scan_grid(s.grid_w, s.grid_h)
        self._drawing_path = None
        self._schedule_preview()

    def open_sample(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open sample", "", "Sample (*.npz)")
        if not path:
            return
        try:
            drawing = CellDrawing.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(exc))
            return
        missing = drawing.reconcile_with_library(self.library)
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Materials missing",
                "This sample used materials that are not in your library:\n\n"
                + "\n".join(f"  - {m}" for m in missing)
                + "\n\nThose areas were reset to the background material.")
        self.canvas.set_drawing(drawing)
        self._drawing_path = path
        self.setWindowTitle(f"Brillouin Sample Simulator - {os.path.basename(path)}")
        self._schedule_preview()

    def save_sample(self):
        if self._drawing_path is None:
            return self.save_sample_as()
        self.canvas.drawing.material_names = list(self.library.names)
        self.canvas.drawing.save(self._drawing_path)
        self.statusBar().showMessage(f"Saved {self._drawing_path}", 6000)

    def save_sample_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save sample", "sample.bcell.npz", "Sample (*.npz)")
        if not path:
            return
        self.canvas.drawing.material_names = list(self.library.names)
        self._drawing_path = self.canvas.drawing.save(path)
        self.setWindowTitle(
            f"Brillouin Sample Simulator - {os.path.basename(self._drawing_path)}")
        self.statusBar().showMessage(f"Saved {self._drawing_path}", 6000)

    def save_library(self):
        path = self.library.save(DEFAULT_LIBRARY_PATH)
        self.statusBar().showMessage(f"Material library saved to {path}", 6000)

    def generate_single(self):
        """Write one scan with the current settings."""
        if self.canvas.drawing is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save simulated scan",
            writer.default_stem("sample", self.scan_panel.settings()) + ".csv",
            "CSV (*.csv)")
        if not path:
            return

        settings = self.scan_panel.settings()
        dialog = QtWidgets.QProgressDialog("Simulating scan...", "Cancel", 0, 100, self)
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setMinimumDuration(0)

        def progress(done, total, message):
            dialog.setLabelText(message)
            dialog.setValue(int(100 * done / max(total, 1)))
            QtWidgets.QApplication.processEvents()
            return not dialog.wasCanceled()

        try:
            result = simulate_scan(self.canvas.drawing, self.library,
                                   self.instrument, settings, progress=progress)
        except Exception:
            dialog.close()
            QtWidgets.QMessageBox.critical(self, "Simulation failed",
                                           traceback.format_exc())
            return
        dialog.close()
        if result is None:
            self.statusBar().showMessage("Cancelled", 5000)
            return

        writer.write_scan_csv(path, result, self.instrument)
        stem = os.path.splitext(path)[0]
        writer.write_truth_sidecar(stem + "_truth.npz", result, self.instrument,
                                   os.path.basename(stem))
        err = np.median(np.abs(result.bfs_noise_error_mhz()))
        self.statusBar().showMessage(
            f"Wrote {os.path.basename(path)} and its truth sidecar - "
            f"median shift error {err:.2f} MHz", 15000)

    def fill_canvas(self):
        if self.canvas.drawing is None:
            return
        self.canvas.push_undo()
        self.canvas.drawing.fill_all(self.canvas.material_index)
        self.canvas._after_edit()

    def clear_canvas(self):
        if self.canvas.drawing is None:
            return
        self.canvas.push_undo()
        self.canvas.drawing.fill_all(0)
        self.canvas._after_edit()

    def show_about(self):
        QtWidgets.QMessageBox.information(
            self, "About",
            "<b>Brillouin Sample Simulator</b><br><br>"
            "Paint a synthetic cell out of named materials, choose a scan "
            "resolution, and generate CSV files with known ground truth.<br><br>"
            "Each material contributes an area-normalised Lorentzian weighted "
            "by how much of the pixel it occupies, so a mixed pixel keeps the "
            "same total peak area.<br><br>"
            "Thermal and laser noise follow Oscar Nieves, <i>Noise in "
            "Stimulated Brillouin Scattering</i> (2021), chapters 3 and 5. "
            "Shot noise, dark counts and read noise are added on top so the "
            "output matches real photon-counting data.")

    def closeEvent(self, event):
        try:
            self.library.save(DEFAULT_LIBRARY_PATH)
        except Exception:
            pass
        super().closeEvent(event)
