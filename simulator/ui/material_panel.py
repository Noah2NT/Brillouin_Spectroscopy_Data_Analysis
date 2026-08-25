"""Material palette and the material editor dialog."""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from ..core import physics, spectrum
from ..core.materials import Material, MaterialLibrary


class MaterialDialog(QtWidgets.QDialog):
    """Create or edit one material.

    The Brillouin shift can be typed directly, which is how the values in this
    project are actually known, or computed from refractive index and acoustic
    velocity using thesis Eq. (1.1).
    """

    def __init__(self, material, instrument, parent=None, title="Edit material"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        self._instrument = instrument
        self._color = QtGui.QColor(material.color)

        form = QtWidgets.QFormLayout()

        self.name = QtWidgets.QLineEdit(material.name)
        form.addRow("Name", self.name)

        self.color_btn = QtWidgets.QPushButton()
        self.color_btn.setFixedHeight(26)
        self._refresh_color_button()
        self.color_btn.clicked.connect(self._choose_color)
        form.addRow("Colour", self.color_btn)

        self.bfs = self._spin(material.bfs_ghz, 0.001, 100.0, 3, " GHz")
        form.addRow("Brillouin shift", self.bfs)

        self.observed = self._spin(0.0, 0.0, 40.0, 4, " GHz")
        self.observed.setToolTip(
            "The linewidth you actually measure -- what a fit of the recorded "
            "spectrum returns, instrument broadening included. This is what "
            "the Half Width column of your CSVs reports (doubled, since that "
            "column is a half width). Type your measured value here and the "
            "intrinsic width below is solved for you.")
        form.addRow("Observed linewidth", self.observed)

        self.linewidth = self._spin(material.linewidth_ghz, 0.0, 20.0, 4, " GHz")
        self.linewidth.setToolTip(
            "The material's own linewidth, with the instrument taken back "
            "out. This is what gets stored, because the observed width "
            "depends on instrument settings and this does not. Edit either "
            "field; the other follows.")
        form.addRow("Intrinsic linewidth", self.linewidth)

        self.width_note = QtWidgets.QLabel("")
        self.width_note.setWordWrap(True)
        self.width_note.setStyleSheet("color:#8892a4; font-size:11px;")
        form.addRow("", self.width_note)

        self.b_eff = self._spin(material.brillouin_efficiency, 0.0, 100.0, 3, "")
        form.addRow("Brillouin efficiency", self.b_eff)

        self.e_eff = self._spin(material.elastic_efficiency, 0.0, 100.0, 3, "")
        form.addRow("Elastic efficiency", self.e_eff)

        self.temp_scale = QtWidgets.QCheckBox("Signal scales with temperature")
        self.temp_scale.setChecked(material.temperature_scaling)
        self.temp_scale.setToolTip(
            "Thesis Eq. (3.12): sigma = k_B * T * alpha_ac, so Brillouin\n"
            "signal is proportional to absolute temperature.")
        form.addRow("", self.temp_scale)

        box = QtWidgets.QGroupBox("Optical properties (optional)")
        box.setToolTip("Used only by the shift calculator below.")
        bl = QtWidgets.QFormLayout(box)
        self.n_idx = self._spin(material.refractive_index or 0.0, 0.0, 5.0, 4, "")
        self.v_a = self._spin(material.acoustic_velocity_ms or 0.0, 0.0, 20000.0, 1, " m/s")
        bl.addRow("Refractive index", self.n_idx)
        bl.addRow("Acoustic velocity", self.v_a)
        calc = QtWidgets.QPushButton("Compute shift from these  (Eq. 1.1)")
        calc.clicked.connect(self._compute_bfs)
        bl.addRow("", calc)
        self.calc_note = QtWidgets.QLabel("")
        self.calc_note.setWordWrap(True)
        self.calc_note.setStyleSheet("color:#8892a4;")
        bl.addRow("", self.calc_note)

        self.notes = QtWidgets.QPlainTextEdit(material.notes)
        self.notes.setFixedHeight(52)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(box)
        layout.addWidget(QtWidgets.QLabel("Notes"))
        layout.addWidget(self.notes)
        layout.addWidget(buttons)

        self.linewidth.valueChanged.connect(self._on_intrinsic_changed)
        self.observed.valueChanged.connect(self._on_observed_changed)
        self._syncing = False
        self._on_intrinsic_changed()

    @staticmethod
    def _spin(value, lo, hi, decimals, suffix):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(10 ** -decimals * 10)
        s.setSuffix(suffix)
        s.setValue(float(value))
        return s

    def _refresh_color_button(self):
        self.color_btn.setStyleSheet(
            f"background:{self._color.name()}; border:1px solid #555;")
        self.color_btn.setText(self._color.name())

    def _choose_color(self):
        c = QtWidgets.QColorDialog.getColor(self._color, self, "Material colour")
        if c.isValid():
            self._color = c
            self._refresh_color_button()

    def _on_intrinsic_changed(self):
        """Intrinsic edited -> recompute what would be observed."""
        if getattr(self, "_syncing", False):
            return
        self._syncing = True
        value = spectrum.observed_linewidth_from_intrinsic(
            self.linewidth.value(), self._instrument)
        self.observed.setValue(value)
        self._syncing = False
        self._refresh_width_note()

    def _on_observed_changed(self):
        """Observed edited -> solve back for the intrinsic width."""
        if getattr(self, "_syncing", False):
            return
        intrinsic = spectrum.intrinsic_linewidth_for_observed(
            self.observed.value(), self._instrument)
        self._syncing = True
        self.linewidth.setValue(0.0 if intrinsic is None else intrinsic)
        self._syncing = False
        self._refresh_width_note()

    def _refresh_width_note(self):
        floor = spectrum.instrument_linewidth_floor(self._instrument)

    def _compute_bfs(self):
        if self.n_idx.value() <= 0 or self.v_a.value() <= 0:
            self.calc_note.setText("Set both refractive index and acoustic velocity first.")
            return
        value = physics.bfs_from_optics(
            self.n_idx.value(), self.v_a.value(),
            self._instrument.wavelength_nm, self._instrument.scattering_angle_deg)
        self.bfs.setValue(value)
        self.calc_note.setText(
            f"nu_B = 2 n v_a sin(theta/2) / lambda = {value:.4f} GHz "
            f"at lambda = {self._instrument.wavelength_nm:.0f} nm, "
            f"theta = {self._instrument.scattering_angle_deg:.0f} deg.")

    def _accept(self):
        problems = self.result_material().validate()
        if problems:
            QtWidgets.QMessageBox.warning(self, "Invalid material", "\n".join(problems))
            return
        self.accept()

    def result_material(self):
        return Material(
            name=self.name.text().strip() or "Unnamed",
            color=self._color.name(),
            bfs_ghz=self.bfs.value(),
            linewidth_ghz=self.linewidth.value(),
            brillouin_efficiency=self.b_eff.value(),
            elastic_efficiency=self.e_eff.value(),
            refractive_index=self.n_idx.value() or None,
            acoustic_velocity_ms=self.v_a.value() or None,
            temperature_scaling=self.temp_scale.isChecked(),
            notes=self.notes.toPlainText(),
        )


class MaterialPanel(QtWidgets.QWidget):
    """The palette: pick the active material, and manage the library."""

    material_selected = QtCore.Signal(int)
    library_changed = QtCore.Signal()

    def __init__(self, library, instrument, parent=None):
        super().__init__(parent)
        self.library = library
        self.instrument = instrument

        self.list = QtWidgets.QListWidget()
        self.list.setIconSize(QtCore.QSize(18, 18))
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemDoubleClicked.connect(lambda _: self.edit_current())

        buttons = QtWidgets.QGridLayout()
        for i, (label, slot, tip) in enumerate([
            ("Add", self.add_material, "Create a new material"),
            ("Edit", self.edit_current, "Edit the selected material"),
            ("Duplicate", self.duplicate_current, "Copy the selected material"),
            ("Delete", self.delete_current, "Remove the selected material"),
        ]):
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            buttons.addWidget(b, i // 2, i % 2)


        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)

        self.refresh()

    # -- helpers -------------------------------------------------------------

    def refresh(self, select=None):
        current = self.list.currentRow() if select is None else select
        self.list.blockSignals(True)
        self.list.clear()
        for m in self.library:
            pix = QtGui.QPixmap(18, 18)
            pix.fill(QtGui.QColor(m.color))
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(pix),
                                             f"{m.name}   {m.bfs_ghz:.3f} GHz")
            item.setToolTip(
                f"{m.name}\nShift {m.bfs_ghz:.4f} GHz\n"
                f"Observed linewidth {spectrum.observed_linewidth(m, self.instrument):.4f} GHz\n"
                f"Intrinsic linewidth {m.linewidth_ghz:.4f} GHz\n"
                f"Brillouin efficiency {m.brillouin_efficiency:g}\n"
                f"Elastic efficiency {m.elastic_efficiency:g}\n"
                + (m.notes or ""))
            self.list.addItem(item)
        self.list.blockSignals(False)
        if len(self.library):
            self.list.setCurrentRow(min(max(current, 0), len(self.library) - 1))

    def current_index(self):
        return max(self.list.currentRow(), 0)

    def _on_row_changed(self, row):
        if row >= 0:
            self.material_selected.emit(row)

    def select_index(self, index):
        if 0 <= index < len(self.library):
            self.list.setCurrentRow(index)

    # -- actions -------------------------------------------------------------

    def add_material(self):
        dlg = MaterialDialog(Material(name="New material", color="#66bb6a"),
                             self.instrument, self, "Add material")
        if dlg.exec():
            self.library.add(dlg.result_material())
            self.refresh(len(self.library) - 1)
            self.library_changed.emit()

    def edit_current(self):
        i = self.current_index()
        if not len(self.library):
            return
        dlg = MaterialDialog(self.library[i], self.instrument, self)
        if dlg.exec():
            self.library.update(i, dlg.result_material())
            self.refresh(i)
            self.library_changed.emit()

    def duplicate_current(self):
        i = self.current_index()
        if not len(self.library):
            return
        src = self.library[i]
        self.library.add(src.copy_with(name=src.name + " copy"))
        self.refresh(len(self.library) - 1)
        self.library_changed.emit()

    def delete_current(self):
        i = self.current_index()
        if i == 0:
            QtWidgets.QMessageBox.information(
                self, "Background material",
                "The first material is the canvas background and cannot be "
                "deleted. Edit it instead, or move another material up.")
            return
        name = self.library[i].name
        if QtWidgets.QMessageBox.question(
                self, "Delete material",
                f"Delete '{name}'?\n\nAnything painted with it in the current "
                f"drawing becomes background.") != QtWidgets.QMessageBox.Yes:
            return
        self.library.remove(i)
        self.refresh(max(i - 1, 0))
        self.library_changed.emit()
