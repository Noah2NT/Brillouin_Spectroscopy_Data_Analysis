"""Scan geometry, noise and instrument controls."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..core import physics
from ..core.instrument import PRESETS
from ..core.scan import ScanSettings


def _spin(value, lo, hi, decimals=3, suffix="", step=None):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setSingleStep(step if step is not None else 10 ** -decimals * 10)
    s.setValue(float(value))
    return s


def _int_spin(value, lo, hi, suffix=""):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setSuffix(suffix)
    s.setValue(int(value))
    return s


class ScanPanel(QtWidgets.QWidget):
    """Everything about how the scan is acquired."""

    settings_changed = QtCore.Signal()
    instrument_changed = QtCore.Signal()

    def __init__(self, settings, instrument, parent=None):
        super().__init__(parent)
        self.instrument = instrument

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_scan_tab(settings), "Scan")
        tabs.addTab(self._build_noise_tab(settings), "Noise")
        tabs.addTab(self._build_instrument_tab(instrument), "Instrument")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)

        self._wire()

    # -- tabs ----------------------------------------------------------------

    def _build_scan_tab(self, s):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)

        self.grid_w = _int_spin(s.grid_w, 2, 500, " px")
        self.grid_h = _int_spin(s.grid_h, 2, 500, " px")
        self.step_um = _spin(s.step_um, 0.01, 500.0, 3, " um")
        f.addRow("Grid width", self.grid_w)
        f.addRow("Grid height", self.grid_h)
        f.addRow("Step size", self.step_um)

        self.span = QtWidgets.QLabel("-")
        self.span.setStyleSheet("color:#8892a4;")
        f.addRow("Scanned area", self.span)

        self.exposure = _spin(s.exposure_s, 0.01, 600.0, 2, " s")
        self.exposure.setToolTip(
            "Dwell time per pixel. The main noise knob: peak area scales with\n"
            "it, so shot-noise-limited precision improves as its square root.")
        f.addRow("Exposure", self.exposure)

        self.psf = _spin(s.psf_fwhm_um, 0.0, 20.0, 3, " um")
        self.psf.setToolTip(
            "Confocal spot size (FWHM). Blurs material composition before the\n"
            "canvas is downsampled, so pixels mix optically as well as\n"
            "geometrically. Set to 0 for purely geometric coverage.")
        f.addRow("PSF spot size", self.psf)

        self.reference_fit = QtWidgets.QCheckBox(
            "Also fit the noise-free spectrum (recommended)")
        self.reference_fit.setChecked(s.reference_fit)
        self.reference_fit.setToolTip(
            "Records what the same fitter returns with no noise at all.\n"
            "Comparing a noisy fit against that isolates error caused by\n"
            "noise from error caused by the single-Lorentzian model.\n"
            "Roughly doubles generation time.")
        f.addRow("", self.reference_fit)
        return w

    def _build_noise_tab(self, s):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.temperature = _spin(s.temperature_k, 1.0, 1000.0, 1, " K")
        self.temperature.setToolTip(
            "Thesis Eq. (3.12): sigma = k_B * T * alpha_ac, so Brillouin\n"
            "signal is proportional to absolute temperature.")
        f.addRow("Temperature", self.temperature)

        self.speckle_auto = QtWidgets.QCheckBox("Derive from exposure and linewidth")
        self.speckle_auto.setChecked(s.speckle_modes is None)
        self.speckle_modes = _spin(s.speckle_modes or 1000.0, 1.0, 1e12, 1, "")
        self.speckle_modes.setEnabled(s.speckle_modes is not None)
        self.speckle_note = QtWidgets.QLabel("")
        self.speckle_note.setWordWrap(True)
        self.speckle_note.setStyleSheet("color:#8892a4; font-size:11px;")
        f.addRow("Thermal speckle modes", self.speckle_auto)
        f.addRow("", self.speckle_modes)
        f.addRow("", self.speckle_note)

        self.jitter = _spin(s.bfs_jitter_mhz, 0.0, 500.0, 1, " MHz")
        self.jitter.setToolTip(
            "Per-pixel random shift of the whole spectrum, modelling slow\n"
            "laser frequency drift.")
        f.addRow("Laser drift (per pixel)", self.jitter)

        self.conserve = QtWidgets.QCheckBox(
            "Force identical total peak area in every pixel")
        self.conserve.setChecked(s.conserve_area)
        self.conserve.setToolTip(
            "On: a mixed pixel carries exactly the same area as a pure one,\n"
            "regardless of which materials it contains.\n"
            "Off: brighter materials genuinely return more signal.")
        f.addRow("", self.conserve)
        return w

    def _build_instrument_tab(self, inst):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)

        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(PRESETS))
        apply_btn = QtWidgets.QPushButton("Apply preset")
        apply_btn.clicked.connect(self._apply_preset)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.preset, 1)
        row.addWidget(apply_btn)
        holder = QtWidgets.QWidget()
        holder.setLayout(row)
        f.addRow("Preset", holder)

        self.n_channels = _int_spin(inst.n_channels, 32, 4096, " ch")
        self.f_start = _spin(inst.start_frequency_ghz, -200.0, 0.0, 3, " GHz")
        self.f_end = _spin(inst.end_frequency_ghz, 0.0, 200.0, 3, " GHz")
        f.addRow("Spectral channels", self.n_channels)
        f.addRow("Axis start", self.f_start)
        f.addRow("Axis end", self.f_end)

        self.axis_note = QtWidgets.QLabel("")
        self.axis_note.setWordWrap(True)
        self.axis_note.setStyleSheet("color:#8892a4; font-size:11px;")
        f.addRow("", self.axis_note)

        self.m1 = _spin(inst.marker1_ghz, 0.0, 200.0, 3, " GHz")
        self.m2 = _spin(inst.marker2_ghz, 0.0, 200.0, 3, " GHz")
        f.addRow("Fit window start", self.m1)
        f.addRow("Fit window end", self.m2)

        self.wavelength = _spin(inst.wavelength_nm, 200.0, 3000.0, 1, " nm")
        self.wavelength.setToolTip(
            "Used only by the material shift calculator (Eq. 1.1).\n"
            "Around 720 nm reproduces the ~5.51 GHz water peak in this\n"
            "project's data for n = 1.333 and v_a = 1483 m/s.")
        f.addRow("Laser wavelength", self.wavelength)

        self.gauss = _spin(inst.spectrometer_gaussian_linewidth_ghz, 0.0, 5.0, 3, " GHz")
        self.gauss.setToolTip(
            "Gaussian part of the spectrometer response. Real peaks here fall\n"
            "off faster than a Lorentzian; this is what reproduces that.")
        f.addRow("Spectrometer Gaussian (FWHM)", self.gauss)

        self.photon_rate = _spin(inst.photon_rate, 1.0, 1e6, 1, " cts/s")
        f.addRow("Brillouin photon rate", self.photon_rate)
        self.elastic = _spin(inst.elastic_peak_counts, 0.0, 1e6, 0, " cts")
        f.addRow("Elastic peak height", self.elastic)
        self.dark = _spin(inst.dark_rate, 0.0, 1000.0, 3, " cts/s")
        f.addRow("Dark rate", self.dark)
        self.read_noise = _spin(inst.read_noise_counts, 0.0, 100.0, 2, " cts")
        f.addRow("Read noise", self.read_noise)
        self.excess = _spin(inst.excess_noise_factor, 1.0, 3.0, 3, "")
        self.excess.setToolTip("1.0 for photon counting, ~1.414 for EMCCD.")
        f.addRow("Excess noise factor", self.excess)
        return w

    # -- wiring --------------------------------------------------------------

    def _wire(self):
        for widget in (self.grid_w, self.grid_h, self.step_um, self.exposure,
                       self.psf, self.temperature, self.jitter,
                       self.speckle_modes):
            widget.valueChanged.connect(self._emit_settings)
        for widget in (self.conserve, self.reference_fit):
            widget.toggled.connect(self._emit_settings)
        self.speckle_auto.toggled.connect(self._on_speckle_auto)

        for widget in (self.n_channels, self.f_start, self.f_end, self.m1, self.m2,
                       self.wavelength, self.gauss, self.photon_rate, self.elastic,
                       self.dark, self.read_noise, self.excess):
            widget.valueChanged.connect(self._emit_instrument)

        self._emit_settings()
        self._emit_instrument()

    def _on_speckle_auto(self, checked):
        self.speckle_modes.setEnabled(not checked)
        self._emit_settings()

    def _apply_preset(self):
        preset = PRESETS[self.preset.currentText()]()
        self.n_channels.setValue(preset.n_channels)
        self.f_start.setValue(preset.start_frequency_ghz)
        self.f_end.setValue(preset.end_frequency_ghz)
        self.m1.setValue(preset.marker1_ghz)
        self.m2.setValue(preset.marker2_ghz)

    def _emit_settings(self):
        s = self.settings()
        self.span.setText(f"{s.span_um[0]:g} x {s.span_um[1]:g} um "
                          f"({s.n_pixels} pixels)")
        modes = physics.speckle_mode_count(
            s.exposure_s, 0.65) if self.speckle_auto.isChecked() else self.speckle_modes.value()
        rel = 100.0 / max(modes, 1.0) ** 0.5
        self.settings_changed.emit()

    def _emit_instrument(self):
        self.apply_to_instrument(self.instrument)
        problems = self.instrument.validate()
        axis = self.instrument.frequency_axis()
        if problems:
            text = "PROBLEM: " + " ".join(problems)
            self.axis_note.setStyleSheet("color:#e06c5f; font-size:11px;")
        else:
            self.axis_note.setStyleSheet("color:#8892a4; font-size:11px;")
        self.instrument_changed.emit()

    # -- accessors -----------------------------------------------------------

    def settings(self):
        return ScanSettings(
            grid_w=self.grid_w.value(), grid_h=self.grid_h.value(),
            step_um=self.step_um.value(), exposure_s=self.exposure.value(),
            temperature_k=self.temperature.value(), psf_fwhm_um=self.psf.value(),
            conserve_area=self.conserve.isChecked(),
            speckle_modes=None if self.speckle_auto.isChecked() else self.speckle_modes.value(),
            bfs_jitter_mhz=self.jitter.value(), seed=None,
            reference_fit=self.reference_fit.isChecked(),
        )

    def apply_to_instrument(self, inst):
        inst.n_channels = self.n_channels.value()
        inst.start_frequency_ghz = self.f_start.value()
        inst.end_frequency_ghz = self.f_end.value()
        inst.marker1_ghz = self.m1.value()
        inst.marker2_ghz = self.m2.value()
        inst.wavelength_nm = self.wavelength.value()
        inst.spectrometer_gaussian_linewidth_ghz = self.gauss.value()
        inst.photon_rate = self.photon_rate.value()
        inst.elastic_peak_counts = self.elastic.value()
        inst.dark_rate = self.dark.value()
        inst.read_noise_counts = self.read_noise.value()
        inst.excess_noise_factor = self.excess.value()
        return inst
