"""Live preview: what the scan will look like, and one pixel's spectrum.

Runs a small, fast preview rather than a full scan -- no per-pixel fitting, so
it stays interactive while the canvas is being painted.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from ..core import noise, raster, spectrum

#: Preview noise is deterministic so the plot is stable while painting.
PREVIEW_SEED = 20240101


class PreviewPanel(QtWidgets.QWidget):
    """Two plots: the Brillouin shift map, and the spectrum of one pixel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(4.2, 5.4), layout="constrained")
        self.figure.patch.set_facecolor("#141821")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax_map = self.figure.add_subplot(2, 1, 1)
        self.ax_spec = self.figure.add_subplot(2, 1, 2)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["True Brillouin shift", "Dominant material",
                            "Pixel mixing", "Peak area"])
        self.mode.currentIndexChanged.connect(lambda _: self.refresh())

        self.pixel_label = QtWidgets.QLabel("Click the map to inspect a pixel")
        self.pixel_label.setStyleSheet("color:#8892a4; font-size:11px;")
        self.pixel_label.setWordWrap(True)

        self.log_scale = QtWidgets.QCheckBox("Log intensity")
        self.log_scale.setChecked(True)
        self.log_scale.toggled.connect(lambda _: self._draw_spectrum())

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Map"))
        top.addWidget(self.mode, 1)
        top.addWidget(self.log_scale)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.pixel_label)

        self._state = None
        self._selected = 0
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._style_axes()

    def _style_axes(self):
        for ax in (self.ax_map, self.ax_spec):
            ax.set_facecolor("#1b2029")
            ax.tick_params(colors="#8892a4", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#3d4657")
            ax.xaxis.label.set_color("#8892a4")
            ax.yaxis.label.set_color("#8892a4")
            ax.title.set_color("#c8d0dc")

    # -- data ----------------------------------------------------------------

    def update_preview(self, drawing, library, instrument, settings):
        """Recompute the preview.  Cheap: composition plus one noisy spectrum set."""
        if drawing is None or not len(library):
            return
        materials = list(library)
        gw, gh = settings.grid_w, settings.grid_h

        fractions = raster.fractions_from_drawing(
            drawing, len(materials), gw, gh, psf_fwhm_um=settings.psf_fwhm_um)
        truth = spectrum.truth_parameters(
            fractions, materials, instrument, settings.exposure_s,
            settings.temperature_k, settings.conserve_area)

        # Fixed seed on purpose: the preview redraws after every brush
        # stroke, and re-rolling the noise each time would make the
        # spectrum flicker for no informational gain.
        rng = np.random.default_rng(PREVIEW_SEED)
        density = spectrum.compose_densities(
            fractions, materials, instrument, settings.exposure_s,
            settings.temperature_k, settings.conserve_area)
        counts = noise.apply_detector(density, settings.exposure_s, instrument, rng)

        self._state = dict(fractions=fractions, truth=truth, counts=counts,
                           density=density, instrument=instrument,
                           settings=settings, names=list(library.names))
        self._selected = min(self._selected, gw * gh - 1)
        self.refresh()

    def refresh(self):
        if self._state is None:
            return
        self._draw_map()
        self._draw_spectrum()
        self.canvas.draw_idle()

    def _draw_map(self):
        st = self._state
        s = st["settings"]
        gw, gh = s.grid_w, s.grid_h
        mode = self.mode.currentText()

        if mode == "True Brillouin shift":
            data = st["truth"]["bfs_ghz"].reshape(gh, gw)
            cmap, label = "viridis", "GHz"
        elif mode == "Dominant material":
            data = st["truth"]["dominant"].reshape(gh, gw).astype(float)
            cmap, label = "tab10", "index"
        elif mode == "Pixel mixing":
            data = (1.0 - st["truth"]["purity"]).reshape(gh, gw)
            cmap, label = "magma", "1 - purity"
        else:
            data = st["truth"]["area"].reshape(gh, gw)
            cmap, label = "inferno", "counts*GHz"

        self.ax_map.clear()
        im = self.ax_map.imshow(data, origin="lower", cmap=cmap,
                                interpolation="nearest", aspect="equal")
        self.ax_map.set_title(f"{mode}   ({gw} x {gh} px)", fontsize=9)
        y, x = divmod(self._selected, gw)
        self.ax_map.plot([x], [y], marker="s", mfc="none", mec="#ffd166",
                         ms=7, mew=1.4)
        if getattr(self, "_cbar", None) is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
        self._cbar = self.figure.colorbar(im, ax=self.ax_map, fraction=0.046)
        self._cbar.ax.tick_params(colors="#8892a4", labelsize=7)
        self._cbar.set_label(label, color="#8892a4", fontsize=8)
        self._style_axes()

    def _draw_spectrum(self):
        if self._state is None:
            return
        st = self._state
        inst = st["instrument"]
        axis = inst.frequency_axis()
        i = self._selected
        counts = st["counts"][i]
        density = st["density"][i]

        self.ax_spec.clear()
        self.ax_spec.plot(axis, np.maximum(counts, 0.3), lw=0.8, color="#4a90c2",
                          label="simulated counts")
        self.ax_spec.plot(axis, np.maximum(density, 0.3), lw=1.0, color="#ffd166",
                          label="noise-free")
        self.ax_spec.axvspan(inst.marker1_ghz, inst.marker2_ghz, color="#ffffff",
                             alpha=0.06, label="fit window")
        if self.log_scale.isChecked():
            self.ax_spec.set_yscale("log")
        self.ax_spec.set_xlabel("Frequency shift (GHz)", fontsize=8)
        self.ax_spec.set_ylabel("Counts", fontsize=8)
        self.ax_spec.legend(fontsize=7, facecolor="#1b2029", edgecolor="#3d4657",
                            labelcolor="#c8d0dc")
        self._style_axes()

        frac = st["fractions"][i]
        parts = ", ".join(f"{st['names'][j]} {100 * frac[j]:.0f}%"
                          for j in np.argsort(frac)[::-1] if frac[j] > 0.005)
        gw = st["settings"].grid_w
        y, x = divmod(i, gw)
        self.pixel_label.setText(
            f"Pixel ({x}, {y}) - {parts or 'empty'}\n"
            f"true shift {st['truth']['bfs_ghz'][i]:.4f} GHz, "
            f"true linewidth {st['truth']['linewidth_ghz'][i]:.4f} GHz, "
            f"peak {counts[inst.marker_mask()].max():.0f} counts")

    def _on_click(self, event):
        if self._state is None or event.inaxes is not self.ax_map:
            return
        s = self._state["settings"]
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if 0 <= x < s.grid_w and 0 <= y < s.grid_h:
            self._selected = y * s.grid_w + x
            self.refresh()
