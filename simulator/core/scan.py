"""Running a whole simulated scan.

Ties the painted drawing, the material library and the instrument together and
produces the integer count spectra that go into a CSV, plus the noise-free
ground truth that goes into the sidecar.

The fitted columns are produced by genuinely fitting the noisy spectra with the
project's own fitter, so a generated file contains exactly the information a
real one does and nothing more.  Ground truth is kept strictly separate.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict, field, replace

import numpy as np

from . import noise, physics, raster, spectrum

#: Repository root, so the project's own analysis modules can be imported.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import brillouin_fitting as bf  # noqa: E402


@dataclass
class ScanSettings:
    """Everything about how a scan is acquired, as opposed to what is scanned.

    Attributes
    ----------
    grid_w, grid_h : int
        Scan grid in pixels.  These become the ``X`` and ``Y`` columns.
    step_um : float
        Stage step between adjacent pixels.  The CSV stores step *indices*, not
        microns -- downstream code multiplies by its own ``STEP_UM`` constant,
        which is 5.0 in ``analysis/line_scans/carboplatin_line_scans.py``.
    exposure_s : float
        Dwell time per pixel.  The main noise knob: peak area scales with it,
        so shot-noise-limited precision improves as its square root.
    temperature_k : float
        Sample temperature.  Feeds thesis Eq. (3.12), so the Brillouin signal
        scales linearly with it.
    psf_fwhm_um : float
        Confocal spot size.  Blurs material composition before downsampling.
    conserve_area : bool
        Force every pixel to carry identical total peak area regardless of what
        it is made of.  True is the strict "mix the Lorentzians, keep the area"
        reading; false lets brighter materials be brighter.
    speckle_modes : float or None
        Override for the number of averaged thermal realisations.  ``None``
        derives it from exposure and linewidth, which at realistic exposures
        makes thermal speckle negligible next to shot noise.
    bfs_jitter_mhz : float
        Standard deviation of a slow per-pixel shift of the whole spectrum,
        modelling laser frequency drift.
    seed : int or None
        Random seed.  ``None`` -- the default -- draws a fresh one from system
        entropy for every run, so you get different noise each time without
        having to manage it.  Whatever was used is resolved onto the returned
        result and written into the manifest and the truth sidecar, so any run
        can still be reproduced exactly by passing that number back in.
    reference_fit : bool
        Also fit the *noise-free* spectrum of every pixel and record the
        result as ground truth.  Worth the doubled fitting time: the analytic
        truth says what the materials are, but a single Lorentzian fitted to a
        Voigt-shaped, multi-material peak cannot return exactly that even with
        infinite signal.  Comparing a noisy fit against the noise-free fit
        isolates the error caused by *noise* from the error caused by the
        model, which is the distinction a noise-tolerance study needs.
    fit_max_iter : int
        Maximum nonlinear-fit evaluations per attempt. Extreme
        photon-starved stress tests may lower this so unconstrained fits fail
        quickly instead of spending minutes chasing noise.
    """

    grid_w: int = 40
    grid_h: int = 40
    step_um: float = 1.0
    exposure_s: float = 10.0
    temperature_k: float = 300.0
    psf_fwhm_um: float = 0.6
    conserve_area: bool = False
    speckle_modes: float | None = None
    bfs_jitter_mhz: float = 0.0
    seed: int | None = None
    reference_fit: bool = True
    fit_max_iter: int = 2000

    @property
    def n_pixels(self):
        return int(self.grid_w) * int(self.grid_h)

    @property
    def span_um(self):
        """Physical extent covered by the scan, (width, height) in microns."""
        return (self.grid_w * self.step_um, self.grid_h * self.step_um)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class ScanResult:
    """Output of :func:`simulate_scan`."""

    counts: np.ndarray            # (n_pixels, n_channels) int32
    fitted_curves: np.ndarray     # (n_pixels, n_channels) float, zero outside markers
    fit_params: np.ndarray        # (n_pixels, 4) [area, centre, linewidth, base]
    fractions: np.ndarray         # (n_pixels, n_materials)
    truth: dict                   # noise-free ground truth per pixel
    x_index: np.ndarray           # (n_pixels,) int
    y_index: np.ndarray           # (n_pixels,) int
    settings: ScanSettings
    material_names: list

    @property
    def n_pixels(self):
        return int(self.counts.shape[0])

    def bfs_map(self):
        """Fitted Brillouin shift as an image, for previewing."""
        return self.fit_params[:, 1].reshape(self.settings.grid_h, self.settings.grid_w)

    def truth_bfs_map(self):
        return self.truth["bfs_ghz"].reshape(self.settings.grid_h, self.settings.grid_w)

    def bfs_error_mhz(self):
        """Fitted minus true shift, in MHz, per pixel.

        This is *total* error: noise plus whatever a single Lorentzian cannot
        represent about a Voigt-shaped, possibly multi-material peak.
        """
        return (self.fit_params[:, 1] - self.truth["bfs_ghz"]) * 1000.0

    def bfs_noise_error_mhz(self):
        """Fitted shift minus the noise-free fit, in MHz, per pixel.

        The part of the error that noise is actually responsible for.  Falls
        back to :meth:`bfs_error_mhz` when no reference fit was run.
        """
        ref = self.truth.get("noisefree_bfs_ghz")
        if ref is None:
            return self.bfs_error_mhz()
        return (self.fit_params[:, 1] - ref) * 1000.0


def resolve_seed(settings):
    """Return settings with a concrete seed, drawing one if none was given.

    Recording the seed matters more than choosing it: runs should differ by
    default, but a run you cannot reproduce is a run you cannot debug.
    """
    if settings.seed is not None:
        return settings
    drawn = int(np.random.SeedSequence().generate_state(1)[0])
    return replace(settings, seed=drawn)


def scan_grid_indices(grid_w, grid_h):
    """X and Y step indices in CSV row order: X fastest, then Y.

    ``brillouin_preprocessing.reshape_to_map`` does a bare
    ``reshape(len(y_vals), len(x_vals))``, so getting this order wrong silently
    transposes every map downstream.
    """
    ys, xs = np.meshgrid(np.arange(int(grid_h)), np.arange(int(grid_w)),
                         indexing="ij")
    return xs.ravel().astype(int), ys.ravel().astype(int)


def simulate_scan(drawing, library, instrument, settings, progress=None):
    """Simulate a full scan of a painted drawing.

    Parameters
    ----------
    drawing : CellDrawing
    library : MaterialLibrary
    instrument : Instrument
    settings : ScanSettings
    progress : callable, optional
        Called as ``progress(done, total, message)`` during fitting, which is
        the slow part.  Returning ``False`` cancels the run.

    Returns
    -------
    ScanResult or None
        ``None`` if the caller cancelled.
    """
    settings = resolve_seed(settings)
    rng = np.random.default_rng(settings.seed)
    materials = list(library)

    fractions = raster.fractions_from_drawing(
        drawing, len(materials), settings.grid_w, settings.grid_h,
        psf_fwhm_um=settings.psf_fwhm_um,
    )

    jitter = np.zeros(settings.n_pixels)
    if settings.bfs_jitter_mhz > 0:
        jitter = rng.standard_normal(settings.n_pixels) * (settings.bfs_jitter_mhz / 1000.0)

    density = spectrum.compose_densities(
        fractions, materials, instrument,
        exposure_s=settings.exposure_s,
        temperature_k=settings.temperature_k,
        conserve_area=settings.conserve_area,
        bfs_offsets=jitter,
    )

    mean_linewidth = float(np.mean(
        [spectrum.observed_linewidth(m, instrument) for m in materials]))
    modes = noise.effective_speckle_modes(
        settings.exposure_s, mean_linewidth, settings.speckle_modes)
    density = noise.apply_thermal_speckle(density, modes, rng)

    counts = noise.apply_detector(density, settings.exposure_s, instrument, rng)

    truth = spectrum.truth_parameters(
        fractions, materials, instrument,
        exposure_s=settings.exposure_s,
        temperature_k=settings.temperature_k,
        conserve_area=settings.conserve_area,
    )
    truth["bfs_ghz"] = truth["bfs_ghz"] + jitter

    if settings.reference_fit:
        _, ref_params = fit_scan(
            density, instrument, progress=progress,
            message="Fitting noise-free reference",
            max_iter=settings.fit_max_iter)
        if ref_params is None:
            return None
        truth["noisefree_area"] = ref_params[:, 0]
        truth["noisefree_bfs_ghz"] = ref_params[:, 1]
        truth["noisefree_linewidth_ghz"] = ref_params[:, 2]
        truth["noisefree_base"] = ref_params[:, 3]

    fitted_curves, fit_params = fit_scan(
        counts, instrument, progress=progress,
        max_iter=settings.fit_max_iter)
    if fitted_curves is None:
        return None

    xs, ys = scan_grid_indices(settings.grid_w, settings.grid_h)
    return ScanResult(
        counts=counts, fitted_curves=fitted_curves, fit_params=fit_params,
        fractions=fractions, truth=truth, x_index=xs, y_index=ys,
        settings=settings, material_names=list(library.names),
    )


def fit_scan(counts, instrument, progress=None, message="Fitting spectra",
             max_iter=2000):
    """Fit every pixel's anti-Stokes peak with the project's own Lorentzian.

    Uses ``brillouin_fitting.fit_lorentzian_iterative`` directly rather than
    ``fit_all_spectra``, because the latter passes ``max_iter=50`` straight into
    ``curve_fit``'s ``maxfev`` and silently falls back to the initial guess on
    noisy spectra.

    Returns
    -------
    fitted_curves : ndarray (n_pixels, n_channels)
        The fitted Lorentzian inside the marker window, zero outside -- the
        same convention real files use.
    fit_params : ndarray (n_pixels, 4)
        ``[area, centre_ghz, linewidth_ghz, base]``.  ``area`` is in the CSV's
        ``Indensity`` convention, i.e. ``amp * pi * hwhm``.  ``linewidth_ghz``
        is a full width at half maximum; the CSV's ``Half Width`` column gets
        half of it.
    """
    axis = instrument.frequency_axis()
    mask = instrument.marker_mask()
    win_freq = axis[mask]

    n_pixels = counts.shape[0]
    fitted_curves = np.zeros((n_pixels, instrument.n_channels), dtype=float)
    fit_params = np.zeros((n_pixels, 4), dtype=float)

    for i in range(n_pixels):
        y = counts[i, mask].astype(float)
        p0, _ = bf.initial_lorentzian_guess(y, win_freq)
        curve, popt = bf.fit_lorentzian_iterative(
            win_freq, y, params=p0, max_iter=int(max_iter))
        amp, centre, gamma, base = popt
        fitted_curves[i, mask] = curve
        # gamma from the fitter is a half width; report the full width so every
        # user-facing number in this package is a linewidth.
        fit_params[i] = (physics.amplitude_to_area(amp, gamma), centre,
                         2.0 * gamma, base)

        if progress is not None and (i % 64 == 0 or i == n_pixels - 1):
            if progress(i + 1, n_pixels, message) is False:
                return None, None

    return fitted_curves, fit_params
