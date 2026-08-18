"""Spectrometer and detector description.

Defaults reproduce the acquisition settings of every 2D scan in ``data/csv``:
512 spectral channels spanning -14.9 to +14.9 GHz, with the elastic line at the
centre of the axis.

One geometric detail drives several decisions here.  ``np.linspace(-14.9, 14.9,
512)`` has an *even* number of points, so no channel lands exactly on zero --
the two central channels sit at -0.029159 and +0.029159 GHz.  Real files show
the same thing (their Rayleigh maximum is at index 256, f = +0.0292), so the
simulator reproduces it rather than idealising it away.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

import numpy as np


@dataclass
class Instrument:
    """Spectrometer axis, optical parameters and detector chain.

    Attributes
    ----------
    n_channels : int
        Spectral channels per pixel.  Every file in this project uses 512;
        one paclitaxel file uses 1024.
    start_frequency_ghz, end_frequency_ghz : float
        Axis limits written to the CSV.  Must be symmetric about zero or
        ``brillouin_preprocessing.fold_brillouin_spectrum`` will misalign the
        fold.
    marker1_ghz, marker2_ghz : float
        Fit window on the anti-Stokes side, written to the CSV and used for the
        fitted columns.  Real files carry values like 2.81 / 8.48.
    wavelength_nm, scattering_angle_deg : float
        Probe geometry, used only by the optional Eq. (1.1) calculator.
    laser_linewidth_mhz : float
        Intrinsic laser linewidth (FWHM).  Thesis Eq. (3.20) gives the laser a
        Lorentzian spectrum of this width, which adds directly to the observed
        Brillouin linewidth.
    spectrometer_linewidth_ghz : float
        Lorentzian part of the instrument response (FWHM), which adds to the
        observed linewidth.
    spectrometer_gaussian_linewidth_ghz : float
        Gaussian part of the instrument response, applied by convolution.  Real
        spectra in this project are not pure Lorentzians -- their wings fall
        off faster, which is why the vendor's fitted ``Base`` column is
        typically around -30 counts rather than positive.  Reproducing that
        matters, because a pure-Lorentzian sample would be subtly easier to
        fit than real data.
    free_spectral_range_ghz : float
        VIPA free spectral range.  Real spectra in this project show extra
        peaks near +-10.5 GHz whose separation from the elastic line is
        ``FSR - BFS``; with a 5.5 GHz shift that implies an FSR near 16 GHz,
        which is the default.  The neighbouring-order *elastic* lines then fall
        at +-16 GHz, outside the +-14.9 GHz window, which is why they are not
        seen and why ``adjacent_order_fraction`` defaults to zero.
    include_adjacent_orders : bool
        Master switch for both kinds of neighbouring-order line.
    adjacent_brillouin_fraction : float
        Strength of the neighbouring-order Brillouin lines at
        ``+-(FSR - BFS)``, relative to the main doublet.  Defaults to zero.

        Real scans here do show extra peaks near +-10.5 GHz, and a
        neighbouring order was the first explanation tried.  It does not hold
        up well: it needs an FSR of 15.98 GHz that nothing independently
        supports, and in ``water_plexiglass.csv`` the analogous 11.75 GHz peak
        is provably a *material* (its intensity anti-correlates with the water
        peak at r = -0.96 and varies 563-fold across the field, whereas a
        diffraction order would track the elastic line).  Polystyrene lands at
        10.4 GHz at this wavelength and those samples are cultured in plastic
        plates, so the peak is better modelled as a substrate material -- see
        the "Polystyrene dish" entry in the material library.  The knob stays
        here in case a genuine order-overlap needs modelling.
    elastic_linewidth_ghz : float
        Lorentzian linewidth (FWHM) of the elastic (Rayleigh) line before the instrument
        response.  Kept very small on purpose: the elastic line is the
        spectrometer's response to essentially monochromatic light, so its
        observed shape should be dominated by the Gaussian response rather than
        by Lorentzian wings.  Making it Lorentzian instead floods the Brillouin
        window with tail counts that real data does not have.
    elastic_peak_counts : float
        Height of the central elastic line *after* the instrument response,
        counts.  Real data: 20k-60k.
    adjacent_order_fraction : float
        Height of the neighbouring-order elastic lines relative to the central
        one.  Real data ranges from about 0.01 to 0.25.
    photon_rate : float
        Brillouin counts per second at unit efficiency and unit temperature
        scaling.  Combined with exposure this sets the peak area, and hence
        the ``Indensity`` column.
    dark_rate : float
        Dark counts per channel per second.
    read_noise_counts : float
        Gaussian read noise standard deviation, counts.
    excess_noise_factor : float
        EMCCD multiplicative excess noise.  1.0 for a photon-counting detector,
        sqrt(2) ~ 1.414 for an electron-multiplying one.
    baseline_offset : float
        Constant offset added before digitisation.  Real files show slightly
        negative fitted baselines, which comes out of the fit rather than being
        put in here.
    """

    n_channels: int = 512
    start_frequency_ghz: float = -14.9
    end_frequency_ghz: float = 14.9
    marker1_ghz: float = 2.813950378
    marker2_ghz: float = 8.475348432

    wavelength_nm: float = 720.0
    scattering_angle_deg: float = 180.0
    laser_linewidth_mhz: float = 0.1
    spectrometer_linewidth_ghz: float = 0.04
    spectrometer_gaussian_linewidth_ghz: float = 0.40

    free_spectral_range_ghz: float = 15.98
    include_adjacent_orders: bool = True
    elastic_linewidth_ghz: float = 0.030
    elastic_peak_counts: float = 30000.0
    adjacent_order_fraction: float = 0.0
    adjacent_brillouin_fraction: float = 0.0

    photon_rate: float = 1900.0
    dark_rate: float = 0.12
    read_noise_counts: float = 0.8
    excess_noise_factor: float = 1.0
    baseline_offset: float = 0.0

    # -- derived quantities --------------------------------------------------

    def frequency_axis(self):
        """The axis exactly as downstream code will reconstruct it.

        Mirrors ``brillouin_preprocessing.load_brillouin_csv``, which does
        ``np.linspace(startFrequency, endFrequency, n_channels)``.
        """
        return np.linspace(self.start_frequency_ghz, self.end_frequency_ghz,
                           self.n_channels)

    @property
    def bin_width_ghz(self):
        axis = self.frequency_axis()
        return float(axis[1] - axis[0])

    @property
    def laser_linewidth_ghz(self):
        """Laser contribution to the observed linewidth, thesis Eq. (3.20)."""
        return self.laser_linewidth_mhz / 1000.0

    def marker_mask(self):
        """Boolean mask selecting the anti-Stokes fit window."""
        axis = self.frequency_axis()
        return (axis >= self.marker1_ghz) & (axis <= self.marker2_ghz)

    def validate(self):
        """Return a list of problems that would make output incompatible."""
        problems = []
        if self.n_channels < 16:
            problems.append("Need at least 16 spectral channels.")
        centre = (self.start_frequency_ghz + self.end_frequency_ghz) / 2.0
        if abs(centre) > 1e-9:
            problems.append(
                f"Frequency axis must be symmetric about zero (centre is {centre:+.4f} GHz); "
                "folding in brillouin_preprocessing assumes the elastic line is at the middle."
            )
        if self.marker1_ghz >= self.marker2_ghz:
            problems.append("marker1 must be below marker2.")
        if self.marker2_ghz > self.end_frequency_ghz:
            problems.append("marker2 lies outside the frequency axis.")
        if self.photon_rate <= 0:
            problems.append("Photon rate must be positive.")
        return problems

    # -- persistence ---------------------------------------------------------

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "instrument": self.to_dict()}, fh, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data.get("instrument", data))


def preset_narrow():
    """The +-14.9 GHz preset every 2D scan in ``data/csv`` uses."""
    return Instrument()


def preset_wide():
    """The +-24.8 GHz preset used by the older line scans."""
    return Instrument(
        start_frequency_ghz=-24.8, end_frequency_ghz=24.8,
        marker1_ghz=3.0940117930757047, marker2_ghz=7.911313657407408,
        free_spectral_range_ghz=33.0,
    )


PRESETS = {
    "Narrow (+/-14.9 GHz, 512 ch)": preset_narrow,
    "Wide (+/-24.8 GHz, 512 ch)": preset_wide,
}
