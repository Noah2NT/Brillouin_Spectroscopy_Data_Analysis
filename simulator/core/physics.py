"""Brillouin line shapes and the optical relations behind them.

Equation numbers refer to Oscar Nieves, *Noise in Stimulated Brillouin
Scattering* (PhD thesis, University of Technology Sydney, 2021).

Two Lorentzian parameterisations appear in this project and it matters which is
which:

* **Area form** -- what the instrument writes into the CSV.
  ``Fitting Data = Indensity/pi * HW / ((f - Frequency)**2 + HW**2) + Base``
  so ``Indensity`` is the integrated area and ``Half Width`` is the HWHM.
* **Amplitude form** -- what ``brillouin_fitting.lorentzian`` uses:
  ``amp / (1 + ((f - mean)/gamma)**2) + c``.

The two are related by ``area = amp * pi * gamma``.  Mixing them up silently
rescales every peak by a factor of ~pi*gamma, so always convert explicitly with
:func:`amplitude_to_area` / :func:`area_to_amplitude`.
"""

from __future__ import annotations

import numpy as np

#: Boltzmann constant, J/K.
K_B = 1.380649e-23

#: Speed of light in vacuum, m/s.
C_LIGHT = 299_792_458.0


def lorentzian_unit_area(freq, center, hwhm):
    """Lorentzian normalised to unit area.

    ``integral(lorentzian_unit_area(f, v, g) df) == 1`` over the whole real
    line.  This is the building block for material mixing: scaling each
    material's line by its area fraction and summing gives a spectrum whose
    total area is independent of how the pixel is divided up.

    Parameters
    ----------
    freq : ndarray
        Frequency axis, GHz.
    center : float
        Line centre, GHz.
    hwhm : float
        Half width at half maximum, GHz.

    Returns
    -------
    ndarray
        Line shape, units of 1/GHz.
    """
    hwhm = max(float(hwhm), 1e-6)
    return (1.0 / np.pi) * hwhm / ((np.asarray(freq, dtype=float) - center) ** 2 + hwhm ** 2)


def lorentzian_amplitude(freq, amp, center, hwhm, base=0.0):
    """Lorentzian in the amplitude parameterisation used by ``brillouin_fitting``."""
    hwhm = max(float(hwhm), 1e-6)
    return amp / (1.0 + ((np.asarray(freq, dtype=float) - center) / hwhm) ** 2) + base


def lorentzian_area_form(freq, area, center, hwhm, base=0.0):
    """Lorentzian in the area parameterisation the instrument CSV uses."""
    return area * lorentzian_unit_area(freq, center, hwhm) + base


def amplitude_to_area(amp, hwhm):
    """Convert an amplitude-form peak height to the CSV's ``Indensity`` column."""
    return float(amp) * np.pi * float(hwhm)


def area_to_amplitude(area, hwhm):
    """Convert the CSV's ``Indensity`` column to an amplitude-form peak height."""
    hwhm = max(float(hwhm), 1e-6)
    return float(area) / (np.pi * hwhm)


def bfs_from_optics(refractive_index, acoustic_velocity, wavelength_nm,
                    scattering_angle_deg=180.0):
    """Brillouin frequency shift from material and geometry -- thesis Eq. (1.1).

    ``nu_B = 2 * n * v_a * sin(theta / 2) / lambda``

    Parameters
    ----------
    refractive_index : float
        Refractive index ``n`` of the medium.
    acoustic_velocity : float
        Acoustic (phonon) group velocity ``v_a``, m/s.
    wavelength_nm : float
        Vacuum wavelength of the probe laser, nm.
    scattering_angle_deg : float
        Scattering angle ``theta``, degrees.  180 is backscattering, which is
        what a confocal Brillouin microscope does.

    Returns
    -------
    float
        Brillouin frequency shift, GHz.
    """
    theta = np.deg2rad(float(scattering_angle_deg))
    lam_m = float(wavelength_nm) * 1e-9
    shift_hz = 2.0 * float(refractive_index) * float(acoustic_velocity) * np.sin(theta / 2.0) / lam_m
    return shift_hz / 1e9


def wavelength_for_bfs(bfs_ghz, refractive_index, acoustic_velocity,
                       scattering_angle_deg=180.0):
    """Invert Eq. (1.1) to find the wavelength implied by an observed shift.

    Useful for calibrating the instrument preset against a known material: the
    water peak measured in this project sits near 5.51 GHz, which for
    ``n = 1.333`` and ``v_a = 1483 m/s`` implies a wavelength near 720 nm.
    """
    theta = np.deg2rad(float(scattering_angle_deg))
    lam_m = (2.0 * float(refractive_index) * float(acoustic_velocity)
             * np.sin(theta / 2.0) / (float(bfs_ghz) * 1e9))
    return lam_m * 1e9


def acoustic_linewidth(acoustic_velocity, acoustic_attenuation):
    """Brillouin FWHM from the acoustic attenuation -- thesis Sec. 3.3.

    The thesis writes ``Gamma = v_a * alpha_ac`` (rad/s), and shows that the
    thermal acoustic background has a Lorentzian power spectrum of FWHM
    ``Gamma``.  Returned here in GHz for convenience.
    """
    gamma_rad_s = float(acoustic_velocity) * float(acoustic_attenuation)
    return gamma_rad_s / (2.0 * np.pi) / 1e9


def attenuation_for_linewidth(fwhm_ghz, acoustic_velocity):
    """Invert :func:`acoustic_linewidth` to get ``alpha_ac`` in 1/m."""
    gamma_rad_s = float(fwhm_ghz) * 1e9 * 2.0 * np.pi
    return gamma_rad_s / max(float(acoustic_velocity), 1e-9)


def combine_widths(*widths):
    """Total width of a cascade of Lorentzian broadenings.

    Works in whatever convention you feed it -- widths add either way, so
    passing FWHMs returns an FWHM and passing HWHMs returns an HWHM.

    Convolving Lorentzians adds their widths, so the intrinsic Brillouin line,
    the laser's own Lorentzian line shape (thesis Eq. 3.20) and the
    spectrometer response combine by simple addition -- no numerical
    convolution needed.
    """
    return float(sum(max(float(w), 0.0) for w in widths))


def thermal_noise_strength(temperature_k, acoustic_attenuation):
    """Thermal noise strength ``sigma = k_B * T * alpha_ac`` -- thesis Eq. (3.12).

    This is the fluctuation-dissipation result: the strength of the driving
    noise on the acoustic field is set by the temperature and the acoustic
    loss.  Because the steady-state acoustic power is proportional to
    ``sigma / alpha_ac = k_B * T``, the *scattered signal* scales linearly with
    temperature, which is how the temperature control in this simulator earns
    its keep.

    Returns
    -------
    float
        Noise strength in J/m.
    """
    return K_B * float(temperature_k) * float(acoustic_attenuation)


def coherence_time(fwhm_ghz):
    """Coherence time of a Lorentzian line of the given FWHM, in seconds."""
    fwhm_hz = max(float(fwhm_ghz), 1e-12) * 1e9
    return 1.0 / (np.pi * fwhm_hz)


def speckle_mode_count(exposure_s, fwhm_ghz):
    """Number of independent thermal realisations averaged in one exposure.

    A detector integrating for ``exposure_s`` averages roughly
    ``exposure_s / tau_coh`` independent samples of the Ornstein-Uhlenbeck
    thermal field (thesis Sec. 3.3), so the relative fluctuation of the
    resulting spectral estimate is ``1/sqrt(M)``.

    For a 0.6 GHz line and a 1 s exposure this is of order 1e12, i.e. thermal
    speckle is utterly negligible next to photon shot noise.  That is the
    physically correct answer, and it is why :mod:`simulator.core.noise` treats
    shot noise as the dominant term.
    """
    return float(exposure_s) / coherence_time(fwhm_ghz)
