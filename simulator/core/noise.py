"""Noise models.

The thermal and laser terms come from Oscar Nieves, *Noise in Stimulated
Brillouin Scattering* (2021), chapters 3 and 5.  The detector chain does not --
the thesis models stimulated Brillouin scattering in a waveguide, not a
photon-counting microscope, and the ``Raw Data`` column in this project's files
is small integer counts.  Reproducing that needs a shot-noise model, so one is
included here and clearly separated from the thesis physics.

How the pieces rank in practice
-------------------------------
The thesis shows the thermal acoustic background is an Ornstein-Uhlenbeck
process whose power spectrum is a Lorentzian of FWHM ``Gamma`` (Sec. 3.3).  A
detector integrating for ``t_exp`` averages about ``t_exp / tau_coh``
independent realisations of it.  For a 0.6 GHz line that coherence time is
around 0.5 ps, so a 1 second exposure averages of order 1e12 realisations and
the thermal fluctuation washes out to parts per million.

So thermal noise does not show up as visible scatter.  What it does control is
the *signal level* -- ``sigma = k_B * T * alpha_ac`` (Eq. 3.12) makes the peak
area proportional to temperature -- and the *linewidth*, through
``Gamma = v_a * alpha_ac``.  The visible pixel-to-pixel scatter in real data is
photon shot noise.  Both are modelled; the speckle term is exposed as a knob so
short-exposure or few-mode regimes can still be explored.
"""

from __future__ import annotations

import numpy as np

from . import physics

#: Above this many averaged thermal realisations the speckle term is dropped
#: entirely -- its relative size is below 1e-4 and it costs real time to draw.
_SPECKLE_NEGLIGIBLE_ABOVE = 1e8

#: Above this, a normal approximation to the Gamma distribution is exact enough
#: and far faster than sampling a Gamma with an enormous shape parameter.
_SPECKLE_NORMAL_ABOVE = 1e3


def apply_thermal_speckle(density, modes, rng):
    """Multiplicative thermal-light fluctuation on a spectral estimate.

    Periodogram bins of the Ornstein-Uhlenbeck thermal field (thesis Eq. 3.18)
    are exponentially distributed about the underlying Lorentzian.  Averaging
    ``modes`` independent realisations turns that into a Gamma distribution
    with shape ``modes`` and mean equal to the true density, i.e. a relative
    standard deviation of ``1/sqrt(modes)``.

    Parameters
    ----------
    density : ndarray
        Noise-free expected counts per channel.
    modes : float or None
        Number of independent realisations averaged.  ``None`` or a
        non-positive value disables the term.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray
        Fluctuated density, same shape.
    """
    density = np.asarray(density, dtype=float)
    if modes is None or modes <= 0 or modes > _SPECKLE_NEGLIGIBLE_ABOVE:
        return density
    if modes > _SPECKLE_NORMAL_ABOVE:
        # Gamma(k, theta) -> Normal(k*theta, sqrt(k)*theta) for large k.
        jitter = rng.standard_normal(density.shape) / np.sqrt(modes)
        return np.clip(density * (1.0 + jitter), 0.0, None)
    return rng.gamma(shape=modes, scale=density / modes)


def apply_detector(density, exposure_s, instrument, rng):
    """Turn an expected-counts spectrum into integer detector counts.

    Applied in the order a real camera does it: dark current accumulates with
    exposure and is itself Poisson, the signal is Poisson, electron
    multiplication inflates the variance, read noise is added at the amplifier,
    and the result is digitised and clipped at zero.

    Returns
    -------
    ndarray of int32
    """
    density = np.clip(np.asarray(density, dtype=float), 0.0, None)
    lam = density + max(instrument.dark_rate, 0.0) * float(exposure_s)
    lam = np.clip(lam + instrument.baseline_offset, 0.0, None)

    counts = rng.poisson(lam).astype(float)

    f = float(instrument.excess_noise_factor)
    if f > 1.0:
        # Electron multiplication preserves the mean but multiplies the
        # variance by F^2; add the shortfall as Gaussian.
        extra_var = np.clip((f * f - 1.0) * lam, 0.0, None)
        counts += rng.standard_normal(counts.shape) * np.sqrt(extra_var)

    if instrument.read_noise_counts > 0:
        counts += rng.standard_normal(counts.shape) * instrument.read_noise_counts

    return np.clip(np.rint(counts), 0, None).astype(np.int32)


def laser_phase_walk(n_steps, dt_s, linewidth_hz, rng):
    """Laser phase as a Brownian motion -- thesis Eqs. (3.23) and (3.27).

    ``phi(t) = sqrt(dOmega) * W(t)``, integrated by Euler-Maruyama as
    ``phi[n+1] = phi[n] + sqrt(dOmega) * dW[n]`` with
    ``dW ~ sqrt(dt) * N(0, 1)``.

    The time-averaged effect of this on a spectrum is simply a Lorentzian
    broadening of FWHM ``dOmega``, which :mod:`simulator.core.spectrum` applies
    analytically.  This function exists so that behaviour can be checked
    against an explicit realisation.

    Returns
    -------
    ndarray
        Phase in radians, length ``n_steps``.
    """
    d_omega = 2.0 * np.pi * float(linewidth_hz)
    increments = np.sqrt(d_omega * float(dt_s)) * rng.standard_normal(int(n_steps))
    phase = np.cumsum(increments)
    phase[0] = 0.0
    return phase


def ou_thermal_series(n_steps, dt_s, gamma_rad_s, sigma, acoustic_velocity,
                      slice_thickness_m, rng, initial=0.0):
    """Explicit Ornstein-Uhlenbeck synthesis of the thermal acoustic field.

    This is thesis Eq. (5.13) verbatim::

        D[n+1] = exp(-Gamma*dt/2) * D[n] + gamma(dt) * (N1 + i*N2)

    with the step weight from Eq. (5.14)::

        gamma(dt) = v_a * sqrt(sigma * (1 - exp(-Gamma*dt)) / (2 * dz * Gamma))

    Slow compared with the closed-form route in :func:`apply_thermal_speckle`,
    and used to prove that shortcut faithful rather than in the hot path.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    dt_s : float
        Time step, seconds.
    gamma_rad_s : float
        Acoustic decay rate ``Gamma = v_a * alpha_ac``, rad/s.
    sigma : float
        Thermal noise strength ``k_B * T * alpha_ac`` (Eq. 3.12).
    acoustic_velocity : float
        ``v_a``, m/s.
    slice_thickness_m : float
        ``dz``, the discretisation length from thesis Fig. 3.1.
    rng : numpy.random.Generator
    initial : complex

    Returns
    -------
    ndarray of complex128
    """
    n_steps = int(n_steps)
    decay = np.exp(-0.5 * gamma_rad_s * dt_s)
    step_weight = acoustic_velocity * np.sqrt(
        sigma * (1.0 - np.exp(-gamma_rad_s * dt_s))
        / (2.0 * slice_thickness_m * gamma_rad_s)
    )

    noise = (rng.standard_normal(n_steps) + 1j * rng.standard_normal(n_steps))
    out = np.empty(n_steps, dtype=complex)
    value = complex(initial)
    for n in range(n_steps):
        value = decay * value + step_weight * noise[n]
        out[n] = value
    return out


def ou_power_spectrum(n_steps, dt_s, gamma_rad_s, sigma, acoustic_velocity,
                      slice_thickness_m, rng, n_realisations=32):
    """Periodogram of :func:`ou_thermal_series`, averaged over realisations.

    Returns
    -------
    freq_hz : ndarray
        Frequency offsets from the line centre, Hz.
    psd : ndarray
        Averaged power spectral density, arbitrary units.
    """
    acc = None
    for _ in range(int(n_realisations)):
        series = ou_thermal_series(n_steps, dt_s, gamma_rad_s, sigma,
                                   acoustic_velocity, slice_thickness_m, rng)
        spec = np.abs(np.fft.fftshift(np.fft.fft(series))) ** 2
        acc = spec if acc is None else acc + spec
    psd = acc / float(n_realisations)
    freq = np.fft.fftshift(np.fft.fftfreq(int(n_steps), d=dt_s))
    return freq, psd


def analytic_ou_spectrum(freq_hz, gamma_rad_s):
    """The Lorentzian the thesis predicts for the thermal background.

    From the power spectral density derived just after Eq. (3.18): the spectrum
    of ``D(z, t)`` is a Lorentzian of FWHM ``Gamma``.  Normalised to unit peak
    here so it can be compared against a periodogram of arbitrary scale.
    """
    omega = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    half = gamma_rad_s / 2.0
    return half ** 2 / (half ** 2 + omega ** 2)


def effective_speckle_modes(exposure_s, fwhm_ghz, override=None):
    """Resolve the speckle mode count, honouring a manual override."""
    if override is not None and override > 0:
        return float(override)
    return physics.speckle_mode_count(exposure_s, fwhm_ghz)
