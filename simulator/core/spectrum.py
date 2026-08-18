"""Building a pixel's spectrum from its material composition.

The mixing rule is the point of this module.  Each material contributes a
Lorentzian *normalised to unit area*, scaled by how much of the pixel it
occupies.  Because the areas add linearly, a pixel split between several
materials has the same total peak area as a pure pixel -- the peak just moves,
broadens or splits.  This is the behaviour the CSV's ``Indensity`` column
already assumes, since that column is an integrated area rather than a height.

A full simulated spectrum contains, in order of height:

* the elastic (Rayleigh) line at the centre of the axis, 20-60k counts in real
  data, which must sit exactly at the axis centre or downstream folding
  misaligns;
* the Brillouin doublet, one Stokes and one anti-Stokes line per material;
* neighbouring diffraction-order lines, which real VIPA spectra show at around
  +-10.5 GHz and which matter because PCA and VCA run on the raw spectrum and
  would otherwise see an unrealistically clean input;
* a flat background.
"""

from __future__ import annotations

import numpy as np

from . import physics


def observed_linewidth_from_intrinsic(intrinsic_ghz, instrument):
    """Observed linewidth (FWHM) produced by a given intrinsic linewidth.

    Lorentzians convolve by adding widths, so the intrinsic Brillouin
    linewidth, the laser's own line (thesis Eq. 3.20) and the Lorentzian part
    of the spectrometer response simply sum.  The Gaussian part is folded in
    with the standard Voigt width approximation, so this stays a fair
    prediction of what a fit will recover.
    """
    lorentz_fwhm = physics.combine_widths(
        intrinsic_ghz,
        instrument.laser_linewidth_ghz,
        instrument.spectrometer_linewidth_ghz,
    )
    gauss_fwhm = float(
        getattr(instrument, "spectrometer_gaussian_linewidth_ghz", 0.0) or 0.0)
    return voigt_fwhm(lorentz_fwhm, gauss_fwhm)


def observed_linewidth(material, instrument):
    """Observed linewidth (FWHM) of a material's line, after broadening."""
    return observed_linewidth_from_intrinsic(material.linewidth_ghz, instrument)


def instrument_linewidth_floor(instrument):
    """Narrowest line the instrument can produce, i.e. intrinsic width zero.

    Nothing can be observed narrower than this, so it is the lower bound when
    converting a measured linewidth back to an intrinsic one.
    """
    return observed_linewidth_from_intrinsic(0.0, instrument)


def intrinsic_linewidth_for_observed(observed_ghz, instrument, max_ghz=50.0):
    """Invert :func:`observed_linewidth_from_intrinsic`.

    Given a linewidth you actually measured -- which already contains the
    instrument's broadening -- return the intrinsic material linewidth that
    would produce it.

    Returns ``None`` when the target is at or below
    :func:`instrument_linewidth_floor`, because no material can make the line
    narrower than the instrument itself.
    """
    observed_ghz = float(observed_ghz)
    if observed_ghz <= instrument_linewidth_floor(instrument) + 1e-9:
        return None
    from scipy.optimize import brentq
    residual = lambda intrinsic: (
        observed_linewidth_from_intrinsic(intrinsic, instrument) - observed_ghz)
    hi = max_ghz
    if residual(hi) < 0:
        return hi
    return float(brentq(residual, 0.0, hi, xtol=1e-10))


def observed_hwhm(material, instrument):
    """Half width at half maximum -- what the Lorentzian maths needs."""
    return observed_linewidth(material, instrument) / 2.0


def voigt_fwhm(lorentz_fwhm, gauss_fwhm):
    """Olivero-Longbothum approximation to the width of a Voigt profile.

    Checked against a numerically exact Voigt profile over the width range
    this simulator uses: worst error 0.022 percent, i.e. about 0.15 MHz on a
    0.68 GHz line.  Far inside the shot-noise scatter, which is several MHz.
    """
    lorentz_fwhm = max(float(lorentz_fwhm), 0.0)
    gauss_fwhm = max(float(gauss_fwhm), 0.0)
    return (0.5346 * lorentz_fwhm
            + np.sqrt(0.2166 * lorentz_fwhm ** 2 + gauss_fwhm ** 2))


def temperature_factor(temperature_k, reference_k=300.0):
    """Signal scaling with temperature, from thesis Eq. (3.12).

    ``sigma = k_B * T * alpha_ac``, and the steady-state acoustic power is
    ``sigma / alpha_ac``, so the scattered Brillouin power is proportional to
    absolute temperature.
    """
    return max(float(temperature_k), 0.0) / float(reference_k)


def material_weights(fractions, materials, conserve_area=False):
    """Per-material Brillouin weights for a batch of pixels.

    Parameters
    ----------
    fractions : ndarray, shape (n_pixels, n_materials)
        Area fraction of each material in each pixel.  Rows sum to 1.
    materials : sequence of Material
    conserve_area : bool
        When true, weights are renormalised so every pixel carries exactly the
        same total area regardless of composition -- the strict reading of
        "sum them and keep the same area".  When false (the default), a pixel
        made of a brighter material genuinely returns more signal, which is
        what a real instrument does.

    Returns
    -------
    ndarray, shape (n_pixels, n_materials)
    """
    fractions = np.atleast_2d(np.asarray(fractions, dtype=float))
    eta = np.array([m.brillouin_efficiency for m in materials], dtype=float)
    weights = fractions * eta[None, :]
    if conserve_area:
        total = weights.sum(axis=1, keepdims=True)
        total[total <= 0] = 1.0
        weights = weights / total
    return weights


def compose_densities(fractions, materials, instrument, exposure_s,
                      temperature_k=300.0, conserve_area=False,
                      bfs_offsets=None):
    """Noise-free expected counts per channel for a batch of pixels.

    Parameters
    ----------
    fractions : ndarray, shape (n_pixels, n_materials)
        Area fractions per pixel, rows summing to 1.
    materials : sequence of Material
    instrument : Instrument
    exposure_s : float
    temperature_k : float
    conserve_area : bool
        See :func:`material_weights`.
    bfs_offsets : ndarray, shape (n_pixels,), optional
        Per-pixel shift of every line, GHz.  Used to inject slow laser drift.

    Returns
    -------
    ndarray, shape (n_pixels, n_channels)
        Expected counts, before any noise is applied.
    """
    fractions = np.atleast_2d(np.asarray(fractions, dtype=float))
    n_pixels = fractions.shape[0]
    axis = instrument.frequency_axis()
    bin_width = instrument.bin_width_ghz

    if bfs_offsets is None:
        bfs_offsets = np.zeros(n_pixels)
    bfs_offsets = np.asarray(bfs_offsets, dtype=float).reshape(n_pixels)

    weights = material_weights(fractions, materials, conserve_area)
    t_factor = temperature_factor(temperature_k)
    budget = float(instrument.photon_rate) * float(exposure_s)

    density = np.zeros((n_pixels, instrument.n_channels), dtype=float)

    for m_idx, material in enumerate(materials):
        w = weights[:, m_idx]
        if not np.any(w > 0):
            continue
        hwhm = observed_hwhm(material, instrument)
        scale = budget * bin_width
        if material.temperature_scaling:
            scale = scale * t_factor

        # Stokes and anti-Stokes each carry half the material's area.
        for sign in (+1.0, -1.0):
            centres = sign * material.bfs_ghz + bfs_offsets
            shape = _lorentzian_rows(axis, centres, hwhm)
            density += (0.5 * scale * w)[:, None] * shape

        in_band = abs(material.bfs_ghz) < abs(axis[-1])
        if (in_band and instrument.include_adjacent_orders
                and instrument.adjacent_brillouin_fraction > 0):
            # Brillouin lines belonging to the neighbouring diffraction order,
            # which fold back into the window at +-(FSR - shift).  Only for
            # materials whose own line is in band -- otherwise a far out-of-band
            # material like a glass coverslip would sprout a spurious peak.
            fsr = instrument.free_spectral_range_ghz
            for sign in (+1.0, -1.0):
                centres = sign * (fsr - material.bfs_ghz) + bfs_offsets
                if abs(centres.mean()) > abs(axis[-1]):
                    continue
                shape = _lorentzian_rows(axis, centres, hwhm)
                density += (0.5 * scale * w
                            * instrument.adjacent_brillouin_fraction)[:, None] * shape

    density += _elastic_component(fractions, materials, instrument, bfs_offsets)
    density = apply_gaussian_response(density, instrument)
    return np.clip(density, 0.0, None)


def apply_gaussian_response(density, instrument):
    """Convolve with the Gaussian part of the spectrometer response.

    Turns each Lorentzian into a Voigt profile.  This is what makes the wings
    fall off faster than a Lorentzian, and it is why a single-Lorentzian fit of
    the result lands on a negative baseline -- exactly the signature the real
    files show, where the vendor ``Base`` column sits near -30 counts.

    Convolution is done in channel space with
    ``scipy.ndimage.gaussian_filter1d``, which is separable and fast enough to
    run over a whole scan at once.
    """
    fwhm = getattr(instrument, "spectrometer_gaussian_linewidth_ghz", 0.0)
    if not fwhm or fwhm <= 0:
        return density
    sigma_ghz = float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_channels = sigma_ghz / max(instrument.bin_width_ghz, 1e-12)
    if sigma_channels < 1e-3:
        return density
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(density, sigma=sigma_channels, axis=-1, mode="nearest")


def _elastic_peak_per_unit_area(instrument):
    """Height a unit-area elastic line ends up with after the Gaussian response.

    Lets ``elastic_peak_counts`` mean the observed height rather than a
    pre-convolution one, which is what is actually comparable to the 20-60k
    counts seen in real files.
    """
    axis = instrument.frequency_axis()
    hwhm = max(instrument.elastic_linewidth_ghz / 2.0, 1e-6)
    probe = _lorentzian_rows(axis, np.zeros(1), hwhm)
    probe = apply_gaussian_response(probe, instrument)
    return max(float(probe.max()), 1e-12)


def _lorentzian_rows(axis, centres, hwhm):
    """Unit-area Lorentzians, one row per centre."""
    hwhm = max(float(hwhm), 1e-6)
    delta = axis[None, :] - np.asarray(centres, dtype=float)[:, None]
    return (1.0 / np.pi) * hwhm / (delta ** 2 + hwhm ** 2)


def _elastic_component(fractions, materials, instrument, bfs_offsets):
    """The Rayleigh line plus any neighbouring-order elastic lines.

    The central line is placed at exactly 0 GHz -- the middle of the axis --
    because ``brillouin_preprocessing.fold_brillouin_spectrum`` folds about
    ``(freq[0] + freq[-1]) / 2`` and any offset would misalign the fold.
    """
    axis = instrument.frequency_axis()
    n_pixels = fractions.shape[0]
    eta = np.array([m.elastic_efficiency for m in materials], dtype=float)
    strength = fractions @ eta  # (n_pixels,)

    hwhm = max(instrument.elastic_linewidth_ghz / 2.0, 1e-6)
    peak = float(instrument.elastic_peak_counts)
    # elastic_peak_counts is the height the line should have *after* the
    # Gaussian response is applied, so the area is calibrated against a probe
    # that has been through the same convolution.
    area = peak / _elastic_peak_per_unit_area(instrument)

    out = np.zeros((n_pixels, instrument.n_channels), dtype=float)
    out += (area * strength)[:, None] * _lorentzian_rows(
        axis, np.zeros(n_pixels), hwhm)

    if instrument.include_adjacent_orders and instrument.adjacent_order_fraction > 0:
        fsr = instrument.free_spectral_range_ghz
        if abs(fsr) <= abs(axis[-1]):
            for sign in (+1.0, -1.0):
                out += (area * strength * instrument.adjacent_order_fraction)[:, None] \
                    * _lorentzian_rows(axis, np.full(n_pixels, sign * fsr), hwhm)
    return out


def truth_parameters(fractions, materials, instrument, exposure_s,
                     temperature_k=300.0, conserve_area=False):
    """Noise-free ground truth for each pixel.

    Returned alongside every generated scan so algorithms can be scored without
    having to trust a fit.

    Returns
    -------
    dict with keys
        ``bfs_ghz``      -- area-weighted true Brillouin shift per pixel
        ``linewidth_ghz`` -- area-weighted true observed linewidth (FWHM)
        ``area``         -- true integrated anti-Stokes area (the ``Indensity``
                            a perfect fit would report)
        ``dominant``     -- index of the highest-fraction material per pixel
        ``purity``       -- that material's area fraction, 1.0 for a pure pixel
    """
    fractions = np.atleast_2d(np.asarray(fractions, dtype=float))
    weights = material_weights(fractions, materials, conserve_area)
    total = weights.sum(axis=1)
    safe = np.where(total > 0, total, 1.0)

    bfs = np.array([m.bfs_ghz for m in materials], dtype=float)
    linewidth = np.array([observed_linewidth(m, instrument) for m in materials], dtype=float)
    warm = np.array([1.0 if m.temperature_scaling else 0.0 for m in materials])

    t_factor = temperature_factor(temperature_k)
    scale_per_material = np.where(warm > 0, t_factor, 1.0)
    budget = float(instrument.photon_rate) * float(exposure_s)
    bin_width = instrument.bin_width_ghz

    # compose_densities lays down 0.5 * budget * bin_width * weight counts per
    # branch, spread over a unit-area Lorentzian.  Integrating that over
    # frequency gives exactly this, in the counts*GHz units of ``Indensity``.
    area = 0.5 * budget * bin_width * (weights * scale_per_material[None, :]).sum(axis=1)

    return {
        "bfs_ghz": (weights @ bfs) / safe,
        "linewidth_ghz": (weights @ linewidth) / safe,
        "area": area,
        "dominant": np.argmax(fractions, axis=1),
        "purity": fractions.max(axis=1),
    }
