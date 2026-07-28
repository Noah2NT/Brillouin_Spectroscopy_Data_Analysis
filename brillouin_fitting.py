import numpy as np


# ─────────────────────────────────────────────
# Core line-shape functions
# ─────────────────────────────────────────────

def lorentzian(x, amp, mean, gamma, c):
    """Lorentzian peak: amp / (1 + ((x - mean) / gamma)^2) + C"""
    return amp / (1 + ((x - mean) / gamma) ** 2) + c


def gaussian(x, amp, mean, std, c):
    """Gaussian peak: amp * exp(-((x-mean)^2) / (2*std^2)) + C"""
    return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2)) + c


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _compute_error(freq, spectrum, params):
    y_pred = lorentzian(freq, *params)
    return np.sum((spectrum - y_pred) ** 2)


def update_parameter(freq, spectrum, params, index, step):
    current_error = _compute_error(freq, spectrum, params)

    params_up = params.copy()
    params_up[index] += step
    error_up = _compute_error(freq, spectrum, params_up)

    params_down = params.copy()
    params_down[index] -= step
    error_down = _compute_error(freq, spectrum, params_down)

    if error_up < current_error and error_up <= error_down:
        return params_up, error_up
    elif error_down < current_error:
        return params_down, error_down
    else:
        return params, current_error


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def initial_lorentzian_guess(spectrum, freq):
    """
    Build a reasonable starting parameter vector for Lorentzian fitting.

    Returns:
        params : ndarray [amp, mean, gamma, C]
        steps  : ndarray — initial gradient-descent step sizes
    """
    c     = float(np.percentile(spectrum, 5))   # baseline estimate
    amp   = float(spectrum.max() - c)
    mean  = float(freq[np.argmax(spectrum)])
    gamma = 0.36

    params = np.array([amp, mean, gamma, c], dtype=float)
    steps  = np.array([max(0.01 * amp, 1.0), 0.01, max(0.01 * gamma, 1e-4), 0.1], dtype=float)
    return params, steps


def fit_lorentzian_iterative(freq, spectrum, params=None, steps=None, max_iter=2000):
    """
    Fit a Lorentzian to a single spectrum using scipy.optimize.curve_fit
    (Levenberg-Marquardt). Robust to any input intensity scale.

    Parameters:
        freq     : ndarray (W)      — frequency axis
        spectrum : ndarray (W)      — intensity values
        params   : ndarray [amp, mean, gamma, C], or None to auto-guess
        steps    : ignored (kept for backwards-compatibility)
        max_iter : int               — maximum iterations

    Returns:
        fitted_curve : ndarray (W)  — evaluated Lorentzian at final params
        final_params : ndarray (4)  — [amp, mean, gamma, C]
    """
    from scipy.optimize import curve_fit

    if params is None:
        params, _ = initial_lorentzian_guess(spectrum, freq)

    p0 = list(params)

    # Bounds: amp >= 0, mean within freq range, gamma > 0, c free
    f_lo = float(np.min(freq))
    f_hi = float(np.max(freq))
    span = max(f_hi - f_lo, 1e-3)
    lower = [0.0,        f_lo, 1e-4, -np.inf]
    upper = [np.inf,     f_hi, span,  np.inf]

    # Clamp p0 inside bounds
    p0[0] = max(p0[0], 0.0)
    p0[1] = float(np.clip(p0[1], f_lo, f_hi))
    p0[2] = float(np.clip(p0[2], 1e-4, span))

    try:
        popt, _ = curve_fit(lorentzian, freq, spectrum, p0=p0,
                            bounds=(lower, upper), maxfev=max_iter)
    except (RuntimeError, ValueError):
        try:
            popt, _ = curve_fit(lorentzian, freq, spectrum, p0=p0,
                                maxfev=max_iter)
        except (RuntimeError, ValueError):
            popt = np.asarray(p0, dtype=float)

    fitted_curve = lorentzian(freq, *popt)
    return fitted_curve, np.asarray(popt, dtype=float)


def fit_all_spectra(peak_freq, peak_spectra, max_iter=50):
    """
    Fit a Lorentzian to every spectrum in the dataset.

    Parameters:
        peak_freq    : ndarray (W)
        peak_spectra : array-like of ndarray, length N
        max_iter     : int

    Returns:
        fitted_curves : ndarray (N, W) — fitted Lorentzian for each spectrum
        fit_params    : ndarray (N, 4) — [amp, mean, gamma, C] per spectrum
    """
    fitted_curves = []
    fit_params    = []

    for spectrum in peak_spectra:
        curve, params = fit_lorentzian_iterative(peak_freq, spectrum,
                                                 max_iter=max_iter)
        fitted_curves.append(curve)
        fit_params.append(params)

    return np.array(fitted_curves), np.array(fit_params)


def extract_fit_parameters(fit_params):
    """
    Unpack the fit parameter matrix into named arrays.

    Parameters:
        fit_params : ndarray (N, 4) — output of fit_all_spectra

    Returns:
        amplitudes  : ndarray (N)
        peak_shifts : ndarray (N) — peak center frequencies
        linewidths  : ndarray (N) — gamma
        baselines   : ndarray (N) — constant offset C
    """
    amplitudes  = fit_params[:, 0]
    peak_shifts = fit_params[:, 1]
    linewidths  = fit_params[:, 2]
    baselines   = fit_params[:, 3]
    return amplitudes, peak_shifts, linewidths, baselines


# ─────────────────────────────────────────────
# Tabletop / double-sigmoid fitting (for line-scan profiles)
# ─────────────────────────────────────────────

def tabletop(x, baseline, amplitude, x_left, x_right, slope_width):
    """
    Smooth tabletop function: low → high plateau → low, with sloped edges.

    Built as the difference of two logistic sigmoids:
        f(x) = baseline + amplitude * ( S(x; x_left) - S(x; x_right) )
    where S(x; x0) = 1 / (1 + exp(-(x - x0) / slope_width)).

    Parameters
    ----------
    x           : ndarray — position axis
    baseline    : float   — flat level outside the cell
    amplitude   : float   — height of the plateau above baseline
    x_left      : float   — position of the left (rising) edge midpoint
    x_right     : float   — position of the right (falling) edge midpoint
    slope_width : float   — controls how sharp the edges are
                            (smaller = sharper, larger = more gradual)
    """
    from scipy.special import expit
    sw = max(abs(slope_width), 1e-6)
    sig_l = expit((x - x_left) / sw)
    sig_r = expit((x - x_right) / sw)
    return baseline + amplitude * (sig_l - sig_r)


def fit_tabletop(x, y, allow_negative=False):
    """
    Fit a tabletop curve to a 1D profile (e.g. Brillouin shift along a line).

    Parameters
    ----------
    x, y           : ndarray — position axis and profile. `x` is assumed to be
                     sorted ascending (the initial guess reads the first/last
                     10% of `y` as the outside-the-cell level).
    allow_negative : bool — if True the plateau may sit BELOW the baseline
                     (negative amplitude). Needed for signals such as the
                     linewidth, where the cell can be narrower than the
                     surrounding medium. Default False keeps the original
                     amplitude >= 0 behaviour.

    Returns
    -------
    params : dict with keys 'baseline', 'amplitude', 'x_left', 'x_right',
             'slope_width'
    fitted : ndarray — the tabletop evaluated at x
    """
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Fit only on finite points; the returned curve is still evaluated on the
    # full input x so the caller gets an array of the original length.
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5:
        raise ValueError("fit_tabletop: fewer than 5 finite points")
    xf, yf = x[finite], y[finite]

    # Initial guesses from the data
    n = len(yf)
    edge_window = max(1, n // 10)
    baseline_0 = float(np.median(np.concatenate([yf[:edge_window],
                                                 yf[-edge_window:]])))
    up = float(np.max(yf) - baseline_0)
    down = float(np.min(yf) - baseline_0)
    if allow_negative and abs(down) > abs(up):
        amplitude_0 = down          # plateau dips below the surrounding level
    else:
        amplitude_0 = up
    half_level = baseline_0 + amplitude_0 / 2.0

    # Points on the plateau side of the half-way level
    if amplitude_0 >= 0:
        beyond = np.where(yf > half_level)[0]
    else:
        beyond = np.where(yf < half_level)[0]
    if len(beyond) >= 2:
        x_left_0 = float(xf[beyond[0]])
        x_right_0 = float(xf[beyond[-1]])
    else:
        # fallback: use 1/3 and 2/3 of the range
        x_left_0 = float(xf[n // 3])
        x_right_0 = float(xf[2 * n // 3])

    span = xf[-1] - xf[0]
    slope_width_0 = max(span / 50.0, 1e-3)

    p0 = [baseline_0, amplitude_0, x_left_0, x_right_0, slope_width_0]

    # Bounds: keep parameters physical and robust to non-monotonic x
    x_lo = float(np.min(xf))
    x_hi = float(np.max(xf))
    # Median spacing of the UNIQUE x values: when several scans are pooled onto
    # a shared axis many points share an x, and a raw diff would collapse to 0.
    ux = np.unique(xf)
    pixel_spacing = float(np.median(np.diff(ux))) if len(ux) > 1 else 1.0
    min_slope = max(pixel_spacing * 0.5, 1e-3)
    span_abs = max(x_hi - x_lo, 1e-3)

    amp_lo = -np.inf if allow_negative else 0.0
    lower = [-np.inf, amp_lo, x_lo, x_lo, min_slope]
    upper = [ np.inf, np.inf, x_hi, x_hi, span_abs]

    # Clip the initial guess to be strictly inside the bounds
    if allow_negative:
        if abs(p0[1]) < 1e-6:
            p0[1] = 1e-6
    else:
        p0[1] = max(p0[1], 1e-6)         # amplitude > 0
    p0[2] = float(np.clip(p0[2], x_lo, x_hi))
    p0[3] = float(np.clip(p0[3], x_lo, x_hi))
    p0[4] = float(np.clip(p0[4], min_slope, span_abs))

    try:
        popt, _ = curve_fit(tabletop, xf, yf, p0=p0,
                            bounds=(lower, upper), maxfev=5000)
    except (RuntimeError, ValueError):
        # Fallback: try without bounds — at least returns *some* curve
        try:
            popt, _ = curve_fit(tabletop, xf, yf, p0=p0, maxfev=5000)
        except (RuntimeError, ValueError):
            popt = p0

    params = {
        "baseline":    popt[0],
        "amplitude":   popt[1],
        "x_left":      popt[2],
        "x_right":     popt[3],
        "slope_width": popt[4],
    }
    fitted = tabletop(x, *popt)
    return params, fitted


def tabletop_regions(x, params, edge_pad=2.0):
    """
    Use a fitted tabletop to label each position as 'outside', 'edge', or
    'center'.

    The edge zone extends `edge_pad * slope_width` on either side of the
    fitted edge midpoints.

    Parameters
    ----------
    x         : ndarray   — position axis
    params    : dict      — output of fit_tabletop
    edge_pad  : float     — how many slope_widths to count as part of the
                            'edge' region (default 2.0 covers ~88% of the
                            sigmoid transition)

    Returns
    -------
    region : ndarray of dtype object/str, same length as x, values from
             {'outside', 'edge', 'center'}
    """
    x = np.asarray(x, dtype=float)
    xL = params["x_left"]
    xR = params["x_right"]
    pad = edge_pad * params["slope_width"]

    region = np.full(len(x), "outside", dtype=object)
    in_edge_left = (x >= xL - pad) & (x <= xL + pad)
    in_edge_right = (x >= xR - pad) & (x <= xR + pad)
    in_center = (x > xL + pad) & (x < xR - pad)

    region[in_edge_left | in_edge_right] = "edge"
    region[in_center] = "center"
    return region
