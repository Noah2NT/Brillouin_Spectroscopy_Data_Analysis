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


def fit_lorentzian_iterative(freq, spectrum, params=None, steps=None, max_iter=50):
    """
    Fit a Lorentzian to a single spectrum using iterative coordinate descent.

    Parameters:
        freq     : ndarray (W)      — frequency axis
        spectrum : ndarray (W)      — intensity values
        params   : ndarray [amp, mean, gamma, C], or None to auto-guess
        steps    : ndarray (4,), or None to auto-guess
        max_iter : int               — maximum iterations

    Returns:
        fitted_curve : ndarray (W)  — evaluated Lorentzian at final params
        final_params : ndarray (4)  — [amp, mean, gamma, C]
    """
    if params is None or steps is None:
        params, steps = initial_lorentzian_guess(spectrum, freq)

    params = params.copy()
    steps  = steps.copy()
    phase  = 0

    for _ in range(max_iter):
        params_prev = params.copy()
        for idx in range(4):
            params, _ = update_parameter(freq, spectrum, params, idx, steps[idx])

        if np.array_equal(params, params_prev):
            if phase < 4:
                phase += 1
                steps *= 0.5
            else:
                break

    fitted_curve = lorentzian(freq, *params)
    return fitted_curve, params


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
