"""
Maximum Entropy Reconstruction (MER) for Brillouin spectrum denoising.

Implementation based on:
    Xiang, Foreman & Torok,
    "SNR enhancement in brillouin microspectroscopy using spectrum reconstruction"
    Biomedical Optics Express 11(2), 1020-1031 (2020).
    https://doi.org/10.1364/BOE.380798

The algorithm finds the spectrum f that maximises the Shannon entropy
    S = -sum(f_i * log(f_i))
subject to the chi-squared data-fit constraint
    chi2 = (1/N) * sum((R @ f - d)^2 / sigma^2) <= chi2_0 (~1 for 95% confidence).

This is solved by maximising the Lagrangian Q = S - lambda*(chi2 - chi2_0)
via conjugate-gradient ascent with a Wolfe-conditions line search.
"""

import numpy as np


# --------------------------------------------------------------------------
# Core entropy / chi-squared helpers
# --------------------------------------------------------------------------

def _entropy(f, eps=1e-12):
    fp = np.clip(f, eps, None)
    return -np.sum(fp * np.log(fp))


def _grad_entropy(f, eps=1e-12):
    fp = np.clip(f, eps, None)
    return -(np.log(fp) + 1.0)


def _chi2(f, d, sigma, R=None):
    residual = (R @ f - d) if R is not None else (f - d)
    return np.sum((residual / sigma) ** 2) / len(d)


def _grad_chi2(f, d, sigma, R=None):
    N = len(d)
    if R is None:
        residual = f - d
        return (2.0 / N) * residual / sigma**2
    residual = R @ f - d
    return (2.0 / N) * (R.T @ (residual / sigma**2))


def _lagrangian(f, d, sigma, lam, R=None):
    # chi2_0 is a constant w.r.t. f, so we drop it for line-search comparisons
    return _entropy(f) - lam * _chi2(f, d, sigma, R)


def _grad_lagrangian(f, d, sigma, lam, R=None):
    return _grad_entropy(f) - lam * _grad_chi2(f, d, sigma, R)


# --------------------------------------------------------------------------
# Wolfe line search (for ascent)
# --------------------------------------------------------------------------

def _wolfe_line_search(f, p, grad, d, sigma, lam, R,
                       mu_init=1.0, c1=1e-4, c2=0.9,
                       max_iter=25, eps=1e-12):
    """
    Backtracking line search satisfying the Wolfe conditions
    (rewritten for maximisation rather than minimisation).
    """
    Q0 = _lagrangian(f, d, sigma, lam, R)
    g_dot_p = float(np.dot(grad, p))

    if g_dot_p <= 0:
        # not an ascent direction - fall back to gradient
        p = grad
        g_dot_p = float(np.dot(grad, p))
        if g_dot_p <= 0:
            return 0.0, f, grad

    mu = mu_init
    for _ in range(max_iter):
        f_new = np.clip(f + mu * p, eps, None)
        Q_new = _lagrangian(f_new, d, sigma, lam, R)

        # Armijo (sufficient increase, ascent form)
        if Q_new >= Q0 + c1 * mu * g_dot_p:
            grad_new = _grad_lagrangian(f_new, d, sigma, lam, R)
            # Curvature condition (ascent form)
            if np.dot(grad_new, p) <= c2 * g_dot_p:
                return mu, f_new, grad_new
        mu *= 0.5

    f_new = np.clip(f + mu * p, eps, None)
    return mu, f_new, _grad_lagrangian(f_new, d, sigma, lam, R)


# --------------------------------------------------------------------------
# Main MER routine
# --------------------------------------------------------------------------

def maximum_entropy_reconstruction(data, sigma=None, R=None, lam=1.0,
                                   max_iter=1000, tol=0.05, f_init=None,
                                   init_smooth_sigma=2.0):
    """
    Reconstruct a denoised spectrum from a single noisy spectrum using MER.

    Parameters
    ----------
    data : ndarray (N,)
        Noisy spectrum.
    sigma : float, ndarray (N,), or None
        Noise standard deviation. If None, estimated from the median absolute
        deviation of the first differences (assumes white-ish noise).
    R : ndarray (N, N) or None
        System (instrument) response matrix. None means R = identity.
    lam : float
        Lagrange multiplier balancing entropy against data fit.
        Larger -> closer to the noisy data, Smaller -> smoother (higher entropy).
    max_iter : int
        Maximum number of conjugate-gradient iterations.
    tol : float
        Convergence threshold on the gradient-balance metric
        0.5 * | grad_S / |grad_S|  -  grad_chi2 / |grad_chi2| |^2 .
    f_init : ndarray (N,) or None
        Initial estimate. Defaults to the data clipped to positive values.

    Returns
    -------
    f : ndarray (N,)
        Reconstructed spectrum (always positive).
    info : dict
        Convergence diagnostics: 'converged', 'iterations', 'final_chi2',
        'final_balance', and per-iteration 'history'.
    """
    d = np.asarray(data, dtype=float)
    N = len(d)

    # Noise estimation if not supplied (MAD of first differences -> std)
    if sigma is None:
        sigma_scalar = np.median(np.abs(np.diff(d))) / 0.6745
        sigma_scalar = max(sigma_scalar, 1e-6)
        sigma = np.full(N, sigma_scalar)
    else:
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (N,)).copy()

    # Positive initialisation - MER requires f > 0 for log to be defined
    # A lightly smoothed copy of the data helps convergence dramatically.
    if f_init is None:
        floor = max(1e-6 * np.max(np.abs(d)), 1e-12)
        if init_smooth_sigma and init_smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            f = gaussian_filter1d(d, sigma=init_smooth_sigma)
        else:
            f = d.copy()
        f = np.clip(f, floor, None)
    else:
        f = np.clip(np.asarray(f_init, dtype=float), 1e-12, None)

    grad = _grad_lagrangian(f, d, sigma, lam, R)
    p = grad.copy()

    history = {'chi2': [], 'entropy': [], 'balance': []}
    converged = False
    it = 0

    for it in range(max_iter):
        mu, f, grad_new = _wolfe_line_search(f, p, grad, d, sigma, lam, R)

        # Termination metric from the paper: how aligned grad_S and grad_chi2 are
        gS = _grad_entropy(f)
        gC = _grad_chi2(f, d, sigma, R)
        nS = np.linalg.norm(gS)
        nC = np.linalg.norm(gC)
        balance = (0.5 * np.sum((gS / nS - gC / nC) ** 2)
                   if nS > 0 and nC > 0 else np.inf)

        history['chi2'].append(_chi2(f, d, sigma, R))
        history['entropy'].append(_entropy(f))
        history['balance'].append(balance)

        if balance < tol:
            converged = True
            break

        # Polak-Ribiere conjugate-gradient update
        denom = float(np.dot(grad, grad)) + 1e-12
        beta = max(0.0, float(np.dot(grad_new, grad_new - grad)) / denom)
        p = grad_new + beta * p
        grad = grad_new

        # Periodic restart (steepest ascent) helps stability
        if (it + 1) % 25 == 0:
            p = grad.copy()

    info = {
        'converged': converged,
        'iterations': it + 1,
        'final_chi2': history['chi2'][-1] if history['chi2'] else None,
        'final_balance': history['balance'][-1] if history['balance'] else None,
        'history': history,
    }
    return f, info


# --------------------------------------------------------------------------
# Batch helper for hyperspectral cubes
# --------------------------------------------------------------------------

def mer_batch(spectra, sigma=None, R=None, lam=1.0,
              max_iter=1000, tol=0.05, verbose=False,
              init_smooth_sigma=2.0):
    """
    Apply MER independently to each spectrum in a (N_spectra, N_bins) array.

    Returns
    -------
    reconstructed : ndarray (N_spectra, N_bins)
    n_converged   : int — number of spectra that satisfied the tolerance
    """
    spectra = np.asarray(spectra, dtype=float)
    out = np.empty_like(spectra)
    n_converged = 0
    total = spectra.shape[0]
    for i in range(total):
        f, info = maximum_entropy_reconstruction(
            spectra[i], sigma=sigma, R=R, lam=lam,
            max_iter=max_iter, tol=tol,
            init_smooth_sigma=init_smooth_sigma,
        )
        out[i] = f
        if info['converged']:
            n_converged += 1
        if verbose and (i + 1) % max(1, total // 10) == 0:
            print(f"  MER: {i+1}/{total} spectra processed "
                  f"({n_converged} converged)")
    return out, n_converged
