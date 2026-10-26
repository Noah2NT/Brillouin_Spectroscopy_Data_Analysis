"""Solve for EM_2 instead of picking it, with noise-weighted least squares.

The hydrogel blob scan holds two materials, so two spectra should describe
every pixel:

    S ~ X @ EM      S (N,B) measured, EM = [EM_1; EM_2] (2,B), X = [x1,x2] (N,2)

EM_1 is known -- it is the co-add of the confident background pixels. EM_2 and
X are solved for, minimising

    || (S - X @ EM) / SIGMA ||_F^2

subject to  X @ 1 = 1,  X >= 0,  EM_2 >= 0,  EM_2 @ 1 = 1.

WHY THE WEIGHTS. Noise here is shot noise, so its size follows the signal: a
photon transfer curve on the background pixels gives var = alpha * signal with
R^2 ~ 0.999 and negligible read noise. Across one spectrum sigma varies by
about 12x between the peak and the tail, so an UNWEIGHTED fit would be
dominated by the peak bins and would happily wreck the tail -- which is
exactly where a second material is most visible, because EM_1 is near zero
there and cannot mask anything. Dividing by sigma turns each residual into a
z-score, so bins compete on significance rather than raw magnitude. It is also
the maximum-likelihood estimator, and it makes chi2/dof ~ 1 a meaningful
statement about success rather than a number in arbitrary units.

SIGMA DEPENDS ON THE FIT, not on the data, so the weights are recomputed each
iteration -- iteratively reweighted least squares. Using the observed counts
instead would bias the weights wherever noise happened to dip.

THE SCALE IS NOT IDENTIFIABLE. x2 * (EM_2 - EM_1) is unchanged under
EM_2 - EM_1 -> c(EM_2 - EM_1), x2 -> x2/c, and because EM_2 - EM_1 sums to
zero the sum-to-one constraint does NOT pin c down. Only the inequalities do:

    x2 <= 1     lower bound on c
    EM_2 >= 0   upper bound on c

Between those the objective is exactly flat -- every c fits identically well.
So both ends are reported rather than one being passed off as the answer:

    x2max   scaled so the most hydrogel-rich pixel is called pure
    nonneg  scaled until EM_2 first touches zero, the most extreme spectrum
            still physically possible
"""

from pathlib import Path
import importlib.util
import sys, os

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_decomposition as bd
import brillouin_fitting as bf


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bge = _load("background_endmember")

DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "two_material"
DEFAULT_TARGET = "Hydrogl1"

N_ITER = 60
SEED = 42
EPS = 1e-12


# -------------------------------------------------
# Noise model
# -------------------------------------------------

def fit_alpha(raw_bg):
    """Photon transfer curve: var = alpha * signal, fitted on background px.

    Each pixel is compared with its OWN expected spectrum (the co-added shape
    rescaled to that pixel's total counts), so genuine pixel-to-pixel
    differences cancel and only noise is left. Taking the variance ACROSS
    pixels instead folds real scene variation into the estimate and inflates
    it several-fold.
    """
    shape = raw_bg.sum(axis=0) / raw_bg.sum()
    total = raw_bg.sum(axis=1, keepdims=True)
    expected = shape[None, :] * total
    sq = (raw_bg - expected) ** 2

    e, v = expected.ravel(), sq.ravel()
    edges = np.quantile(e, np.linspace(0, 1, 21))
    mids, vars_ = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (e >= lo) & (e < hi)
        if k.sum() > 50:
            mids.append(e[k].mean())
            vars_.append(v[k].mean())
    mids, vars_ = np.array(mids), np.array(vars_)
    alpha, intercept = np.polyfit(mids, vars_, 1)
    pred = alpha * mids + intercept
    ss_tot = float(((vars_ - vars_.mean()) ** 2).sum())
    r2 = 1.0 - float(((vars_ - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return float(alpha), float(intercept), float(r2)


def sigma_of(model, alpha, counts):
    """Per-element noise sigma of a sum-normalised spectrum: sqrt(a*s/C)."""
    return np.sqrt(alpha * np.clip(model, EPS, None) / counts[:, None])


def sigma_mix(x2, EM_1, EM_2, a1, a2, counts):
    """Noise sigma with each endmember carrying its own alpha.

    For Poisson the variance equals the mean, and the mean is the linear
    mixture, so VARIANCES blend by abundance -- not sigmas. A pixel that is
    half EM_1 and half EM_2 has

        var = 0.5*(a1*EM_1/C) + 0.5*(a2*EM_2/C)

    i.e. sqrt(0.5*var_1 + 0.5*var_2), NOT 0.5*sigma_1 + 0.5*sigma_2. Blending
    sigmas linearly would understate the noise everywhere except at the two
    pure ends.

    Collapses exactly to sigma_of when a1 == a2, so this is a strict
    generalisation of the single-alpha model.
    """
    x1 = 1.0 - x2
    var = (x1[:, None] * a1 * EM_1[None, :]
           + x2[:, None] * a2 * EM_2[None, :]) / counts[:, None]
    return np.sqrt(np.clip(var, EPS, None))


def fit_alpha_pure(raw):
    """alpha for a group of pixels assumed to be the same material.

    Same photon transfer curve as fit_alpha, just applied to any group. It
    measures scatter WITHIN the group about the group's own co-added shape and
    never touches the two-material model's residual, which is what keeps chi2
    an independent test rather than something tuned to come out at 1.
    """
    return fit_alpha(raw)


# -------------------------------------------------
# The solve
# -------------------------------------------------

def solve(S, EM_1, counts, alpha, n_iter=N_ITER, a2=None, EM_2_ref=None):
    """Weighted rank-1 alternating least squares for EM_2 and x2.

    Works on u = EM_2 - EM_1 rather than on EM_2 directly. That disposes of
    the sum-to-one constraint (u sums to zero, so any scaling of u leaves
    EM_2 summing to one) and leaves non-negativity of EM_2 as purely a
    question of scale, settled afterwards.
    """
    def _sigma(model, x2v, cnt):
        """Blended sigma once a second alpha is known; single-alpha until then."""
        if a2 is None or EM_2_ref is None:
            return sigma_of(model, alpha, cnt)
        top = float(x2v.max())
        frac = np.clip(x2v / top, 0.0, 1.0) if top > 0 else x2v
        return sigma_mix(frac, EM_1, EM_2_ref, alpha, a2, cnt)

    D = S - EM_1[None, :]

    # Start from the unweighted rank-1 answer. The SVD must be UNCENTERED:
    # the model d_p = x2_p * u has no intercept term, so removing the mean
    # first would fit a different model.
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    u = Vt[0]
    x2 = D @ u
    if x2.sum() < 0:                 # SVD sign is arbitrary; orient outward
        u, x2 = -u, -x2

    def cap_for(direction):
        """Largest x2 keeping EM_1 + x2*direction non-negative in every bin.

        This IS the EM_2 >= 0 constraint, expressed per pixel. Without it the
        model can go negative, and since the weights are 1/sigma^2 =
        C/(alpha*model), a bin driven to the numerical floor collects a weight
        of order 1e12 and hijacks the whole regression -- which is exactly how
        the first version of this diverged.
        """
        neg = direction < 0
        return float(np.min(EM_1[neg] / -direction[neg])) if neg.any() else np.inf

    history, best = [], None
    for _ in range(n_iter):
        x2 = np.clip(x2, 0.0, cap_for(u))
        model = EM_1[None, :] + x2[:, None] * u[None, :]
        w = 1.0 / _sigma(model, x2, counts) ** 2

        # x2: one weighted scalar regression per pixel
        x2 = ((w * D) @ u) / np.maximum((w * u[None, :] ** 2).sum(axis=1), EPS)
        x2 = np.clip(x2, 0.0, cap_for(u))

        # u: one weighted scalar regression per bin
        u = (w * D * x2[:, None]).sum(axis=0) / np.maximum(
            (w * x2[:, None] ** 2).sum(axis=0), EPS)
        u -= u.mean()                # hold sum(u) = 0 exactly
        scale = np.linalg.norm(u)
        if scale > 0:                # fix the gauge so the loop cannot drift
            u, x2 = u / scale, x2 * scale
        x2 = np.clip(x2, 0.0, cap_for(u))

        model = EM_1[None, :] + x2[:, None] * u[None, :]
        sig = _sigma(model, x2, counts)
        chi2 = float((((S - model) / sig) ** 2).mean())
        history.append(chi2)
        # Keep the best iterate rather than whatever the last one happened to
        # be: alternating least squares is not guaranteed monotone here.
        if best is None or chi2 < best[0]:
            best = (chi2, u.copy(), x2.copy())

    return best[1], best[2], np.array(history)


def scale_x2max(u, x2):
    """Call the most hydrogel-rich pixel pure: max(x2) = 1."""
    c = float(x2.max())
    return c, x2 / max(c, EPS)


def scale_nonneg(u, x2, EM_1):
    """Push EM_2 = EM_1 + c*u out until it first touches zero."""
    neg = u < 0
    if not neg.any():
        return scale_x2max(u, x2)
    c = float(np.min(-EM_1[neg] / u[neg]))
    return c, x2 / max(c, EPS)


def peak_width(spectrum, shift):
    """Lorentzian peak position and width -- 'did it come out a curve?'."""
    _, params = bf.fit_all_spectra(shift, np.clip(spectrum, 0, None)[None, :])
    _, shifts, widths, _ = bf.extract_fit_parameters(params)
    return float(shifts[0]), float(widths[0])


def spectral_angle(a, b):
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# -------------------------------------------------
# Plotting
# -------------------------------------------------

def to_map(values, yi, xi, shape):
    grid = np.full(shape, np.nan)
    grid[yi, xi] = values
    return grid


def render_solve(stem, data, bfs_raw, bg_flat, blob_flat, EM_1, EM_2a, EM_2b,
                 x2a, ca, cb, S, alpha, alpha2, counts, hist, z, out_path):
    shape, yi, xi = data["shape"], data["yi"], data["xi"]
    shift = data["shift_freq"]
    model = (1 - x2a)[:, None] * EM_1[None, :] + x2a[:, None] * EM_2a[None, :]
    sig = sigma_mix(x2a, EM_1, EM_2a, alpha, alpha2, counts)
    resid = S - model
    z_all = np.abs(resid) / sig
    chi2_dof = (z_all ** 2).mean(axis=1)
    within = (z_all <= z).mean(axis=1)

    fig = plt.figure(figsize=(20, 14.5), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(bfs_raw, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8, label="GHz")
    ax.set_title("BFS map", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1:3])
    ax.plot(shift, EM_1, lw=1.8, color="tab:blue", label="EM_1 (background)")
    ax.plot(shift, EM_2a, lw=2.4, color="tab:red",
            label=f"EM_2, x2max scaling (c={ca:.4f})")
    ax.plot(shift, EM_2b, lw=1.5, ls="--", color="tab:orange",
            label=f"EM_2, nonneg scaling (c={cb:.4f})")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("norm. intensity")
    p1, w1 = peak_width(EM_1, shift)
    p2, w2 = peak_width(EM_2a, shift)
    ax.set_title(f"THE SOLVED ENDMEMBER\nEM_1 {p1:.3f} GHz (width {w1:.3f})   "
                 f"|   EM_2 {p2:.3f} GHz (width {w2:.3f})", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(hist, lw=1.6, color="tab:purple")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="chi2/dof = 1")
    ax.set_xlabel("iteration"); ax.set_ylabel("chi2 / dof")
    ax.set_title(f"convergence (final {hist[-1]:.3f})", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(to_map(x2a, yi, xi, shape), origin="lower", cmap="viridis",
                   vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("x2  (hydrogel fraction, x2max scaling)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    hi = float(np.nanpercentile(chi2_dof, 98))
    im = ax.imshow(to_map(chi2_dof, yi, xi, shape), origin="lower",
                   cmap="magma", vmin=0, vmax=hi)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("chi2/dof per pixel\n(1.0 = as good as the noise allows)",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    for mask, colour, lab in ((bg_flat, "tab:blue", "background"),
                              (~(bg_flat | blob_flat), "0.45", "uncertain"),
                              (blob_flat, "tab:red", "blob")):
        if mask.any():
            ax.hist(chi2_dof[mask], bins=40, color=colour, alpha=0.6, label=lab)
    ax.axvline(1.0, color="k", lw=1.6, label="chi2/dof = 1")
    ax.set_xlabel("chi2 / dof"); ax.set_ylabel("pixels")
    ax.set_title("is the fit as good as the noise allows?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(to_map(100 * within, yi, xi, shape), origin="lower",
                   cmap="viridis", vmin=90, vmax=100)
    fig.colorbar(im, ax=ax, shrink=0.8, label="%")
    ax.set_title(f"% of bins inside tau = {z:.2f} sigma", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    order = np.argsort(x2a)
    picks = order[np.linspace(0, len(order) - 1, 4).round().astype(int)]
    for j, p in enumerate(picks):
        ax = fig.add_subplot(gs[2, j])
        ax.plot(shift, S[p], lw=1.6, color="0.35", label="measured")
        ax.plot(shift, model[p], lw=1.3, ls="--", color="tab:red",
                label=f"fit, x2={x2a[p]:.3f}")
        ax.fill_between(shift, model[p] - z * sig[p], model[p] + z * sig[p],
                        color="tab:red", alpha=0.18, label="+/- tau")
        ax.plot(shift, resid[p], lw=1.0, color="tab:green", label="residual")
        ax.axhline(0, color="0.5", lw=0.7)
        cls = ("background" if bg_flat[p] else
               "blob" if blob_flat[p] else "uncertain")
        ax.set_title(f"{cls} px {p}   chi2/dof={chi2_dof[p]:.2f}", fontsize=9)
        ax.set_xlabel("Shift (GHz)")
        ax.legend(fontsize=6); ax.grid(alpha=0.3)

    fig.suptitle(f"{stem}   |   solved two-material fit   |   "
                 "S = X @ [EM_1; EM_2], weighted by shot noise", fontsize=13)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    return chi2_dof, within


def render_vca(stem, data, EM_1, EM_2a, x2a, S, em_vca, ab_vca, out_path):
    """Same scan unmixed by VCA(2) on its own, beside the solved pair."""
    shape, yi, xi = data["shape"], data["yi"], data["xi"]
    shift = data["shift_freq"]

    # Order VCA's pair so its background-like member comes first.
    if np.abs(em_vca[1] - EM_1).sum() < np.abs(em_vca[0] - EM_1).sum():
        em_vca = em_vca[::-1]
        ab_vca = ab_vca[:, ::-1]

    ab_solved = np.column_stack([1 - x2a, x2a])

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(shift, EM_1, lw=2.2, color="tab:blue", label="EM_1 solved (co-add)")
    ax.plot(shift, EM_2a, lw=2.2, color="tab:red", label="EM_2 solved")
    ax.plot(shift, em_vca[0], lw=1.3, ls="--", color="tab:cyan",
            label="VCA(2) endmember A")
    ax.plot(shift, em_vca[1], lw=1.3, ls="--", color="tab:orange",
            label="VCA(2) endmember B")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("norm. intensity")
    ax.set_title("solved endmembers vs what VCA(2) picks on its own",
                 fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(to_map(x2a, yi, xi, shape), origin="lower", cmap="viridis",
                   vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("x2, solved", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(to_map(ab_vca[:, 1], yi, xi, shape), origin="lower",
                   cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("second-endmember abundance, VCA(2)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(x2a, ab_vca[:, 1], s=5, alpha=0.4, color="tab:purple")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9)
    r = float(np.corrcoef(x2a, ab_vca[:, 1])[0, 1])
    ax.set_xlabel("x2 solved"); ax.set_ylabel("VCA(2) abundance")
    ax.set_title(f"do they agree per pixel?   r = {r:.4f}", fontsize=10)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(shift, EM_2a, lw=2.0, color="tab:red", label="EM_2 solved")
    ax.plot(shift, em_vca[1], lw=1.4, ls="--", color="tab:orange",
            label="VCA(2) B")
    ax.plot(shift, EM_2a - em_vca[1], lw=1.0, color="k", label="difference")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Shift (GHz)")
    ax.set_title("second endmember: solved vs VCA\nspectral angle "
                 f"{spectral_angle(EM_2a, em_vca[1]):.1f} deg", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    for j, (em, ab, name) in enumerate(
            [(np.vstack([EM_1, EM_2a]), ab_solved, "solved"),
             (em_vca, ab_vca, "VCA(2)")]):
        ax = fig.add_subplot(gs[1, 2 + j])
        err = np.linalg.norm(S - ab @ em, axis=1)
        im = ax.imshow(to_map(err, yi, xi, shape), origin="lower", cmap="magma")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"reconstruction error, {name}\nmean {err.mean():.2e}",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{stem}   |   VCA(2) on the hydrogel blob, against the "
                 "solved pair", fontsize=13)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    return em_vca, ab_vca


def render_width_test(stem, data, S, EM_1, EM_2a, x2a, bg_flat, blob_flat,
                      out_path):
    """Mixture or continuum? The hypotheses predict opposite linewidths.

    Combining two peaks at different positions always yields something BROADER
    than either, so a genuine two-material mixture must show an arch: width
    maximal near x2 = 0.5, narrow at both pure ends. A single peak that merely
    slides in position keeps a roughly CONSTANT width.

    The predicted curve is not asserted, it is measured -- synthesise
    (1-x2)*EM_1 + x2*EM_2 on a grid of x2 and fit each synthetic spectrum with
    exactly the same Lorentzian routine used on the real pixels, so any bias
    in the fitter cancels between the two.
    """
    shift = data["shift_freq"]
    _, params = bf.fit_all_spectra(shift, S)
    _, meas_shift, meas_width, _ = bf.extract_fit_parameters(params)

    grid = np.linspace(0, 1, 41)
    synth = (1 - grid)[:, None] * EM_1[None, :] + grid[:, None] * EM_2a[None, :]
    _, sp = bf.fit_all_spectra(shift, synth)
    _, pred_shift, pred_width, _ = bf.extract_fit_parameters(sp)

    fig = plt.figure(figsize=(18, 5.6), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig)
    groups = ((bg_flat, "tab:blue", "background"),
              (~(bg_flat | blob_flat), "0.5", "uncertain"),
              (blob_flat, "tab:red", "blob"))

    ax = fig.add_subplot(gs[0, 0])
    for mask, colour, lab in groups:
        if mask.any():
            ax.scatter(x2a[mask], meas_width[mask], s=7, alpha=0.45,
                       color=colour, label=lab)
    ax.plot(grid, pred_width, lw=2.4, color="k",
            label="predicted, 2-material mixture")
    ax.set_xlabel("x2  (hydrogel fraction)")
    ax.set_ylabel("fitted width (GHz)")
    ax.set_title("THE TEST: arch = mixture, flat = sliding peak", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for mask, colour, lab in groups:
        if mask.any():
            ax.scatter(x2a[mask], meas_shift[mask], s=7, alpha=0.45,
                       color=colour, label=lab)
    ax.plot(grid, pred_shift, lw=2.4, color="k", label="predicted, mixture")
    ax.set_xlabel("x2"); ax.set_ylabel("fitted peak (GHz)")
    ax.set_title("peak position vs abundance", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    inb = blob_flat & np.isfinite(meas_width)
    excess = meas_width[inb] - np.interp(x2a[inb], grid, pred_width)
    ax.scatter(x2a[inb], excess, s=8, alpha=0.5, color="tab:red")
    ax.axhline(0, color="k", lw=1.2, label="mixture prediction")
    ax.set_xlabel("x2"); ax.set_ylabel("measured - predicted width (GHz)")
    ax.set_title("width excess over the mixture model"
                 + chr(10) + "median {:+.4f} GHz".format(np.median(excess)),
                 fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(stem + "   |   two materials, or one peak that slides?",
                 fontsize=13)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    return meas_width, pred_width, grid


# -------------------------------------------------
# Driver
# -------------------------------------------------

def process(path):
    stem = path.stem
    data = bge.load_cube(path)
    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])
    bg_mask, blob_mask, _, _ = bge.find_background(data, bfs_raw, bfs_sm)
    EM_1, _, _ = bge.build_endmember(data, bg_mask)

    yi, xi = data["yi"], data["xi"]
    bg_flat, blob_flat = bg_mask[yi, xi], blob_mask[yi, xi]
    S = data["folded_norm"]
    counts = data["folded_raw"].sum(axis=1)
    B = S.shape[1]
    z = float(norm.ppf(1 - 0.05 / (2 * B)))

    alpha, icept, r2 = fit_alpha(data["folded_raw"][bg_flat])
    print(f"  noise model: var = {alpha:.3f}*signal + {icept:.1f}  R2={r2:.4f}"
          f" | z(N={B}) = {z:.3f}")

    # Pass 1: single alpha, enough to locate the near-pure hydrogel pixels.
    u, x2_raw, hist = solve(S, EM_1, counts, alpha)
    ca, x2a = scale_x2max(u, x2_raw)
    EM_2a = EM_1 + ca * u

    # alpha_2 from the pixels closest to pure hydrogel, measured exactly as
    # alpha_1 was. Those pixels are not identical (there is a gradient), so
    # their scatter carries some real variation and alpha_2 comes out biased
    # HIGH -- which biases toward excusing the misfit, so the ratio below
    # should be read as an upper bound on how much of chi2 is noise.
    top = None
    if blob_flat.any():
        top = x2a >= np.quantile(x2a[blob_flat], 0.90)
    if top is not None and int(top.sum()) >= 30:
        alpha2, icept2, r2b = fit_alpha_pure(data["folded_raw"][top])
        n_top = int(top.sum())
    else:
        alpha2, icept2, r2b, n_top = alpha, 0.0, float("nan"), 0
    print(f"  alpha_1 (background) = {alpha:.3f} | alpha_2 (top-decile blob, "
          f"n={n_top}) = {alpha2:.3f} R2={r2b:.4f} | a2/a1 = "
          f"{alpha2 / alpha:.2f}")

    # Pass 2: re-solve with the blended noise model.
    u, x2_raw, hist = solve(S, EM_1, counts, alpha, a2=alpha2,
                            EM_2_ref=EM_2a)
    ca, x2a = scale_x2max(u, x2_raw)
    cb, x2b = scale_nonneg(u, x2_raw, EM_1)
    EM_2a, EM_2b = EM_1 + ca * u, EM_1 + cb * u

    print(f"  chi2/dof {hist[0]:.3f} -> {hist[-1]:.3f} over {len(hist)} iters")
    feasible = "feasible" if cb >= ca else "INFEASIBLE: nonneg tighter than x2<=1"
    print(f"  scalings: x2max c={ca:.4f}, nonneg c={cb:.4f}  ({feasible})")
    p1, w1 = peak_width(EM_1, data["shift_freq"])
    p2, w2 = peak_width(EM_2a, data["shift_freq"])
    print(f"  EM_1 peak {p1:.3f} GHz width {w1:.3f} | "
          f"EM_2 peak {p2:.3f} GHz width {w2:.3f} | "
          f"EM_2 min {EM_2a.min():.3e}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chi2_dof, within = render_solve(
        stem, data, bfs_raw, bg_flat, blob_flat, EM_1, EM_2a, EM_2b, x2a,
        ca, cb, S, alpha, alpha2, counts, hist, z,
        OUTPUT_DIR / f"{stem}_solve.png")
    print(f"  chi2/dof: bg {chi2_dof[bg_flat].mean():.3f}, "
          f"blob {chi2_dof[blob_flat].mean():.3f} | "
          f"bins inside tau: {100 * within.mean():.2f}%")

    mw, pw, grid = render_width_test(
        stem, data, S, EM_1, EM_2a, x2a, bg_flat, blob_flat,
        OUTPUT_DIR / f"{stem}_width_test.png")
    inb = blob_flat & np.isfinite(mw)
    exc = mw[inb] - np.interp(x2a[inb], grid, pw)
    print(f"  width test: mixture predicts an arch spanning "
          f"{pw.min():.3f}-{pw.max():.3f} GHz "
          f"(+{1000 * (pw.max() - pw.min()):.0f} MHz); measured blob width "
          f"excess median {np.median(exc):+.4f} GHz")

    em_vca, _ = bd.vca(S, num_endmembers=2, seed=SEED)
    ab_vca = bd.estimate_abundances(S, em_vca)
    em_vca, ab_vca = render_vca(stem, data, EM_1, EM_2a, x2a, S, em_vca,
                                ab_vca, OUTPUT_DIR / f"{stem}_vca_compare.png")
    print(f"  VCA(2) vs solved: EM_1 angle "
          f"{spectral_angle(EM_1, em_vca[0]):.1f} deg, EM_2 angle "
          f"{spectral_angle(EM_2a, em_vca[1]):.1f} deg | abundance r = "
          f"{np.corrcoef(x2a, ab_vca[:, 1])[0, 1]:.4f}")

    pd.DataFrame({"shift_ghz": data["shift_freq"], "EM_1": EM_1,
                  "EM_2_x2max": EM_2a, "EM_2_nonneg": EM_2b,
                  "EM_2_vca_A": em_vca[0], "EM_2_vca_B": em_vca[1],
                  "u": u}).to_csv(
        OUTPUT_DIR / f"{stem}_endmembers.csv", index=False)


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or [DEFAULT_TARGET]
    for path in sorted(DATA_DIR.glob("*.csv")):
        if not any(w.lower() in path.name.lower() for w in want):
            continue
        print(f"[{path.name}]")
        try:
            process(path)
        except Exception:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
