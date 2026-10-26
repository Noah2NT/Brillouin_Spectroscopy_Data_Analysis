"""Per-pixel fit of TWO free Lorentzians vs ONE on the hydrogel scan.

lorentzian_plus_background_fit showed a single Lorentzian leaves D/dof ~ 14
even on pure-background pixels of Hydrogl1: the measured line is not one
Lorentzian. That script fixed it with an empirical background shape; here the
question is whether a purely parametric alternative does the same job:

    1L   mu = L(nu; a, m, g) + c                       4 params
    2L   mu = L(nu; a1, m1, g1) + L(nu; a2, m2, g2) + c  7 params

Both use the Poisson NLL on counts/alpha (shot noise with camera gain alpha,
see lorentzian_plus_background_fit). Reconstruction error is the Poisson
deviance per dof; the 1L -> 2L drop dD is chi^2(3) under the null, so
dD > 7.81 means the second Lorentzian is significant at 5%.

Outputs (outputs/two_lorentzian/): fit_examples.png (fits + measured−fit
residuals for background / blob / worst pixels), maps.png (D/dof maps, dD,
per-class histograms, peak positions), pixels.csv.
"""

from pathlib import Path
import importlib.util
import sys, os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_fitting as bf


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bge = _load("background_endmember")
tms = _load("two_material_solve")
lpb = _load("lorentzian_plus_background_fit")

DATA_PATH = Path("data/csv/Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv")
OUTPUT_DIR = Path("outputs") / "two_lorentzian"
LRT_CUT = 7.81          # chi^2(3) at 5%: three extra parameters
N_EXAMPLES = 4
CLASS_NAMES = lpb.CLASS_NAMES
ALPHA = [1.0]           # camera gain, set in main; used for the ±1σ band


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def model2(nu, p):
    """p = [a1, m1, g1, a2, m2, g2, c]."""
    return bf.lorentzian(nu, *p[:3], 0.0) + bf.lorentzian(nu, *p[3:6], p[6])


def _jac2(nu, p):
    cols = []
    for a, m, g in (p[:3], p[3:6]):
        u = (nu - m) / g
        d = 1.0 + u ** 2
        cols += [1.0 / d, a * 2 * u / (g * d ** 2), a * 2 * u ** 2 / (g * d ** 2)]
    cols.append(np.ones_like(nu))
    return np.column_stack(cols)


# On unit-normalised counts the NLL is O(1) and L-BFGS-B's default ftol
# (~2e-9 relative) stops ~50 deviance units short of the optimum; tighten it.
OPTS = dict(ftol=1e-13, gtol=1e-9, maxiter=5000)


def fit(y, nu, alpha, p0, bounds, model, jac):
    def f(p):
        mu = np.clip(model(nu, p), 1e-9, None)
        nll = (mu - y * np.log(mu)).sum() / alpha
        return nll, ((1.0 - y / mu) / alpha) @ jac(nu, p)
    r = minimize(f, p0, jac=True, method="L-BFGS-B", bounds=bounds, options=OPTS)
    return r.x, float(r.fun)


def fit_all(y, nu, alpha):
    N, B = y.shape
    lo, hi, span = float(nu.min()), float(nu.max()), float(nu.max() - nu.min())
    b1 = [(0, None), (lo, hi), (1e-3, span), (0, None)]
    b2 = [(0, None), (lo, hi), (1e-3, span)] * 2 + [(0, None)]
    _, p_ols = bf.fit_all_spectra(nu, y)
    p1s = np.zeros((N, 4)); p2s = np.zeros((N, 7)); D = np.zeros((N, 2))
    for i in range(N):
        yi = y[i]
        tot = max(yi.sum(), 1.0)                     # fit on O(1) counts
        ys = yi / tot
        p0 = np.array([max(p_ols[i, 0], 0) / tot, np.clip(p_ols[i, 1], lo, hi),
                       np.clip(p_ols[i, 2], 1e-3, span), max(p_ols[i, 3], 0) / tot])
        p1, _ = fit(ys, nu, alpha, p0, b1, lpb.model, lambda n, p: lpb._jac_cols(n, p, None))
        a, m, g, c = p1
        # ponytail: three starts (second peak right / left / broad underneath)
        # guard the obvious local minima; grid if "worst remaining" looks stuck.
        starts = [[0.7 * a, m, g, 0.3 * a, min(m + 0.3, hi), g, c],
                  [0.7 * a, m, g, 0.3 * a, max(m - 0.3, lo), g, c],
                  [0.8 * a, m, 0.7 * g, 0.2 * a, m, min(2.5 * g, span), c]]
        p2, _ = min((fit(ys, nu, alpha, np.array(s), b2, model2, _jac2) for s in starts),
                    key=lambda t: t[1])
        if p2[1] > p2[4]:                             # order peaks by centre
            p2 = np.concatenate([p2[3:6], p2[:3], p2[6:]])
        p1 = p1 * [tot, 1, 1, tot]
        p2 = p2 * [tot, 1, 1, tot, 1, 1, tot]
        p1s[i], p2s[i] = p1, p2
        D[i] = [lpb.deviance(yi, bf.lorentzian(nu, *p1), alpha),
                lpb.deviance(yi, model2(nu, p2), alpha)]
    return dict(p1=p1s, p2=p2s, D=D, D_dof=D / np.array([B - 4, B - 7]),
                dD=D[:, 0] - D[:, 1])


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def render_examples(y, nu, R, cls, out_path):
    D2 = R["D_dof"][:, 1]
    rng = np.random.default_rng(0)
    rows = []
    for k, name in ((2, "background"), (1, "hydrogel blob")):
        idx = np.flatnonzero(cls == k)
        rows.append((f"typical {name} px", rng.choice(idx, N_EXAMPLES, replace=False)))
    rows.append(("worst remaining (largest D/dof after 2L)", np.argsort(-D2)[:N_EXAMPLES]))

    fig, axes = plt.subplots(2 * len(rows), N_EXAMPLES,
                             figsize=(4.2 * N_EXAMPLES, 4.4 * len(rows)),
                             gridspec_kw=dict(height_ratios=[3, 1.2] * len(rows)),
                             constrained_layout=True)
    for r, (label, idx) in enumerate(rows):
        for c, i in enumerate(idx):
            ax, axr = axes[2 * r, c], axes[2 * r + 1, c]
            p1, p2 = R["p1"][i], R["p2"][i]
            f1, f2 = bf.lorentzian(nu, *p1), model2(nu, p2)
            ax.plot(nu, y[i], ".", color="0.3", ms=3, label="counts")
            ax.plot(nu, f1, color="0.6", lw=1.2, label="1 Lorentzian")
            ax.plot(nu, f2, color="k", lw=1.6, label="2 Lorentzians")
            ax.plot(nu, bf.lorentzian(nu, *p2[:3], p2[6]), "--", color="tab:red", lw=1, label="L₁ + c")
            ax.plot(nu, bf.lorentzian(nu, *p2[3:6], p2[6]), "--", color="tab:blue", lw=1, label="L₂ + c")
            ax.set_title(
                f"px {i} [{CLASS_NAMES[cls[i]]}]  dD={R['dD'][i]:.0f}\n"
                f"D/dof {R['D_dof'][i, 0]:.2f}→{D2[i]:.2f}   ν₁ {p2[1]:.2f} ν₂ {p2[4]:.2f} GHz\n"
                f"1L ν {p1[1]:.3f} γ {p1[2]:.2f}", fontsize=8)
            ax.grid(alpha=0.3); ax.tick_params(labelbottom=False)
            sig = np.sqrt(ALPHA[0] * np.clip(f2, 1, None))
            axr.plot(nu, y[i] - f1, color="0.6", lw=1)
            axr.plot(nu, y[i] - f2, color="k", lw=1.2)
            axr.fill_between(nu, -sig, sig, color="tab:orange", alpha=0.25, lw=0)
            axr.axhline(0, color="k", lw=0.5)
            axr.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(label, fontsize=9); axr.set_ylabel("measured − fit", fontsize=8)
            if r == len(rows) - 1:
                axr.set_xlabel("Shift (GHz)")
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle(f"{DATA_PATH.stem}: 1 vs 2 Lorentzians (Poisson fit; "
                 "orange band = ±1σ shot noise)", fontsize=11)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_maps(data, R, cls, out_path):
    yi, xi, shape = data["yi"], data["xi"], data["shape"]
    D = R["D_dof"]
    vmax = float(np.nanpercentile(D[:, 0], 98))
    nu_dom = np.where(R["p2"][:, 0] >= R["p2"][:, 3], R["p2"][:, 1], R["p2"][:, 4])
    nlim = (float(np.nanpercentile(R["p1"][:, 1], 1)), float(np.nanpercentile(R["p1"][:, 1], 99)))
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), constrained_layout=True)
    panels = [(D[:, 0], "D/dof, 1 Lorentzian", "magma", 0, vmax),
              (D[:, 1], "D/dof, 2 Lorentzians", "magma", 0, vmax),
              (R["dD"], "dD = D(1L) − D(2L)", "viridis", None, None),
              (R["p1"][:, 1], "ν, 1 Lorentzian (GHz)", "coolwarm", *nlim),
              (nu_dom, "ν of dominant peak, 2L (GHz)", "coolwarm", *nlim),
              (R["p2"][:, 1], "ν₁ (lower peak), 2L", "coolwarm", None, None),
              (R["p2"][:, 4], "ν₂ (upper peak), 2L", "coolwarm", None, None)]
    for ax, (v, title, cmap, vmin, vmax_) in zip(axes.ravel(), panels):
        im = ax.imshow(bge.scatter_map(v, yi, xi, shape), origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax_)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1, 3]
    hi = np.nanpercentile(D, 99)
    bins = np.linspace(0, max(hi, 1.5), 60)
    for k, col in ((2, "tab:blue"), (1, "tab:red")):
        sel = cls == k
        ax.hist(np.clip(D[sel, 0], 0, bins[-1]), bins, alpha=0.35, color=col,
                label=f"{CLASS_NAMES[k]} 1L (med {np.median(D[sel, 0]):.1f})")
        ax.hist(np.clip(D[sel, 1], 0, bins[-1]), bins, alpha=0.7, color=col,
                histtype="step", lw=1.6,
                label=f"{CLASS_NAMES[k]} 2L (med {np.median(D[sel, 1]):.2f})")
    ax.axvline(1, color="k", ls=":", lw=0.8)
    ax.set_xlabel("D/dof"); ax.set_title("reconstruction error per class", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(DATA_PATH.stem, fontsize=11)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────

def _selfcheck():
    rng = np.random.default_rng(0)
    nu = np.linspace(2.0, 8.0, 112)
    truth = np.array([3000.0, 5.3, 0.3, 1500.0, 5.9, 0.4, 30.0])
    y = rng.poisson(model2(nu, truth)).astype(float)
    R = fit_all(y[None, :], nu, 1.0)
    p = R["p2"][0]
    assert lpb.deviance(y, model2(nu, p), 1.0) <= lpb.deviance(y, model2(nu, truth), 1.0) + 1e-6, p
    assert abs(p[1] - 5.3) < 0.05 and abs(p[4] - 5.9) < 0.05, p
    assert R["D_dof"][0, 1] < 1.5, R["D_dof"]
    print(f"selfcheck ok: nu1={p[1]:.3f} nu2={p[4]:.3f} D/dof={R['D_dof'][0, 1]:.2f}")


def main():
    _selfcheck()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = bge.load_cube(DATA_PATH)
    nu = data["shift_freq"]
    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], nu, data["yi"],
                                      data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, nu, data["yi"], data["xi"], data["shape"])
    bg_mask, cell_mask, _, _ = bge.find_background(data, bfs_raw, bfs_sm)
    yi, xi = data["yi"], data["xi"]
    cls = np.zeros(len(yi), dtype=int)
    cls[cell_mask[yi, xi]] = 1
    cls[bg_mask[yi, xi]] = 2
    y = 2.0 * data["folded_raw"]
    alpha, _, r2 = tms.fit_alpha(y[cls == 2])
    ALPHA[0] = alpha
    print(f"{len(yi)} px | bg {int((cls == 2).sum())} blob {int((cls == 1).sum())} "
          f"| alpha={alpha:.2f} R2={r2:.3f}")

    R = fit_all(y, nu, alpha)
    render_examples(y, nu, R, cls, OUTPUT_DIR / "fit_examples.png")
    render_maps(data, R, cls, OUTPUT_DIR / "maps.png")
    p2 = R["p2"]
    pd.DataFrame(dict(
        yi=yi, xi=xi, cls=CLASS_NAMES[cls],
        nu_1L=R["p1"][:, 1], gamma_1L=R["p1"][:, 2],
        a1=p2[:, 0], nu1=p2[:, 1], gamma1=p2[:, 2],
        a2=p2[:, 3], nu2=p2[:, 4], gamma2=p2[:, 5], c=p2[:, 6],
        Ddof_1L=R["D_dof"][:, 0], Ddof_2L=R["D_dof"][:, 1], dD=R["dD"],
    )).to_csv(OUTPUT_DIR / "pixels.csv", index=False)

    for k in (2, 1):
        s = cls == k
        print(f"  {CLASS_NAMES[k]:11s} median D/dof 1L {np.median(R['D_dof'][s, 0]):6.2f}"
              f" -> 2L {np.median(R['D_dof'][s, 1]):5.2f} | "
              f"{100 * (R['dD'][s] > LRT_CUT).mean():.0f}% px significant | "
              f"nu1 {np.median(p2[s, 1]):.3f} nu2 {np.median(p2[s, 4]):.3f} | "
              f"gamma1 {np.median(p2[s, 2]):.2f} gamma2 {np.median(p2[s, 5]):.2f}")


if __name__ == "__main__":
    main()
