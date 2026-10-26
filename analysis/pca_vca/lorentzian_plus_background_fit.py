"""Per-pixel fit of ONE flexible Lorentzian + x * background endmember.

The pipeline's BFS comes from a single 4-parameter Lorentzian fitted by
ordinary least squares. Inside a cell the spectrum is cell signal plus medium,
so one curve is pulled toward the medium peak. Here every pixel is fitted with

    mu(nu) = L(nu; amp, centre, gamma, c) + x * EM_bg(nu)

where EM_bg is the co-added confident-background endmember from
background_endmember.py (unit-sum shape), and x >= 0 is how many background
counts the pixel carries.

LOSS. Raw data are integer photon counts. A photon-transfer curve on the
background pixels gives var = alpha * signal with no read-noise floor
(two_material_solve.fit_alpha), so the noise is shot noise with a camera gain
alpha. The right loss is therefore the Poisson negative log-likelihood on
counts/alpha (a scaled Poisson, exact when alpha == 1).

Three fits per pixel share the same Lorentzian so only the extra term differs:
    ols1  L                OLS  (bf.fit_all_spectra, the current method)
    poi1  L                Poisson NLL (isolates the effect of the loss)
    poi2  L + x*EM_bg      Poisson NLL (the upgrade)

Fit quality is the Poisson deviance D (chi^2 analogue): D/dof ~ 1 means the
fit is as good as shot noise allows. poi1 and poi2 are nested with one extra
parameter, so dD = D_poi1 - D_poi2 > 3.84 is a significant improvement at 5%.

Per sample: <stem>_fit_examples.png (most improved / unchanged / worst
remaining pixels), <stem>_maps.png, <stem>_pixels.csv. Across samples:
summary.csv and summary.png.
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

DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "lorentzian_plus_background"
LRT_CUT = 3.84          # chi^2(1) at 5%: one extra parameter must buy this much
N_EXAMPLES = 4
CLASS_NAMES = np.array(["uncertain", "cell", "background"])


# ─────────────────────────────────────────────
# Loss, deviance, model
# ─────────────────────────────────────────────

def deviance(y, mu, alpha):
    """Poisson deviance 2*sum[y log(y/mu) - (y - mu)] / alpha; y log y = 0 at y=0."""
    mu = np.clip(mu, 1e-9, None)
    ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * (ylogy - (y - mu)).sum() / alpha)


def model(nu, p, em=None):
    """p = [amp, centre, gamma, c] or [amp, centre, gamma, c, x]."""
    mu = bf.lorentzian(nu, *p[:4])
    return mu if em is None else mu + p[4] * em


def _jac_cols(nu, p, em):
    amp, m, g, _ = p[:4]
    u = (nu - m) / g
    d = 1.0 + u ** 2
    cols = [1.0 / d, amp * 2 * u / (g * d ** 2), amp * 2 * u ** 2 / (g * d ** 2),
            np.ones_like(nu)]
    if em is not None:
        cols.append(em)
    return np.column_stack(cols)


def fit_poisson(y, nu, alpha, p0, bounds, em=None):
    def f(p):
        mu = np.clip(model(nu, p, em), 1e-9, None)
        nll = (mu - y * np.log(mu)).sum() / alpha
        grad = ((1.0 - y / mu) / alpha) @ _jac_cols(nu, p, em)
        return nll, grad

    r = minimize(f, p0, jac=True, method="L-BFGS-B", bounds=bounds)
    return r.x, float(r.fun)


def fit_all(y, nu, em, alpha):
    """Return dict of per-pixel params, deviance/dof and dD for the 3 models."""
    N, B = y.shape
    lo, hi, span = float(nu.min()), float(nu.max()), float(nu.max() - nu.min())
    b1 = [(0, None), (lo, hi), (1e-3, span), (0, None)]
    b2 = b1 + [(0, None)]

    _, p_ols = bf.fit_all_spectra(nu, y)                 # the current method
    p_poi1 = np.zeros((N, 4)); p_poi2 = np.zeros((N, 5))
    D = np.zeros((N, 3))
    scale = np.array([1.0, 0, 0, 1.0, 1.0])       # which params carry counts
    for i in range(N):
        yi = y[i]
        # Fit on y/tot so amp, c, x are O(1) like nu and gamma: the Poisson
        # NLL has the same argmin under that rescaling, and L-BFGS-B does not
        # cope with parameters 1e4 apart in scale.
        tot = max(yi.sum(), 1.0)
        ys = yi / tot
        p0 = np.array([max(p_ols[i, 0], 0.0) / tot, np.clip(p_ols[i, 1], lo, hi),
                       np.clip(p_ols[i, 2], 1e-3, span), max(p_ols[i, 3], 0.0) / tot])
        p1, _ = fit_poisson(ys, nu, alpha, p0, b1)
        # ponytail: two starts guard the x/amp local minimum; grid if the
        # "worst remaining" panels show mis-starts.
        starts = [np.append(p1, 0.0),
                  np.array([0.2 * p1[0], p1[1], p1[2], 0.0, 0.8])]
        fits = [fit_poisson(ys, nu, alpha, s, b2, em) for s in starts]
        p2, _ = min(fits, key=lambda t: t[1])
        p1 = p1 * np.where(scale[:4] > 0, tot, 1.0)
        p2 = p2 * np.where(scale > 0, tot, 1.0)
        p_poi1[i], p_poi2[i] = p1, p2
        D[i] = [deviance(yi, model(nu, p_ols[i]), alpha),
                deviance(yi, model(nu, p1), alpha),
                deviance(yi, model(nu, p2, em), alpha)]
    dof = np.array([B - 4, B - 4, B - 5])
    return dict(p_ols=p_ols, p_poi1=p_poi1, p_poi2=p_poi2, D=D,
                D_dof=D / dof, dD=D[:, 1] - D[:, 2])


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def pick_examples(R, D2, cls):
    dD = R["dD"]
    # Half overall, half in-cell: the overall winners are background pixels
    # (EM_bg is their true shape), and the cell effect deserves its own slots.
    half = N_EXAMPLES // 2
    cell = np.flatnonzero(cls == 1)
    improved = np.argsort(-dD)[:half]
    if cell.size:
        top_cell = cell[np.argsort(-dD[cell])]
        improved = np.concatenate([improved,
                                   [i for i in top_cell if i not in improved][:N_EXAMPLES - half]])
    improved = improved[:N_EXAMPLES]
    unchanged = np.flatnonzero(dD < LRT_CUT)
    if unchanged.size:
        unchanged = unchanged[np.argsort(np.abs(D2[unchanged] - np.median(D2)))]
        label = f"unchanged (dD < {LRT_CUT}, typical D/dof)"
    else:   # every pixel improved significantly; show the least changed
        unchanged = np.argsort(dD)
        label = f"least changed (no px with dD < {LRT_CUT})"
    unchanged = unchanged[:N_EXAMPLES]
    worst = np.argsort(-D2)[:N_EXAMPLES]
    return [("most improved (largest dD; 2 overall + 2 in-cell)", improved),
            (label, unchanged),
            ("worst remaining (largest D/dof after)", worst)]


def render_examples(stem, y, nu, em, R, cls, out_path):
    D2 = R["D_dof"][:, 2]
    rows = pick_examples(R, D2, cls)
    fig, axes = plt.subplots(3, N_EXAMPLES, figsize=(4.2 * N_EXAMPLES, 10.5),
                             constrained_layout=True)
    for r, (label, idx) in enumerate(rows):
        for c, i in enumerate(idx):
            ax = axes[r, c]
            p2 = R["p_poi2"][i]
            ax.plot(nu, y[i], ".", color="0.3", ms=3, label="counts")
            ax.plot(nu, model(nu, R["p_ols"][i]), color="0.6", lw=1.2,
                    label="single Lorentzian (OLS)")
            ax.plot(nu, model(nu, p2, em), color="k", lw=1.6,
                    label="Lorentzian + x·EM_bg")
            ax.plot(nu, model(nu, p2), "--", color="tab:red", lw=1,
                    label="Lorentzian part")
            ax.plot(nu, p2[4] * em, "--", color="tab:blue", lw=1,
                    label="x·EM_bg part")
            ax.set_title(
                f"px {i} [{CLASS_NAMES[cls[i]]}]  dD={R['dD'][i]:.1f}\n"
                f"D/dof {R['D_dof'][i, 0]:.2f}→{D2[i]:.2f}   "
                f"ν {R['p_ols'][i, 1]:.3f}→{p2[1]:.3f} GHz   "
                f"x/Σy={p2[4] / max(y[i].sum(), 1):.2f}", fontsize=8)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            if r == 2:
                ax.set_xlabel("Shift (GHz)")
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle(stem, fontsize=11)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_maps(stem, data, y, R, cls, out_path):
    yi, xi, shape = data["yi"], data["xi"], data["shape"]
    xfrac = R["p_poi2"][:, 4] / np.maximum(y.sum(axis=1), 1)
    dnu = R["p_poi2"][:, 1] - R["p_ols"][:, 1]
    dnu_lim = float(np.nanpercentile(np.abs(dnu), 98)) or 1e-3
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    panels = [(R["dD"], "dD = D(single) − D(Lorentzian+EM)", "viridis", None, None),
              (xfrac, "background fraction x/Σy", "cividis", 0, 1),
              (dnu, "Δν (GHz): new − OLS single", "coolwarm", -dnu_lim, dnu_lim)]
    for ax, (v, title, cmap, vmin, vmax) in zip(axes[0], panels):
        im = ax.imshow(bge.scatter_map(v, yi, xi, shape), origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

    for ax, k, name in zip(axes[1, :2], (2, 1), ("background", "cell")):
        sel = cls == k
        if sel.any():
            # Axis from the Poisson-single range: OLS with c < 0 predicts
            # negative counts and its deviance runs to 100+, off to the right.
            hi = np.nanpercentile(R["D_dof"][sel, 1:], 99)
            bins = np.linspace(0, max(hi, 1.5), 50)
            for j, (lab, col) in enumerate([("OLS single", "0.5"),
                                            ("Poisson single", "tab:orange"),
                                            ("Poisson L + x·EM", "tab:blue")]):
                ax.hist(np.clip(R["D_dof"][sel, j], 0, bins[-1]), bins=bins,
                        alpha=0.55, color=col, label=lab)
            ax.axvline(1, color="k", lw=0.8, ls=":")
        ax.set_title(f"D/dof, {name} pixels (n={int(sel.sum())})", fontsize=10)
        ax.set_xlabel("D/dof"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    for k, col in ((2, "tab:blue"), (0, "0.6"), (1, "tab:red")):
        sel = cls == k
        ax.scatter(R["p_ols"][sel, 1], R["p_poi2"][sel, 1], s=5, alpha=0.5,
                   color=col, label=CLASS_NAMES[k])
    lim = (np.nanmin(R["p_ols"][:, 1]), np.nanmax(R["p_ols"][:, 1]))
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlabel("ν, OLS single (GHz)"); ax.set_ylabel("ν, Lorentzian + x·EM (GHz)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(stem, fontsize=11)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_summary(summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    x = np.arange(len(summary)); w = 0.27
    for j, (col, lab, c) in enumerate([("Ddof_cell_ols1", "OLS single", "0.5"),
                                       ("Ddof_cell_poi1", "Poisson single", "tab:orange"),
                                       ("Ddof_cell_poi2", "Poisson L + x·EM", "tab:blue")]):
        axes[0].bar(x + (j - 1) * w, summary[col], w, color=c, label=lab)
    axes[0].axhline(1, color="k", ls=":", lw=0.8)
    axes[0].set_ylabel("median D/dof, cell pixels"); axes[0].legend(fontsize=8)
    for j, (col, lab, c) in enumerate([("pct_sig_bg", "background", "tab:blue"),
                                       ("pct_sig_cell", "cell", "tab:red")]):
        axes[1].bar(x + (j - 0.5) * w, summary[col], w, color=c, label=lab)
    axes[1].set_ylabel("% pixels with dD > 3.84"); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([s[:22] for s in summary["sample"]], rotation=60,
                           ha="right", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────

def process(path):
    stem = path.stem
    data = bge.load_cube(path)
    nu = data["shift_freq"]
    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], nu, data["yi"],
                                      data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, nu, data["yi"], data["xi"],
                                     data["shape"])
    bg_mask, cell_mask, _, info = bge.find_background(data, bfs_raw, bfs_sm)
    if int(bg_mask.sum()) < bge.MIN_BG_PIXELS:
        print(f"  [skip] fewer than {bge.MIN_BG_PIXELS} confident background px")
        return None

    yi, xi = data["yi"], data["xi"]
    cls = np.zeros(len(yi), dtype=int)
    cls[cell_mask[yi, xi]] = 1
    cls[bg_mask[yi, xi]] = 2

    y = 2.0 * data["folded_raw"]                 # Stokes + anti-Stokes counts
    _, summed, _ = bge.build_endmember(data, bg_mask)
    em = summed / summed.sum()
    alpha, intercept, r2 = tms.fit_alpha(y[cls == 2])
    print(f"  {len(yi)} px | bg {int((cls == 2).sum())} cell "
          f"{int((cls == 1).sum())} | alpha={alpha:.2f} intercept={intercept:.1f} "
          f"R2={r2:.3f}")

    R = fit_all(y, nu, em, alpha)
    cell, bg = cls == 1, cls == 2
    xfrac = R["p_poi2"][:, 4] / np.maximum(y.sum(axis=1), 1)
    dnu = R["p_poi2"][:, 1] - R["p_ols"][:, 1]
    sig = R["dD"] > LRT_CUT

    render_examples(stem, y, nu, em, R, cls, OUTPUT_DIR / f"{stem}_fit_examples.png")
    render_maps(stem, data, y, R, cls, OUTPUT_DIR / f"{stem}_maps.png")
    pd.DataFrame(dict(
        yi=yi, xi=xi, cls=CLASS_NAMES[cls],
        nu_ols=R["p_ols"][:, 1], gamma_ols=R["p_ols"][:, 2],
        nu_poi1=R["p_poi1"][:, 1], nu_poi2=R["p_poi2"][:, 1],
        gamma_poi2=R["p_poi2"][:, 2], amp_poi2=R["p_poi2"][:, 0],
        c_poi2=R["p_poi2"][:, 3], x_poi2=R["p_poi2"][:, 4], x_frac=xfrac,
        Ddof_ols1=R["D_dof"][:, 0], Ddof_poi1=R["D_dof"][:, 1],
        Ddof_poi2=R["D_dof"][:, 2], dD=R["dD"],
    )).to_csv(OUTPUT_DIR / f"{stem}_pixels.csv", index=False)

    def med(v, m):
        return float(np.median(v[m])) if m.any() else np.nan

    row = dict(sample=stem, alpha=alpha, alpha_r2=r2, n_px=len(yi),
               n_cell=int(cell.sum()), n_bg=int(bg.sum()))
    for j, tag in enumerate(("ols1", "poi1", "poi2")):
        row[f"Ddof_all_{tag}"] = float(np.median(R["D_dof"][:, j]))
        row[f"Ddof_cell_{tag}"] = med(R["D_dof"][:, j], cell)
        row[f"Ddof_bg_{tag}"] = med(R["D_dof"][:, j], bg)
    row.update(
        pct_sig_all=100 * sig.mean(),
        pct_sig_cell=100 * sig[cell].mean() if cell.any() else np.nan,
        pct_sig_bg=100 * sig[bg].mean() if bg.any() else np.nan,
        dD_cell_median=med(R["dD"], cell),
        dnu_cell_median=med(dnu, cell), dnu_bg_median=med(dnu, bg),
        xfrac_cell_median=med(xfrac, cell), xfrac_bg_median=med(xfrac, bg),
    )
    print(f"  D/dof cell: ols {row['Ddof_cell_ols1']:.2f} -> poi1 "
          f"{row['Ddof_cell_poi1']:.2f} -> poi2 {row['Ddof_cell_poi2']:.2f} | "
          f"sig cell {row['pct_sig_cell']:.0f}% bg {row['pct_sig_bg']:.0f}% | "
          f"dnu cell {row['dnu_cell_median']:+.3f} GHz | x/sum(y) bg "
          f"{row['xfrac_bg_median']:.2f} cell {row['xfrac_cell_median']:.2f}")
    return row


def _selfcheck():
    rng = np.random.default_rng(0)
    nu = np.linspace(2.0, 8.0, 112)
    em = bf.lorentzian(nu, 1.0, 5.0, 0.3, 0.02); em /= em.sum()
    truth = np.array([3000.0, 5.4, 0.35, 20.0, 8000.0])
    y = rng.poisson(model(nu, truth, em)).astype(float)
    R = fit_all(y[None, :], nu, em, 1.0)
    p = R["p_poi2"][0]
    # The optimizer must reach at least the likelihood of the truth; how far
    # the fit sits from the truth is then noise, not a fitting failure.
    assert deviance(y, model(nu, p, em), 1.0) <= deviance(y, model(nu, truth, em), 1.0) + 1e-6, p
    assert abs(p[1] - 5.4) < 0.05, p
    assert R["D_dof"][0, 2] < 1.5, R["D_dof"]
    print(f"selfcheck ok: nu={p[1]:.3f} x={p[4]:.0f} D/dof={R['D_dof'][0, 2]:.2f}")


def main():
    _selfcheck()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        print(f"[{path.name}]")
        try:
            row = process(path)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            continue
        if row:
            rows.append(row)
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
        render_summary(summary, OUTPUT_DIR / "summary.png")
        cols = ["sample", "alpha", "Ddof_cell_ols1", "Ddof_cell_poi1",
                "Ddof_cell_poi2", "Ddof_bg_poi2", "pct_sig_cell", "pct_sig_bg",
                "dnu_cell_median", "xfrac_cell_median", "xfrac_bg_median"]
        print("\n" + summary[cols].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
