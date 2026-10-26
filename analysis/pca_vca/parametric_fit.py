"""Second endmember as a parametric Lorentzian, fitted per pixel.

The fixed-endmember model was rejected by the linewidth test: on the hydrogel
blob, peak position AND width both climb with depth into the blob, and no
linear combination of two fixed spectra can follow a peak that slides and
broadens. So keep the linear mixture for the medium -- partial-volume mixing
at the boundary is genuinely real -- and let the second component be a
Lorentzian carrying its own position and width in every pixel:

    s_p = (1 - x2_p) * EM_1  +  x2_p * L(nu_p, Gamma_p)

Three free parameters per pixel instead of one, and each one is physical: x2
is how much of the second material, nu its stiffness, Gamma its damping.

WHY THIS IS IDENTIFIABLE. EM_1 is fixed and known to high SNR, and it sits
0.33 GHz from the second peak. Where x2 -> 1 the fit is just one Lorentzian;
only the mid-range ring carries real x2/nu trade-off, and it is pinned from
both sides. Fits are weighted by the calibrated shot-noise sigma, so
chi2/dof = 1 means "as good as the noise allows" rather than an arbitrary
number in arbitrary units.

THE TRAP, AND THE TEST. Three parameters per pixel MUST fit better than one,
so a lower chi2 proves nothing by itself. The discriminating prediction is
about how nu and Gamma behave ACROSS pixels:

    two fixed materials  ->  nu and Gamma are CONSTANT (one second material)
    a graded material    ->  nu and Gamma VARY, smoothly, with position

Running the identical code on known-truth two-material simulation gives the
first case; running it on the hydrogel blob should give the second. That
comparison, not the chi2, is what the model is judged on -- a lesson from the
component-count work, where two errors survived every internal check and only
ground truth exposed them.

Spatial coherence (Moran's I) of the nu and Gamma maps guards against
overfitting: a real material property varies smoothly, salt-and-pepper means
the fit is chasing noise.
"""

from pathlib import Path
import importlib.util
import sys, os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
ms = _load("medium_subtraction")

DATA_DIR = Path("data/csv")
SIM_DIR = Path("outputs/simulated_data/simulated_water_cytoplasm")
OUTPUT_DIR = Path("outputs") / "parametric"
CONF_X2 = 0.25          # below this the Lorentzian's parameters are unpinned


def unit_lorentzian(nu, centre, gamma):
    """brillouin_fitting.lorentzian renormalised to unit area on the grid.

    Normalised by its own sum rather than analytically, so it lives on the
    same sum-to-one footing as EM_1 and the measured spectra. Without that the
    abundances would not add to one and x2 would silently absorb a scale.
    """
    y = bf.lorentzian(nu, 1.0, centre, max(gamma, 1e-4), 0.0)
    total = y.sum()
    return y / total if total > 0 else y


def fit_pixel(s, nu, EM_1, sigma, guess, bounds):
    """Weighted three-parameter fit for one pixel."""
    def resid(p):
        model = (1.0 - p[0]) * EM_1 + p[0] * unit_lorentzian(nu, p[1], p[2])
        return (s - model) / sigma

    out = least_squares(resid, guess, bounds=bounds, method="trf",
                        xtol=1e-10, ftol=1e-10, max_nfev=400)
    return out.x, float((out.fun ** 2).mean())


def run(path, tag):
    data = bge.load_cube(path)
    nu = data["shift_freq"]
    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], nu,
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, nu, data["yi"], data["xi"],
                                     data["shape"])
    bg_mask, blob_mask, _, _ = bge.find_background(data, bfs_raw, bfs_sm)
    EM_1, _, _ = bge.build_endmember(data, bg_mask)

    yi, xi = data["yi"], data["xi"]
    bg_flat, blob_flat = bg_mask[yi, xi], blob_mask[yi, xi]
    S = data["folded_norm"]
    counts = data["folded_raw"].sum(axis=1)
    alpha, _, _ = tms.fit_alpha(data["folded_raw"][bg_flat])

    # Initial guesses from the fixed-endmember solve: it already places x2 and
    # the rough second-peak position, so the non-linear fit starts near the
    # answer instead of hunting from nothing.
    u, x2_raw, _ = tms.solve(S, EM_1, counts, alpha)
    c0, x2_0 = tms.scale_x2max(u, x2_raw)
    EM_2_0 = EM_1 + c0 * u
    nu0 = float(nu[np.argmax(EM_2_0)])
    g0 = float(tms.peak_width(EM_2_0, nu)[1])
    bounds = ([0.0, float(nu.min()), 0.05], [1.0, float(nu.max()), 3.0])

    n = S.shape[0]
    par, chi2 = np.zeros((n, 3)), np.zeros(n)
    for p in range(n):
        model0 = (1 - x2_0[p]) * EM_1 + x2_0[p] * unit_lorentzian(nu, nu0, g0)
        sigma = np.sqrt(alpha * np.maximum(model0 * counts[p], 1.0)) / counts[p]
        guess = [float(np.clip(x2_0[p], 1e-3, 0.999)), nu0, g0]
        par[p], chi2[p] = fit_pixel(S[p], nu, EM_1, sigma, guess, bounds)

    x2, nu_p, gam_p = par[:, 0], par[:, 1], par[:, 2]

    # Only pixels holding a real second component have identifiable nu/Gamma:
    # where x2 ~ 0 the Lorentzian is multiplied by nothing and its parameters
    # are free to wander. Spread is quoted on the confident subset only.
    conf = x2 > CONF_X2
    if int(conf.sum()) < 20:
        # Fall back to the top fifth by abundance. A fixed cut assumes the
        # scan actually contains near-pure second-material pixels; on scans
        # where it does not, a relative cut still picks the pixels where the
        # Lorentzian carries the most weight and is therefore best pinned.
        cut = float(np.quantile(x2, 0.80))
        conf = x2 >= cut
        print(f"  [note] only {int((x2 > CONF_X2).sum())} px above x2>{CONF_X2}"
              f"; using top quintile (x2 >= {cut:.3f}) instead")
    shape = data["shape"]
    stats = dict(
        tag=tag, sample=path.stem, n_pixels=n, alpha=alpha,
        n_confident=int(conf.sum()),
        chi2_all=float(chi2.mean()), chi2_bg=float(chi2[bg_flat].mean()),
        chi2_blob=float(chi2[blob_flat].mean()) if blob_flat.any() else np.nan,
        nu_mean=float(nu_p[conf].mean()), nu_sd=float(nu_p[conf].std()),
        nu_p5=float(np.percentile(nu_p[conf], 5)),
        nu_p95=float(np.percentile(nu_p[conf], 95)),
        gam_mean=float(gam_p[conf].mean()), gam_sd=float(gam_p[conf].std()),
        gam_p5=float(np.percentile(gam_p[conf], 5)),
        gam_p95=float(np.percentile(gam_p[conf], 95)),
        moran_nu=float(ms.morans_i(nu_p[conf], yi[conf], xi[conf], shape)),
        moran_gamma=float(ms.morans_i(gam_p[conf], yi[conf], xi[conf], shape)),
    )
    return stats, dict(data=data, EM_1=EM_1, S=S, nu=nu, x2=x2, nu_p=nu_p,
                       gam_p=gam_p, chi2=chi2, conf=conf, bfs=bfs_raw,
                       bg=bg_flat, blob=blob_flat)


def to_map(v, yi, xi, shape, sel=None):
    g = np.full(shape, np.nan)
    if sel is None:
        g[yi, xi] = v
    else:
        g[yi[sel], xi[sel]] = v[sel]
    return g


def render(stats, R, out_path):
    shape = R["data"]["shape"]
    yi, xi, nu = R["data"]["yi"], R["data"]["xi"], R["nu"]
    conf, x2 = R["conf"], R["x2"]
    nu_p, gam_p, chi2 = R["nu_p"], R["gam_p"], R["chi2"]

    fig = plt.figure(figsize=(20, 14.5), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)

    for j, (v, nm, cm, sel) in enumerate([
            (R["bfs"], "BFS map (GHz)", "magma", "grid"),
            (x2, "x2   abundance", "viridis", None),
            (nu_p, "nu   peak position (GHz)", "plasma", conf),
            (gam_p, "Gamma   linewidth (GHz)", "cividis", conf)]):
        ax = fig.add_subplot(gs[0, j])
        is_grid = isinstance(sel, str)
        m = v if is_grid else to_map(v, yi, xi, shape, sel)
        im = ax.imshow(m, origin="lower", cmap=cm)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(nm, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(to_map(chi2, yi, xi, shape), origin="lower", cmap="magma",
                   vmin=0, vmax=float(np.nanpercentile(chi2, 98)))
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"chi2/dof per pixel (mean {stats['chi2_all']:.3f})",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    for mask, c, lab in ((R["bg"], "tab:blue", "background"),
                         (R["blob"], "tab:red", "second material")):
        if mask.any():
            ax.hist(chi2[mask], bins=40, color=c, alpha=0.6, label=lab)
    ax.axvline(1.0, color="k", lw=1.8, label="chi2/dof = 1")
    ax.set_xlabel("chi2 / dof"); ax.set_ylabel("pixels")
    ax.set_title("as good as the noise allows?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(x2[conf], gam_p[conf], s=7, alpha=0.45, color="tab:red")
    ax.axhline(gam_p[conf].mean(), color="k", ls="--", lw=1.2,
               label=f"mean {gam_p[conf].mean():.3f}")
    ax.set_xlabel("x2"); ax.set_ylabel("Gamma (GHz)")
    ax.set_title(f"THE TEST: linewidth vs abundance\n"
                 f"sd = {stats['gam_sd']:.4f} GHz", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    ax.scatter(x2[conf], nu_p[conf], s=7, alpha=0.45, color="tab:purple")
    ax.axhline(nu_p[conf].mean(), color="k", ls="--", lw=1.2,
               label=f"mean {nu_p[conf].mean():.3f}")
    ax.set_xlabel("x2"); ax.set_ylabel("nu (GHz)")
    ax.set_title(f"position vs abundance\nsd = {stats['nu_sd']:.4f} GHz",
                 fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    idx = np.flatnonzero(conf)
    picks = idx[np.linspace(0, len(idx) - 1, 3).round().astype(int)] \
        if len(idx) >= 3 else idx
    for j, p in enumerate(picks[:3]):
        ax = fig.add_subplot(gs[2, j])
        model = ((1 - x2[p]) * R["EM_1"]
                 + x2[p] * unit_lorentzian(nu, nu_p[p], gam_p[p]))
        ax.plot(nu, R["S"][p], lw=1.6, color="0.35", label="measured")
        ax.plot(nu, model, lw=1.3, ls="--", color="tab:red", label="fit")
        ax.plot(nu, R["S"][p] - model, lw=1.0, color="tab:green",
                label="residual")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_xlabel("Shift (GHz)")
        ax.set_title(f"px {p}   x2={x2[p]:.2f}  nu={nu_p[p]:.3f}  "
                     f"G={gam_p[p]:.3f}\nchi2/dof={chi2[p]:.2f}", fontsize=9)
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    txt = [
        stats["tag"].upper(), "",
        f"chi2/dof  all   {stats['chi2_all']:.3f}",
        f"          bg    {stats['chi2_bg']:.3f}",
        f"          2nd   {stats['chi2_blob']:.3f}", "",
        f"confident px (x2>{CONF_X2})  {stats['n_confident']}", "",
        f"nu     mean  {stats['nu_mean']:.3f} GHz",
        f"       sd    {stats['nu_sd']:.4f}",
        f"       p5-95 {stats['nu_p5']:.3f} - {stats['nu_p95']:.3f}", "",
        f"Gamma  mean  {stats['gam_mean']:.3f} GHz",
        f"       sd    {stats['gam_sd']:.4f}",
        f"       p5-95 {stats['gam_p5']:.3f} - {stats['gam_p95']:.3f}", "",
        f"Moran I  nu     {stats['moran_nu']:.3f}",
        f"         Gamma  {stats['moran_gamma']:.3f}",
        "",
        "constant nu/Gamma => two fixed materials",
        "varying  nu/Gamma => one graded material",
    ]
    ax.text(0.0, 1.0, chr(10).join(txt), va="top", ha="left",
            family="monospace", fontsize=11, transform=ax.transAxes)

    fig.suptitle(f"{stats['sample'][:58]}   |   parametric second endmember:  "
                 "s = (1-x2)*EM_1 + x2*L(nu, Gamma)", fontsize=13)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = []
    hy = [p for p in sorted(DATA_DIR.glob("*.csv")) if "Hydrogl1" in p.name]
    if hy:
        targets.append((hy[0], "hydrogel blob (real)"))
    sim = SIM_DIR / "half_water_half_cytoplasm_100s.csv"
    if sim.exists():
        targets.append((sim, "known truth: 2 fixed materials"))

    rows = []
    for path, tag in targets:
        print(f"[{tag}] {path.name}")
        try:
            stats, R = run(path, tag)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            continue
        rows.append(stats)
        print(f"  chi2/dof all {stats['chi2_all']:.3f} | bg "
              f"{stats['chi2_bg']:.3f} | 2nd {stats['chi2_blob']:.3f}")
        print(f"  nu    {stats['nu_mean']:.3f} +/- {stats['nu_sd']:.4f} GHz "
              f"(p5-95 {stats['nu_p5']:.3f}-{stats['nu_p95']:.3f}) "
              f"Moran {stats['moran_nu']:.3f}")
        print(f"  Gamma {stats['gam_mean']:.3f} +/- {stats['gam_sd']:.4f} GHz "
              f"(p5-95 {stats['gam_p5']:.3f}-{stats['gam_p95']:.3f}) "
              f"Moran {stats['moran_gamma']:.3f}")
        render(stats, R, OUTPUT_DIR / f"{path.stem}_parametric.png")
        pd.DataFrame({"pixel": np.arange(len(R["x2"])), "x2": R["x2"],
                      "nu_ghz": R["nu_p"], "gamma_ghz": R["gam_p"],
                      "chi2_dof": R["chi2"]}).to_csv(
            OUTPUT_DIR / f"{path.stem}_parametric.csv", index=False)

    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "parametric_summary.csv",
                                  index=False)
        print("\n" + pd.DataFrame(rows)[
            ["tag", "chi2_all", "chi2_blob", "nu_sd", "gam_sd",
             "moran_nu", "moran_gamma"]].to_string(index=False))


if __name__ == "__main__":
    main()
