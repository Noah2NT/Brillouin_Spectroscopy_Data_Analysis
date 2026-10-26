"""Figures for the 'solve for the second endmember' strategy on the hydrogel.

Four figures, one question each:
  x2_effect    what solving for x2 gives you, and what it tracks
  recon_error  where the two-material model fits and where it does not
  pixels_good  the best-fitting blob pixels, fit and residual shown
  pixels_bad   the worst-fitting blob pixels, same layout, so the failure
               mode is visible rather than described

Best and worst are chosen by chi2/dof WITHIN THE BLOB. Background pixels
always fit (they are EM_1 by construction) and would make every "good" example
trivial. Ranking inside the blob shows the model succeeding and failing on
the pixels it was actually built for.
"""

from pathlib import Path
import importlib.util
import sys, os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bge = _load("background_endmember")
tms = _load("two_material_solve")

DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "solve_showcase"
N_EXAMPLES = 4
TAU = 3.51          # union-bound 95% envelope for 112 bins


def to_map(v, yi, xi, shape):
    g = np.full(shape, np.nan)
    g[yi, xi] = v
    return g


def solve_hydrogel(path):
    """Mirror two_material_solve.process: two-pass solve with blended alpha."""
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
    bg, blob = bg_mask[yi, xi], blob_mask[yi, xi]
    S = data["folded_norm"]
    counts = data["folded_raw"].sum(axis=1)
    alpha, _, _ = tms.fit_alpha(data["folded_raw"][bg])

    u, x2r, _ = tms.solve(S, EM_1, counts, alpha)
    c, x2 = tms.scale_x2max(u, x2r)
    EM_2 = EM_1 + c * u
    top = x2 >= np.quantile(x2[blob], 0.90)
    alpha2 = (tms.fit_alpha_pure(data["folded_raw"][top])[0]
              if top.sum() >= 30 else alpha)
    u, x2r, hist = tms.solve(S, EM_1, counts, alpha, a2=alpha2, EM_2_ref=EM_2)
    c, x2 = tms.scale_x2max(u, x2r)
    EM_2 = EM_1 + c * u

    model = (1 - x2)[:, None] * EM_1[None, :] + x2[:, None] * EM_2[None, :]
    sig = tms.sigma_mix(x2, EM_1, EM_2, alpha, alpha2, counts)
    resid = S - model
    chi2 = ((resid / sig) ** 2).mean(axis=1)
    return dict(data=data, nu=nu, bfs=bfs_raw, bg=bg, blob=blob, S=S,
                EM_1=EM_1, EM_2=EM_2, x2=x2, model=model, sig=sig,
                resid=resid, chi2=chi2, abs_err=np.abs(resid).sum(axis=1),
                alpha=alpha, alpha2=alpha2, hist=hist)


REGIONS = (("bg", "tab:blue", "background"),
           ("other", "0.5", "uncertain"),
           ("blob", "tab:red", "blob"))


def masks(R):
    return {"bg": R["bg"], "blob": R["blob"], "other": ~(R["bg"] | R["blob"])}


# -------------------------------------------------
# Figure 1: the effect of solved x2
# -------------------------------------------------

def fig_x2(R, out):
    d, shape = R["data"], R["data"]["shape"]
    yi, xi, nu = d["yi"], d["xi"], R["nu"]
    bfs_flat = R["bfs"][yi, xi]
    M = masks(R)
    fig = plt.figure(figsize=(19, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(R["bfs"], origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8, label="GHz")
    ax.set_title("what was measured: BFS map", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(to_map(R["x2"], yi, xi, shape), origin="lower",
                   cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("what was solved: x2, hydrogel fraction", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    p1, w1 = tms.peak_width(R["EM_1"], nu)
    p2, w2 = tms.peak_width(R["EM_2"], nu)
    ax.plot(nu, R["EM_1"], lw=2.2, color="tab:blue",
            label=f"EM_1 background   {p1:.3f} GHz, width {w1:.3f}")
    ax.plot(nu, R["EM_2"], lw=2.2, color="tab:red",
            label=f"EM_2 solved   {p2:.3f} GHz, width {w2:.3f}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("norm. intensity")
    ax.set_title("the two endmembers", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    for k, c, lab in REGIONS:
        ax.scatter(bfs_flat[M[k]], R["x2"][M[k]], s=6, alpha=0.45, color=c,
                   label=lab)
    r = float(np.corrcoef(bfs_flat[M["blob"]], R["x2"][M["blob"]])[0, 1])
    ax.set_xlabel("measured BFS (GHz)"); ax.set_ylabel("solved x2")
    ax.set_title(f"x2 tracks stiffness   (blob r = {r:.3f})", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for k, c, lab in REGIONS:
        ax.hist(R["x2"][M[k]], bins=40, color=c, alpha=0.6, label=lab)
    ax.set_xlabel("x2"); ax.set_ylabel("pixels")
    ax.set_title("distribution by region", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    order = np.argsort(R["x2"])
    picks = order[np.linspace(0, len(order) - 1, 6).round().astype(int)]
    cm = plt.get_cmap("viridis")
    for p in picks:
        ax.plot(nu, R["model"][p], lw=1.5, color=cm(R["x2"][p]),
                label=f"x2 = {R['x2'][p]:.2f}")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("norm. intensity")
    ax.set_title("what the model produces as x2 rises", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Solving for the second endmember: the effect of x2",
                 fontsize=13)
    fig.savefig(out, dpi=125); plt.close(fig)


# -------------------------------------------------
# Figure 2: reconstruction error
# -------------------------------------------------

def fig_error(R, out):
    d, shape = R["data"], R["data"]["shape"]
    yi, xi = d["yi"], d["xi"]
    M = masks(R)
    fig = plt.figure(figsize=(19, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(to_map(R["abs_err"], yi, xi, shape), origin="lower",
                   cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8, label="sum |measured - fit|")
    ax.set_title("reconstruction error, absolute", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    hi = float(np.nanpercentile(R["chi2"], 98))
    im = ax.imshow(to_map(R["chi2"], yi, xi, shape), origin="lower",
                   cmap="magma", vmin=0, vmax=hi)
    fig.colorbar(im, ax=ax, shrink=0.8, label="chi2 / dof")
    ax.set_title("reconstruction error in units of noise"
                 + chr(10) + "(1.0 = as good as the noise allows)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    for k, c, lab in REGIONS:
        ax.hist(R["chi2"][M[k]], bins=45, color=c, alpha=0.6, label=lab)
    ax.axvline(1.0, color="k", lw=1.8, label="chi2/dof = 1")
    ax.set_xlabel("chi2 / dof"); ax.set_ylabel("pixels")
    ax.set_title(f"background {R['chi2'][M['bg']].mean():.2f}   |   "
                 f"blob {R['chi2'][M['blob']].mean():.2f}", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    for k, c, lab in REGIONS:
        ax.scatter(R["x2"][M[k]], R["chi2"][M[k]], s=6, alpha=0.45, color=c,
                   label=lab)
    ax.axhline(1.0, color="k", lw=1.2, ls="--")
    ax.set_xlabel("x2"); ax.set_ylabel("chi2 / dof")
    ax.set_title("error vs abundance: where does it fail?", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    blob = M["blob"]
    mean_res = R["resid"][blob].mean(axis=0)
    sd_res = R["resid"][blob].std(axis=0)
    ax.plot(R["nu"], mean_res, lw=2.0, color="k", label="mean blob residual")
    ax.fill_between(R["nu"], mean_res - sd_res, mean_res + sd_res,
                    color="0.6", alpha=0.4, label="+/- 1 sd across pixels")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Shift (GHz)")
    ax.set_title("what the model systematically misses\n"
                 "(a repeatable shape = model error; flat = noise)",
                 fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(R["hist"], lw=1.8, color="tab:purple")
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xlabel("iteration"); ax.set_ylabel("chi2 / dof")
    ax.set_title(f"convergence   (final {R['hist'][-1]:.3f})", fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle("Reconstruction error of the two-material model", fontsize=13)
    fig.savefig(out, dpi=125); plt.close(fig)


# -------------------------------------------------
# Figures 3 & 4: pixels that work, pixels that don't
# -------------------------------------------------

def fig_pixels(R, picks, title, verdict_colour, out):
    d = R["data"]
    yi, xi, nu = d["yi"], d["xi"], R["nu"]
    lo, hi = np.nanpercentile(R["bfs"], [2, 98])
    fig = plt.figure(figsize=(21, 4.6 * len(picks)), constrained_layout=True)
    gs = GridSpec(len(picks), 3, figure=fig, width_ratios=[1, 1.6, 1.6])

    for r, p in enumerate(picks):
        ax = fig.add_subplot(gs[r, 0])
        im = ax.imshow(R["bfs"], origin="lower", cmap="magma", vmin=lo, vmax=hi)
        ax.axhline(yi[p], color="red", lw=0.6, alpha=0.55)
        ax.axvline(xi[p], color="red", lw=0.6, alpha=0.55)
        ax.plot(xi[p], yi[p], marker="s", ms=12, mfc="none", mec="red", mew=2.2)
        fig.colorbar(im, ax=ax, shrink=0.75, label="GHz")
        ax.set_title(f"px {p}   (y={yi[p]}, x={xi[p]})   x2 = {R['x2'][p]:.3f}",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[r, 1])
        ax.plot(nu, R["S"][p], lw=1.8, color="0.3", label="measured")
        ax.plot(nu, (1 - R["x2"][p]) * R["EM_1"], lw=1.1, ls=":",
                color="tab:blue", label="(1 - x2) EM_1")
        ax.plot(nu, R["x2"][p] * R["EM_2"], lw=1.1, ls=":", color="tab:red",
                label="x2 EM_2")
        ax.plot(nu, R["model"][p], lw=1.6, ls="--", color="tab:purple",
                label="fit = sum")
        ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("norm. intensity")
        ax.set_title("measured vs the two-material fit", fontsize=10)
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[r, 2])
        res = R["resid"][p]
        ax.fill_between(nu, 0, res, color=verdict_colour, alpha=0.2)
        ax.plot(nu, res, lw=1.6, color="k")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("norm. intensity")
        ax.set_title(f"measured - fit   chi2/dof = {R['chi2'][p]:.2f}",
                     fontsize=10, color=verdict_colour)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.savefig(out, dpi=115); plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = [p for p in sorted(DATA_DIR.glob("*.csv")) if "Hydrogl1" in p.name][0]
    print(f"[{path.name}]")
    R = solve_hydrogel(path)
    blob_idx = np.flatnonzero(R["blob"])
    order = blob_idx[np.argsort(R["chi2"][blob_idx])]
    good, bad = order[:N_EXAMPLES], order[-N_EXAMPLES:][::-1]
    cb = R["chi2"][blob_idx]

    print(f"  blob chi2/dof: best {R['chi2'][good].round(2)}   "
          f"worst {R['chi2'][bad].round(2)}")
    print(f"  blob: {np.mean(cb < 1.5) * 100:.0f}% of px under 1.5, "
          f"{np.mean(cb > 3) * 100:.0f}% over 3, median {np.median(cb):.2f}")
    print(f"  bg chi2/dof {R['chi2'][R['bg']].mean():.3f} | "
          f"blob {cb.mean():.3f} | alpha {R['alpha']:.3f} a2 {R['alpha2']:.3f}")

    fig_x2(R, OUTPUT_DIR / "x2_effect.png")
    fig_error(R, OUTPUT_DIR / "recon_error.png")
    fig_pixels(R, good, "Pixels the two-material model explains   "
               "(best chi2/dof in the blob)", "tab:green",
               OUTPUT_DIR / "pixels_good.png")
    fig_pixels(R, bad, "Pixels it does not   (worst chi2/dof in the blob)",
               "tab:red", OUTPUT_DIR / "pixels_bad.png")

    yi, xi = R["data"]["yi"], R["data"]["xi"]
    rows = [dict(pixel=int(p), y=int(yi[p]), x=int(xi[p]),
                 x2=float(R["x2"][p]), chi2_dof=float(R["chi2"][p]),
                 sum_abs_resid=float(R["abs_err"][p]),
                 verdict="works" if p in set(good) else "fails")
            for p in list(good) + list(bad)]
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "showcase_pixels.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
