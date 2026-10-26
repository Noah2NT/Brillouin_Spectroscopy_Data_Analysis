"""Per-pixel look at the background subtraction, across several x metrics.

residual_endmembers.py answers "what endmembers come out". This answers the
question underneath it: for one pixel, how much medium is actually being
removed, and what does the leftover spectrum look like?

For ten pixels spanning cell / uncertain / background it draws, per metric:
    left   the raw spectrum s and the scaled pure background x*m on one axis
    right  the leftover s - x*m, with any negative region shaded

Negativity is the thing to look for. s - x*m < 0 is physically impossible for
a real mixture, so wherever the leftover dips below zero that metric has
removed more medium than the pixel contained, and the "endmember" VCA later
extracts from it is partly subtraction error.

Five metrics, ordered roughly from gentle to aggressive:

  vca3        medium abundance from anchored VCA(3). Model-based: it knows
              about the other two endmembers, so it is the only one that can
              tell "this pixel is dim" from "this pixel is medium".
  lstsq       (s.m)/(m.m), the plain projection of s onto m. Minimises the
              squared residual with no non-negativity constraint at all.
  peak_ratio  s / m at the single bin where m peaks. Intuitive and completely
              local -- and therefore hostage to the noise in that one bin.
  nonneg_p2   2nd percentile of s/m across bins: the largest removal that
              keeps the leftover non-negative, softened so one low bin cannot
              set the answer.
  nonneg_min  strict min of s/m -- the exact non-negativity bound. Guaranteed
              never to over-subtract, and guaranteed to be dragged down by
              whichever bin noise pushed lowest.

k = 1 throughout: each metric removes the full amount it estimates, which is
the only way to compare the estimates themselves rather than a shared scaling.
"""

from pathlib import Path
import importlib.util
import re, sys, os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_decomposition as bd


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bge = _load("background_endmember")
res = _load("residual_endmembers")


TARGET = "Hydrogl1"
DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "subtraction_metrics"

N_PER_CLASS = {"cell": 4, "uncertain": 3, "background": 3}   # 10 pixels
NEG_TOL = 0.02          # a leftover dipping this far below 0 counts as broken
RENORM_SHOW_MAX = 10.0  # beyond this amplification the renorm trace is noise
X_STEPS = np.round(np.arange(0.0, 1.001, 0.1), 2)   # the by-eye sweep
SEED = 42
CLASS_COLOUR = {"cell": "tab:red", "uncertain": "0.45",
                "background": "tab:blue"}


# -------------------------------------------------
# The metrics
# -------------------------------------------------

def x_lstsq(spectra, medium):
    """Unconstrained projection of each spectrum onto the medium."""
    return np.clip(spectra @ medium / (medium @ medium), 0.0, 1.0)


def x_peak_ratio(spectra, medium):
    """Ratio at the single bin where the medium peaks."""
    b = int(np.argmax(medium))
    return np.clip(spectra[:, b] / medium[b], 0.0, 1.0)


def build_metrics(spectra, medium, x_vca):
    """All five estimates of how much medium each pixel holds."""
    return {
        "vca3": x_vca,
        "lstsq": x_lstsq(spectra, medium),
        "peak_ratio": x_peak_ratio(spectra, medium),
        "nonneg_p2": res.nonneg_limit(spectra, medium, pct=2.0),
        "nonneg_min": res.nonneg_limit(spectra, medium, pct=0.0),
    }


def negativity(leftover, reference):
    """Deepest negative dip, as a fraction of the ORIGINAL spectrum's peak.

    0 means the subtraction stayed physical; 0.1 means the dip is a tenth of
    the height the pixel started at, i.e. badly over-subtracted.

    Scaling by the original rather than by the leftover's own peak matters:
    a pixel that subtracts to exactly zero (the hard-set background at x=1)
    has no leftover peak to divide by, and scoring it as infinitely negative
    would flag every background pixel as broken when in fact it cancelled
    perfectly. Input spectra are sum-normalised, so reference.max() > 0.
    """
    return float(max(-leftover.min(), 0.0) / reference.max())


# -------------------------------------------------
# Pixel choice
# -------------------------------------------------

def pick_pixels(x_vca, cell_flat, bg_flat):
    """Ten pixels spanning the three classes and, within each, the x range.

    Picking by evenly spaced quantiles of x_vca rather than at random means
    the ten always cover dim-to-medium-rich instead of clustering wherever
    that class happens to be dense.
    """
    other = ~(cell_flat | bg_flat)
    chosen = []
    for name, mask in (("cell", cell_flat), ("uncertain", other),
                       ("background", bg_flat)):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        order = idx[np.argsort(x_vca[idx])]
        n = N_PER_CLASS[name]
        picks = np.unique(np.linspace(0, len(order) - 1, n).round().astype(int))
        chosen += [(name, int(order[p])) for p in picks]
    return chosen


# -------------------------------------------------
# Plotting
# -------------------------------------------------

def render_metric(name, x, pixels, spectra, medium, shift, bfs_raw, yi, xi,
                  stem, out_path):
    """One figure per metric: ten pixels, three panels each.

    Panels are [where the pixel is | raw + what is subtracted | leftover].
    All ten BFS maps share one colour scale so the marker positions can be
    compared against each other, not just against their own map.
    """
    fig = plt.figure(figsize=(23, 17), constrained_layout=True)
    gs = GridSpec(5, 6, figure=fig)
    lo, hi = np.nanpercentile(bfs_raw, [2, 98])

    for p, (cls, i) in enumerate(pixels):
        r, c = p // 2, 3 * (p % 2)
        s = spectra[i]
        sub = x[i] * medium
        left = s - sub

        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(bfs_raw, origin="lower", cmap="magma", vmin=lo, vmax=hi)
        ax.axhline(yi[i], color="red", lw=0.6, alpha=0.55)
        ax.axvline(xi[i], color="red", lw=0.6, alpha=0.55)
        ax.plot(xi[i], yi[i], marker="s", ms=11, mfc="none", mec="red", mew=2.0)
        fig.colorbar(im, ax=ax, shrink=0.7, label="GHz")
        ax.set_title(f"{cls}  px {i}  (y={yi[i]}, x={xi[i]})", fontsize=9,
                     color=CLASS_COLOUR[cls])
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[r, c + 1])
        ax.plot(shift, s, lw=1.5, color=CLASS_COLOUR[cls], label="raw spectrum")
        ax.plot(shift, sub, lw=1.5, color="tab:blue", ls="--",
                label=f"$x\\,m$   x = {x[i]:.3f}")
        ax.set_title("raw + what is subtracted", fontsize=9)
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
        ax.set_ylabel("norm. intensity", fontsize=7)
        if r == 4:
            ax.set_xlabel("Shift (GHz)", fontsize=8)

        ax = fig.add_subplot(gs[r, c + 2])
        ax.plot(shift, left, lw=1.5, color="k", label="$s - x\\,m$")
        ax.fill_between(shift, left, 0, where=left < 0, color="tab:red",
                        alpha=0.45, interpolate=True, label="over-subtracted")
        # An almost-pure-medium pixel has almost nothing left, so its
        # renormalised trace is the leftover blown up by 1/(1-x). Past ~10x
        # that is amplified noise which flattens everything else in the panel,
        # so the factor goes in the title instead of on the axes.
        surviving = float(np.abs(left).sum())   # negatives count as area
        amp = 1.0 / surviving if surviving > 1e-6 else np.inf
        if amp <= RENORM_SHOW_MAX:
            ax.plot(shift, left / surviving, lw=0.9, color="tab:green",
                    ls=":", label=f"renorm. ($\\times${amp:.1f})")
        ax.axhline(0, color="0.5", lw=0.8)
        neg = negativity(left, s)
        hidden = "" if amp <= RENORM_SHOW_MAX else f"   renorm. x{amp:.0f} off-scale"
        ax.set_title(f"leftover   negativity {neg:.1%}{hidden}", fontsize=9,
                     color=("tab:red" if neg > NEG_TOL else "0.2"))
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
        if r == 4:
            ax.set_xlabel("Shift (GHz)", fontsize=8)

    fig.suptitle(f"{stem}   metric = {name}   (k = 1, full estimated removal)\n"
                 "red shading = leftover went negative, i.e. more medium was "
                 "removed than the pixel held", fontsize=12)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_overview(metrics, pixels, spectra, medium, data, cell_flat,
                    bg_flat, stem, out_path):
    """Cross-metric view: the maps, the ten pixels, and who over-subtracts."""
    shape, yi, xi = data["shape"], data["yi"], data["xi"]
    names = list(metrics)
    fig = plt.figure(figsize=(19, 11), constrained_layout=True)
    gs = GridSpec(2, 5, figure=fig)

    for j, name in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        m = np.full(shape, np.nan)
        m[yi, xi] = metrics[name]
        im = ax.imshow(m, origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"x  |  {name}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax.set_xticks([]); ax.set_yticks([])

    # x per chosen pixel, every metric on one axis
    ax = fig.add_subplot(gs[1, :2])
    labels = [f"{cls[:4]}\n{i}" for cls, i in pixels]
    for name in names:
        ax.plot(range(len(pixels)), [metrics[name][i] for _, i in pixels],
                "o-", lw=1.4, ms=5, label=name)
    ax.set_xticks(range(len(pixels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("x  (fraction of medium removed)")
    ax.set_title("the ten pixels: how much each metric removes", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # over-subtraction rate per metric, per class
    ax = fig.add_subplot(gs[1, 2:4])
    other = ~(cell_flat | bg_flat)
    width = 0.26
    for o, (cls, mask) in enumerate((("cell", cell_flat),
                                     ("uncertain", other),
                                     ("background", bg_flat))):
        vals = []
        for name in names:
            left = spectra[mask] - metrics[name][mask][:, None] * medium[None, :]
            ref = spectra[mask]
            vals.append(100 * np.mean(
                [negativity(r, q) > NEG_TOL for r, q in zip(left, ref)]))
        ax.bar(np.arange(len(names)) + (o - 1) * width, vals, width,
               label=cls, color=CLASS_COLOUR[cls])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_ylabel("% of pixels over-subtracted\n"
                  f"(negativity > {NEG_TOL:.0%} of peak)")
    ax.set_title("which metrics remove more than is there?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    # mean leftover per metric, cell pixels only
    ax = fig.add_subplot(gs[1, 4])
    shift = data["shift_freq"]
    for name in names:
        left = spectra[cell_flat] - metrics[name][cell_flat][:, None] * medium[None, :]
        ax.plot(shift, left.mean(axis=0), lw=1.4, label=name)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("norm. intensity")
    ax.set_title("mean cell leftover", fontsize=10)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

    fig.suptitle(f"{stem}   comparing five ways to decide how much background "
                 "to subtract", fontsize=12)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)


def render_pixel_sweep(cls, i, spectra, medium, shift, bfs_raw, yi, xi,
                      metrics, stem, out_path):
    """One pixel, subtracting progressively more background: x = 0.0 .. 1.0.

    No metric decides anything here -- the point is to show every amount and
    let the eye pick. Each panel holds the same y-limits so the leftover
    visibly shrinks as x grows, while the dotted green trace is that leftover
    renormalised to unit area, which keeps its SHAPE readable even once the
    raw curve has collapsed to a sliver. The five metric estimates are printed
    in the header so you can see where each of them would have stopped.
    """
    s = spectra[i]
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)
    ymax, ymin = s.max() * 1.15, -0.3 * s.max()

    ax = fig.add_subplot(gs[0, 0])
    lo, hi = np.nanpercentile(bfs_raw, [2, 98])
    im = ax.imshow(bfs_raw, origin="lower", cmap="magma", vmin=lo, vmax=hi)
    ax.axhline(yi[i], color="red", lw=0.6, alpha=0.55)
    ax.axvline(xi[i], color="red", lw=0.6, alpha=0.55)
    ax.plot(xi[i], yi[i], marker="s", ms=13, mfc="none", mec="red", mew=2.2)
    fig.colorbar(im, ax=ax, shrink=0.75, label="GHz")
    ax.set_title(f"{cls}  px {i}  (y={yi[i]}, x={xi[i]})", fontsize=11,
                 color=CLASS_COLOUR[cls])
    ax.set_xticks([]); ax.set_yticks([])

    for j, xv in enumerate(X_STEPS):
        slot = j + 1
        ax = fig.add_subplot(gs[slot // 4, slot % 4])
        left = s - xv * medium
        ax.plot(shift, s, lw=0.8, color="0.8", label="raw $s$")
        ax.plot(shift, xv * medium, lw=0.9, ls="--", color="tab:blue",
                alpha=0.75, label="$x\,m$")
        ax.plot(shift, left, lw=1.9, color="k", label="$s-x\,m$")
        ax.fill_between(shift, left, 0, where=left < 0, color="tab:red",
                        alpha=0.45, interpolate=True)
        area = float(np.abs(left).sum())
        if area > 1e-6:
            ax.plot(shift, left / area, lw=1.0, ls=":", color="tab:green",
                    label="renorm.")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_ylim(ymin, ymax)
        neg = negativity(left, s)
        ax.set_title(f"x = {xv:.1f}     negativity {neg:.1%}", fontsize=12,
                     color=("tab:red" if neg > NEG_TOL else "0.15"))
        ax.grid(alpha=0.3)
        if slot == 1:
            ax.legend(fontsize=7)
        if slot >= 8:
            ax.set_xlabel("Shift (GHz)", fontsize=8)

    est = "   ".join(f"{k}={v[i]:.3f}" for k, v in metrics.items())
    fig.suptitle(f"{stem}   |   {cls} px {i}  -- how much background to remove?   |   "
                 f"metric estimates:  {est}", fontsize=13)
    fig.savefig(out_path, dpi=115)
    plt.close(fig)


def residual_floor(spectra, medium, bfs_raw, data, cell_flat, bg_flat,
                   stem, out_path):
    """Subtract exactly 1.0*m from every pixel and measure what is left.

    On a genuine background pixel that subtraction is, by assertion, exactly
    right, so the leftover is pure measurement noise. Its size is therefore a
    NOISE FLOOR: a leftover anywhere else in the scan only carries real signal
    once it clears this. Size is reported three ways because the word is
    ambiguous -- total absolute area sum|r| (directly readable as a fraction
    of the spectrum's own area, which is 1.0 by sum-normalisation), mean |r|
    per bin, and RMS per bin.

    One caveat on the floor: m is the co-add of these same background pixels,
    so each contributed roughly 1/n of it and the scatter measured here is
    very slightly optimistic. With n in the hundreds that bias is far under a
    percent, so it is noted rather than corrected.
    """
    yi, xi, shift = data["yi"], data["xi"], data["shift_freq"]
    resid = spectra - medium[None, :]
    area = np.abs(resid).sum(axis=1)
    rms = np.sqrt((resid ** 2).mean(axis=1))
    other = ~(cell_flat | bg_flat)

    bg = area[bg_flat]
    stats = dict(
        n_background=int(bg_flat.sum()),
        abs_area_mean=float(bg.mean()), abs_area_std=float(bg.std()),
        abs_area_median=float(np.median(bg)),
        abs_area_p5=float(np.percentile(bg, 5)),
        abs_area_p95=float(np.percentile(bg, 95)),
        abs_area_min=float(bg.min()), abs_area_max=float(bg.max()),
        mean_abs_per_bin=float(np.abs(resid[bg_flat]).mean()),
        rms_per_bin_mean=float(rms[bg_flat].mean()),
        rms_per_bin_std=float(rms[bg_flat].std()),
        n_bins=int(resid.shape[1]),
        # Total raw counts per background pixel. Spectra are sum-normalised
        # before any of this, so exposure has been divided out of `resid`;
        # this is the only place the actual photon budget survives, and it is
        # what shot noise should scale against.
        raw_counts_mean=float(data["folded_raw"][bg_flat].sum(axis=1).mean()),
        raw_counts_median=float(
            np.median(data["folded_raw"][bg_flat].sum(axis=1))),
        uncertain_abs_area_mean=(float(area[other].mean())
                                 if other.any() else float("nan")),
        cell_abs_area_mean=float(area[cell_flat].mean()),
        cell_over_floor=float(area[cell_flat].mean() / bg.mean()),
    )

    fig = plt.figure(figsize=(19, 11), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    for mask, colour, lab in ((bg_flat, "tab:blue", "background"),
                              (other, "0.45", "uncertain"),
                              (cell_flat, "tab:red", "cell")):
        if mask.any():
            ax.hist(area[mask], bins=40, color=colour, alpha=0.6, label=lab)
    ax.axvline(bg.mean(), color="k", lw=1.6,
               label="bg mean {:.4f}".format(bg.mean()))
    ax.axvline(bg.mean() + bg.std(), color="k", lw=0.9, ls="--",
               label="+/- 1 sd {:.4f}".format(bg.std()))
    ax.axvline(max(bg.mean() - bg.std(), 0.0), color="k", lw=0.9, ls="--")
    ax.set_xlabel("sum |s - m|   (spectrum area = 1.0)")
    ax.set_ylabel("pixels")
    ax.set_title("leftover size after removing exactly 1.0 m", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    grid = np.full(data["shape"], np.nan)
    grid[yi, xi] = area
    im = ax.imshow(grid, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8, label="sum |s - m|")
    ax.set_title("where the leftover is large", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    r_bg = resid[bg_flat]
    mu, sd = r_bg.mean(axis=0), r_bg.std(axis=0)
    ax.plot(shift, medium, lw=1.2, color="tab:blue", alpha=0.45,
            label="pure endmember m")
    ax.plot(shift, mu, lw=1.6, color="k", label="mean background leftover")
    ax.fill_between(shift, mu - sd, mu + sd, color="0.6", alpha=0.5,
                    label="+/- 1 sd across pixels")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("norm. intensity")
    ax.set_title("the leftover is noise, centred on zero", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(shift, sd, lw=1.5, color="tab:purple", label="sd of leftover")
    ax.plot(shift, medium * sd.max() / medium.max(), lw=1.0, ls="--",
            color="tab:blue", alpha=0.7, label="m (rescaled)")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("sd across background px")
    ax.set_title("noise tracks the peak, as shot noise would", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    bfs_flat = bfs_raw[yi, xi]
    ax.scatter(bfs_flat[bg_flat], area[bg_flat], s=6, alpha=0.5,
               color="tab:blue", label="background")
    if other.any():
        ax.scatter(bfs_flat[other], area[other], s=6, alpha=0.5, color="0.45",
                   label="uncertain")
    ax.axhline(bg.mean(), color="k", lw=1.2, label="bg mean")
    ax.axhline(bg.mean() + 3 * bg.std(), color="tab:red", lw=1.0, ls="--",
               label="bg mean + 3 sd")
    ax.set_xlabel("pixel BFS (GHz)"); ax.set_ylabel("sum |s - m|")
    ax.set_title("does the leftover grow toward the bead?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    txt = [
        "BACKGROUND PIXELS (n = {}), x = 1.0 exactly".format(
            stats["n_background"]),
        "",
        "sum |r|   mean    {:.5f}".format(stats["abs_area_mean"]),
        "          sd      {:.5f}".format(stats["abs_area_std"]),
        "          median  {:.5f}".format(stats["abs_area_median"]),
        "          p5-p95  {:.5f} - {:.5f}".format(stats["abs_area_p5"],
                                                   stats["abs_area_p95"]),
        "          range   {:.5f} - {:.5f}".format(stats["abs_area_min"],
                                                   stats["abs_area_max"]),
        "",
        "mean |r| per bin  {:.3e}".format(stats["mean_abs_per_bin"]),
        "RMS per bin mean  {:.3e}".format(stats["rms_per_bin_mean"]),
        "            sd    {:.3e}".format(stats["rms_per_bin_std"]),
        "bins              {}".format(stats["n_bins"]),
        "",
        "SAME SUBTRACTION ELSEWHERE (context):",
        "  uncertain mean sum|r|   {:.5f}".format(
            stats["uncertain_abs_area_mean"]),
        "  cell      mean sum|r|   {:.5f}".format(
            stats["cell_abs_area_mean"]),
        "  cell / background floor {:.1f}x".format(stats["cell_over_floor"]),
        "",
        "The spectrum's own area is 1.0, so sum|r|",
        "reads directly as a fraction of the signal.",
    ]
    ax.text(0.0, 1.0, chr(10).join(txt), va="top", ha="left",
            family="monospace", fontsize=11, transform=ax.transAxes)

    fig.suptitle(stem + "   |   noise floor: remove exactly 1.0 x the pure "
                 "endmember from background pixels", fontsize=13)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    return stats


EXPOSURE_RE = re.compile(r"_(\d+(?:\.\d+)?)s(?:[_\s.]|$)")


def exposure_from_name(stem):
    """Seconds of integration, parsed from the filename, or None.

    Only some scans record it. Returning None rather than a guess keeps
    unlabelled scans out of the fit instead of silently inventing an x value.
    """
    m = EXPOSURE_RE.search(stem)
    return float(m.group(1)) if m else None


def prepare(path):
    """Load a scan, build the trimap, co-add the pure background endmember."""
    data = bge.load_cube(path)
    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])
    bg_mask, cell_mask, _, _ = bge.find_background(data, bfs_raw, bfs_sm)
    medium, _, _ = bge.build_endmember(data, bg_mask)
    yi, xi = data["yi"], data["xi"]
    return dict(data=data, bfs_raw=bfs_raw, medium=medium,
                bg_flat=bg_mask[yi, xi], cell_flat=cell_mask[yi, xi])


def render_floor_vs_exposure(df, out_path):
    """Is the noise floor set by how long each pixel was integrated?

    Two x-axes, because the filename is a poor record. Exposure appears in
    only some names, so that panel has a handful of points at very few
    distinct values. Total raw counts is available for every scan and is the
    variable shot noise actually depends on -- exposure only matters through
    it, and only if laser power and detector gain were held constant.

    Prediction: spectra are sum-normalised, so a per-bin count n with Poisson
    error sqrt(n) becomes a relative error sqrt(n)/N. Summed over bins that
    scales as 1/sqrt(N), giving slope -1/2 on log-log against counts and a
    straight line through the origin against 1/sqrt(N).
    """
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    d = df.sort_values("abs_area_mean")

    ax = fig.add_subplot(gs[0, 0])
    ax.barh(range(len(d)), d["abs_area_mean"], xerr=d["abs_area_std"],
            color="tab:blue", alpha=0.8, error_kw=dict(lw=1.0))
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([n[:38] for n in d["sample"]], fontsize=7)
    ax.set_xlabel("noise floor   mean sum|r|  (+/- 1 sd)")
    ax.set_title("floor per sample", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    ax = fig.add_subplot(gs[0, 1])
    lab = df.dropna(subset=["exposure_s"])
    if len(lab):
        ax.errorbar(lab["exposure_s"], lab["abs_area_mean"],
                    yerr=lab["abs_area_std"], fmt="o", ms=8, capsize=4,
                    color="tab:red")
        for _, r in lab.iterrows():
            ax.annotate(r["sample"][:20], (r["exposure_s"], r["abs_area_mean"]),
                        fontsize=6, xytext=(6, 3), textcoords="offset points")
        t = np.linspace(lab["exposure_s"].min() * 0.85,
                        lab["exposure_s"].max() * 1.15, 50)
        ref = lab.loc[lab["exposure_s"].idxmax()]
        ax.plot(t, ref["abs_area_mean"] * np.sqrt(ref["exposure_s"] / t),
                "--", color="0.4", lw=1.2,
                label="shot noise 1/sqrt(t), through longest exposure")
        ax.legend(fontsize=7)
    ax.set_xlabel("exposure from filename (s)")
    ax.set_ylabel("mean sum|r|")
    ax.set_title("floor vs stated exposure ({} of {} scans record it)".format(
        len(lab), len(df)), fontsize=10)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.errorbar(df["raw_counts_mean"], df["abs_area_mean"],
                yerr=df["abs_area_std"], fmt="o", ms=7, capsize=3,
                color="tab:blue")
    n = np.linspace(df["raw_counts_mean"].min() * 0.7,
                    df["raw_counts_mean"].max() * 1.4, 50)
    med = df["abs_area_mean"].median() * np.sqrt(df["raw_counts_mean"].median())
    ax.plot(n, med / np.sqrt(n), "--", color="0.4", lw=1.3,
            label="pure shot noise, slope -1/2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean raw counts per background pixel")
    ax.set_ylabel("mean sum|r|")
    ax.set_title("floor vs photon budget (all scans)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, 1])
    inv = 1.0 / np.sqrt(df["raw_counts_mean"].to_numpy())
    y = df["abs_area_mean"].to_numpy()
    ax.scatter(inv, y, s=55, color="tab:blue", zorder=3)
    for xv, yv, nm in zip(inv, y, df["sample"]):
        ax.annotate(nm[:18], (xv, yv), fontsize=6, xytext=(5, 3),
                    textcoords="offset points")
    if len(df) > 2:
        slope, icept = np.polyfit(inv, y, 1)
        xx = np.linspace(min(inv.min() * 0.9, 0.0), inv.max() * 1.05, 20)
        pred = slope * inv + icept
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.plot(xx, slope * xx + icept, "-", color="tab:red", lw=1.4,
                label="fit  R^2 = {:.3f}".format(r2))
        ax.axhline(icept, color="0.5", lw=0.9, ls=":",
                   label="intercept {:.4f} = floor NOT from shot noise".format(
                       icept))
        ax.legend(fontsize=7)
    ax.set_xlabel("1 / sqrt(mean raw counts)")
    ax.set_ylabel("mean sum|r|")
    ax.set_title("shot noise alone would pass through the origin", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Noise floor across scans: remove exactly 1.0 x the pure "
                 "endmember from every background pixel", fontsize=13)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)


def floor_all_samples():
    """Run only the noise-floor test, on every scan in data/csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        stem = path.stem
        print("[{}]".format(path.name))
        try:
            prep = prepare(path)
            if int(prep["bg_flat"].sum()) < bge.MIN_BG_PIXELS:
                print("  [skip] not enough confident background")
                continue
            st = residual_floor(
                prep["data"]["folded_norm"], prep["medium"], prep["bfs_raw"],
                prep["data"], prep["cell_flat"], prep["bg_flat"], stem,
                OUTPUT_DIR / (stem + "_residual_floor.png"))
        except Exception as exc:
            print("  [error] {}: {}".format(type(exc).__name__, exc))
            continue
        st["sample"] = stem
        st["exposure_s"] = exposure_from_name(stem)
        rows.append(st)
        print("  floor {:.5f} +/- {:.5f} | counts {:.3e} | exposure {}".format(
            st["abs_area_mean"], st["abs_area_std"], st["raw_counts_mean"],
            st["exposure_s"] if st["exposure_s"] else "not in filename"))

    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = ["sample", "exposure_s", "n_background", "raw_counts_mean",
            "abs_area_mean", "abs_area_std", "abs_area_median",
            "rms_per_bin_mean", "cell_abs_area_mean", "cell_over_floor"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(OUTPUT_DIR / "noise_floor_all_samples.csv", index=False)
    render_floor_vs_exposure(df, OUTPUT_DIR / "noise_floor_all_samples.png")
    print(chr(10) + df[cols].to_string(index=False))


# -------------------------------------------------
# Driver
# -------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = [p for p in sorted(DATA_DIR.glob("*.csv"))
               if TARGET.lower() in p.name.lower()]
    if not matches:
        print(f"no csv matching {TARGET!r} in {DATA_DIR}")
        return
    path = matches[0]
    stem = path.stem
    print(f"[{path.name}]")

    data = bge.load_cube(path)
    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])
    bg_mask, cell_mask, _, _ = bge.find_background(data, bfs_raw, bfs_sm)

    medium, _, _ = bge.build_endmember(data, bg_mask)
    yi, xi = data["yi"], data["xi"]
    bg_flat, cell_flat = bg_mask[yi, xi], cell_mask[yi, xi]

    # Unlike residual_endmembers, the background is NOT hard-set to m here.
    # Doing that forces every background leftover to an identically flat zero,
    # which proves nothing because it was true by construction. On the real
    # spectra "does x come out near 1 on background?" becomes an actual
    # measurement of each metric, and the leftover shows the noise that
    # genuinely survives.
    spectra = data["folded_norm"]
    em3, _ = bge.vca_anchored(spectra, 3, medium, seed=SEED,
                              candidates=np.flatnonzero(cell_flat))
    x_vca = bd.estimate_abundances(spectra, em3)[:, 0]

    metrics = build_metrics(spectra, medium, x_vca)
    pixels = pick_pixels(x_vca, cell_flat, bg_flat)
    print(f"  {int(cell_flat.sum())} cell / "
          f"{int((~(cell_flat | bg_flat)).sum())} uncertain / "
          f"{int(bg_flat.sum())} background px; showing {len(pixels)}")

    for name, x in metrics.items():
        render_metric(name, x, pixels, spectra, medium, data["shift_freq"],
                      bfs_raw, yi, xi, stem,
                      OUTPUT_DIR / f"{stem}_pixels_{name}.png")
        left = spectra - x[:, None] * medium[None, :]
        negs = np.array([negativity(r, q) for r, q in zip(left, spectra)])
        print(f"  {name:11s} x: bg {x[bg_flat].mean():.3f}  "
              f"cell {x[cell_flat].mean():.3f}  |  over-subtracted "
              f"{100 * (negs > NEG_TOL).mean():5.1f}% of px "
              f"(cell {100 * (negs[cell_flat] > NEG_TOL).mean():5.1f}%)")

    for cls, i in pixels:
        render_pixel_sweep(cls, i, spectra, medium, data["shift_freq"],
                           bfs_raw, yi, xi, metrics, stem,
                           OUTPUT_DIR / f"{stem}_sweep_{cls}_px{i}.png")
    print(f"  wrote {len(pixels)} per-pixel sweeps "
          f"(x = {X_STEPS[0]:.1f} .. {X_STEPS[-1]:.1f})")

    floor = residual_floor(spectra, medium, bfs_raw, data, cell_flat, bg_flat,
                           stem, OUTPUT_DIR / (stem + "_residual_floor.png"))
    pd.DataFrame([floor]).to_csv(
        OUTPUT_DIR / (stem + "_residual_floor.csv"), index=False)
    print("  noise floor  sum|r| = "
          "{:.5f} +/- {:.5f} (median {:.5f}, range {:.5f}-{:.5f}) | "
          "per-bin RMS {:.3e} | cell sits {:.1f}x above the floor".format(
              floor["abs_area_mean"], floor["abs_area_std"],
              floor["abs_area_median"], floor["abs_area_min"],
              floor["abs_area_max"], floor["rms_per_bin_mean"],
              floor["cell_over_floor"]))

    render_overview(metrics, pixels, spectra, medium, data, cell_flat,
                    bg_flat, stem, OUTPUT_DIR / f"{stem}_metric_overview.png")

    rows = []
    for cls, i in pixels:
        row = dict(pixel=i, klass=cls, y=int(yi[i]), x_coord=int(xi[i]))
        for name, x in metrics.items():
            row[f"x_{name}"] = float(x[i])
            row[f"neg_{name}"] = negativity(spectra[i] - x[i] * medium,
                                            spectra[i])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"{stem}_pixel_table.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    floor_all_samples() if "--all" in sys.argv else main()
