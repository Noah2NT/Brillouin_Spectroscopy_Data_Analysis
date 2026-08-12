"""
Core vs periphery Brillouin shift per dose - the Figure 3B/3D analogue.

Reproduces the region split used in Hilai et al., Small 2025, "Mechanical
Evolution of Metastatic Cancer Cells in 3D Microenvironment":

    "we segmented the spheroid into two zones: the core (region within 40% of
     the spheroid's radius from the center) and the periphery (region outside
     of the core)"

On a line scan through a spheroid that is the central 40% of the measured
width as core, and the outer 60% - 30% either side - as periphery. The
surrounding medium is excluded from both.

    |<---------------- spheroid width W ---------------->|
    |   periphery   |        core 0.4W      |  periphery |
                    ^         centre        ^
                 -0.4R                    +0.4R

CAREFUL - two different meanings of "periphery" are now in play:

  * HERE: the outer 60% of the spheroid body, a fixed geometric split.
  * ELSEWHERE (line_scan_feature_analysis, the tabletop panels): the fitted
    EDGE TRANSITION zone, where the shift is on its way from spheroid level
    down to medium level. That one measures how sharp the boundary is.

They are different regions and will not agree; do not read one as a check on
the other.

The paper found the core stiffer than the periphery (Figure 3B) and the
periphery more heterogeneous - higher SD - than the core (Figure 3D). Both
panels below are the dose-series equivalents, with dose on the x axis where
the paper had culture day.

Run from the repo root (the data paths are relative).
"""

import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import carboplatin_line_scans as C
import line_scan_feature_analysis as A


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

# Where the figures are written. None = show only.
FIG_PATH = None

# Fraction of the spheroid RADIUS counted as core, from the paper.
CORE_FRAC = 0.40

# Quantity to split. 'bfs' is what Figure 3B plots.
SIGNAL = "bfs"

CORE_STYLE = dict(color="black", marker="o", linestyle="-", linewidth=2.2,
                  markersize=8, capsize=5, label="Core")
PERI_STYLE = dict(color="0.45", marker="s", linestyle="--", linewidth=2.2,
                  markersize=8, capsize=5, label="Periphery")


# ─────────────────────────────────────────────
# Region split
# ─────────────────────────────────────────────

def core_periphery_masks(r, core_frac=CORE_FRAC):
    """(core, periphery) boolean masks over one scan's positions.

    Both are measured from the scan's own fitted tabletop: the centre is the
    midpoint of the two fitted edges and the radius is half the fitted width.
    Returns (None, None) when the scan has no usable fit.
    """
    tp = r["tab_params"]
    if tp is None:
        return None, None
    pos = np.asarray(r["scan"]["positions"], dtype=float)
    centre = 0.5 * (tp["x_left"] + tp["x_right"])
    radius = 0.5 * (tp["x_right"] - tp["x_left"])
    if not np.isfinite(radius) or radius <= 0:
        return None, None
    d = np.abs(pos - centre)
    core = d <= core_frac * radius
    periphery = (d > core_frac * radius) & (d <= radius)
    return core, periphery


def per_spheroid_stats(results, signal=SIGNAL):
    """One row per FILE: (core mean, periphery mean, core SD, periphery SD).

    A file is one spheroid; the two passes of a two-pass scan are the same
    spheroid measured twice, so they are averaged together before it counts
    as a sample - matching the paper, where n is a count of spheroids.
    """
    by_file = {}
    for r in results:
        core, peri = core_periphery_masks(r)
        if core is None:
            continue
        sig = np.asarray(r.get(signal), dtype=float)
        c, p = sig[core], sig[peri]
        c, p = c[np.isfinite(c)], p[np.isfinite(p)]
        if c.size < 3 or p.size < 3:
            continue
        by_file.setdefault(r["scan"]["filename"], []).append(
            (c.mean(), p.mean(), c.std(ddof=1), p.std(ddof=1)))
    return np.array([np.mean(v, axis=0) for v in by_file.values()])


# ─────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────

def _series(ax, x, vals, style):
    """Mean +/- SD across spheroids, as one line."""
    m = [v.mean() if v.size else np.nan for v in vals]
    sd = [v.std(ddof=1) if v.size > 1 else 0.0 for v in vals]
    ax.errorbar(x, m, yerr=sd, **style)
    return np.array(m), np.array(sd)


def plot_core_periphery(cfg, by_dose):
    """Two panels: mean shift per region, and within-spheroid SD per region."""
    x = np.arange(len(cfg["folders"]))
    labels = [cfg["labels"].get(f, f) for f in cfg["folders"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    _series(axes[0], x, [d[:, 0] for d in by_dose], CORE_STYLE)
    _series(axes[0], x, [d[:, 1] for d in by_dose], PERI_STYLE)
    axes[0].set_ylabel("Brillouin shift (GHz)")
    axes[0].set_title("Mean shift by region", fontsize=13)

    _series(axes[1], x, [d[:, 2] for d in by_dose], CORE_STYLE)
    _series(axes[1], x, [d[:, 3] for d in by_dose], PERI_STYLE)
    axes[1].set_ylabel("SD of Brillouin shift within spheroid (GHz)")
    axes[1].set_title("Mechanical heterogeneity by region", fontsize=13)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("dose")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=11)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

    n = [len(d) for d in by_dose]
    fig.suptitle(f"{cfg['drug']} - core vs periphery "
                 f"(core = inner {CORE_FRAC:.0%} of radius; "
                 f"error bars = SD across spheroids; n = {n})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if FIG_PATH:
        os.makedirs(FIG_PATH, exist_ok=True)
        fig.savefig(os.path.join(FIG_PATH,
                                 f"core_periphery_{cfg['drug']}.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()


def report(cfg, by_dose):
    print(f"\n{'=' * 88}")
    print(f"{cfg['drug']} - core vs periphery, core = inner "
          f"{CORE_FRAC:.0%} of the spheroid radius")
    print("=" * 88)
    print(f"  {'dose':<12}{'n':>3}{'core (GHz)':>16}{'periph (GHz)':>17}"
          f"{'core-periph':>13}{'p (paired)':>12}")
    all_c, all_p = [], []
    for folder, d in zip(cfg["folders"], by_dose):
        lbl = cfg["labels"].get(folder, folder)
        if not len(d):
            print(f"  {lbl:<12}{0:>3}   no usable spheroids")
            continue
        c, p = d[:, 0], d[:, 1]
        all_c.append(c); all_p.append(p)
        diff = c - p
        if len(c) > 1:
            pval = stats.ttest_rel(c, p).pvalue
            ptxt = f"{pval:>12.4f}"
        else:
            ptxt = f"{'n/a':>12}"
        print(f"  {lbl:<12}{len(c):>3}"
              f"{c.mean():>10.4f} +/-{c.std(ddof=1) if len(c) > 1 else 0:>5.4f}"
              f"{p.mean():>11.4f} +/-{p.std(ddof=1) if len(p) > 1 else 0:>5.4f}"
              f"{diff.mean() * 1000:>10.1f} MHz{ptxt}")
    if all_c:
        c, p = np.concatenate(all_c), np.concatenate(all_p)
        t = stats.ttest_rel(c, p)
        w = stats.wilcoxon(c, p)
        print("-" * 88)
        print(f"  Pooled over all doses (n = {len(c)} spheroids): "
              f"core - periphery = {(c - p).mean() * 1000:+.1f} MHz")
        print(f"    paired t-test p = {t.pvalue:.2e},  "
              f"Wilcoxon p = {w.pvalue:.2e},  "
              f"core higher in {int((c > p).sum())}/{len(c)} spheroids")
        sd_c = np.concatenate([d[:, 2] for d in by_dose])
        sd_p = np.concatenate([d[:, 3] for d in by_dose])
        wsd = stats.wilcoxon(sd_c, sd_p)
        print(f"  Within-spheroid SD: core {sd_c.mean() * 1000:.1f} MHz vs "
              f"periphery {sd_p.mean() * 1000:.1f} MHz, "
              f"Wilcoxon p = {wsd.pvalue:.2e}")


if __name__ == "__main__":
    for cfg in A.DRUGS:
        C.DRUG, C.BASE = cfg["drug"], cfg["base"]
        C.FOLDERS, C.FOLDER_COLORS = cfg["folders"], cfg["colors"]
        C.FOLDER_LABELS = cfg["labels"]
        C.FOLDER_PATHS = cfg.get("paths", {})
        by_dose = [per_spheroid_stats(C.load_and_analyse(C.folder_path(f)))
                   for f in cfg["folders"]]
        report(cfg, by_dose)
        plot_core_periphery(cfg, by_dose)
