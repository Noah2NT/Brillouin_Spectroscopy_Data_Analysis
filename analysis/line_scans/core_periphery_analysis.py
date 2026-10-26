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

# Where the figures are written ({drug} is filled in). None = show only.
FIG_PATH = "outputs/Ashutosh research/tfp_calibrated/{drug}"

# Number of bins in the pooled first-derivative profile.
DERIV_BINS = 60

# Quantity to split. 'bfs' is what Figure 3B plots.
SIGNAL = "bfs"

CORE_STYLE = dict(color="black", marker="o", linestyle="-", linewidth=2.2,
                  markersize=8, capsize=5, label="Core")
PERI_STYLE = dict(color="0.45", marker="s", linestyle="--", linewidth=2.2,
                  markersize=8, capsize=5, label="Periphery")


# ─────────────────────────────────────────────
# Region split
# ─────────────────────────────────────────────

def core_periphery_masks(r):
    """(core, periphery) boolean masks over one scan's positions.

    Both come from the scan's own fitted TABLETOP rather than from a fixed
    fraction of the radius: the plateau the fit calls 'center' is the core,
    and the boundary transition it calls 'edge' is the periphery. The medium
    outside the spheroid belongs to neither.

    This replaces the earlier inner-40%-of-radius rule, which imposed the same
    geometric split on every spheroid regardless of where its own boundary
    actually fell. Returns (None, None) when the scan has no usable fit.
    """
    region = r["region"]
    if region is None:
        return None, None
    core = region == "center"
    periphery = region == "edge"
    if not core.any() or not periphery.any():
        return None, None
    return core, periphery


def first_derivative(r):
    """(aligned positions, d(BFS)/dx in MHz/um) for one scan.

    The derivative is what locates the boundary: it is flat near zero across
    the medium and across the plateau, and peaks where the spheroid edge
    rises. Returns (None, None) when the scan has no usable fit.
    """
    tp = r["tab_params"]
    sig = r.get("bfs")
    if tp is None or sig is None:
        return None, None
    pos = np.asarray(r["scan"]["positions"], dtype=float)
    y = np.asarray(sig, dtype=float)
    ok = np.isfinite(pos) & np.isfinite(y)
    if ok.sum() < 5:
        return None, None
    pos, y = pos[ok], y[ok]
    centre = 0.5 * (tp["x_left"] + tp["x_right"])
    return pos - centre, np.gradient(y * 1e3, pos)      # GHz -> MHz


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


def _box_panel(ax, by_dose, c_col, p_col, labels):
    """Core/periphery box pairs per dose, paired t-test p written above."""
    w = 0.2
    for i, d in enumerate(by_dose):
        if not len(d):
            continue
        c, p = d[:, c_col], d[:, p_col]
        for vals, off, col in ((c, -w, "black"), (p, w, "0.45")):
            ax.boxplot(vals, positions=[i + off], widths=0.3, showfliers=False,
                       boxprops=dict(color=col), whiskerprops=dict(color=col),
                       capprops=dict(color=col), medianprops=dict(color=col))
            ax.plot(np.full(len(vals), i + off), vals, "o", markersize=5,
                    markerfacecolor="none", color=col, alpha=0.8)
        if len(c) > 1:
            pv = stats.ttest_rel(c, p).pvalue
            top = max(c.max(), p.max())
            pad = 0.04 * np.ptp(np.concatenate([d[:, [c_col, p_col]].ravel()
                                                for d in by_dose if len(d)]))
            ax.plot([i - w, i - w, i + w, i + w],
                    [top + pad, top + 1.5 * pad, top + 1.5 * pad, top + pad],
                    color="black", linewidth=1)
            ax.text(i, top + 1.7 * pad,
                    f"p={pv:.4f}" if pv >= 5e-5 else "p<0.0001", ha="center",
                    va="bottom", fontsize=10)
        if i:
            ax.axvline(i - 0.5, color="black", linewidth=0.8)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("dose")
    ax.grid(alpha=0.3)
    ax.plot([], [], color="black", label="Core")
    ax.plot([], [], color="0.45", label="Periphery")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))   # room for the p labels
    ax.legend(loc="best", fontsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _line_panel(ax, by_dose, c_col, p_col, labels):
    """Group-mean line per region over the faint per-spheroid markers."""
    x = np.arange(len(labels))
    line_core = {k: v for k, v in CORE_STYLE.items() if k != "capsize"}
    line_peri = {k: v for k, v in PERI_STYLE.items() if k != "capsize"}
    for col, style in ((c_col, line_core), (p_col, line_peri)):
        for i, d in enumerate(by_dose):
            if len(d):
                ax.plot(np.full(len(d), i), d[:, col], style["marker"],
                        markersize=6, color=style["color"], alpha=0.30,
                        linestyle="none", zorder=1)
        ax.plot(x, [d[:, col].mean() if len(d) else np.nan
                    for d in by_dose], zorder=3, **style)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("dose")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


# The four panels of the 2x2 figure, in column order:
# (core column, periphery column, y label, title, drawer).
PANELS = {
    "mean_line": (0, 1, "Brillouin shift (GHz)", "Mean shift by region",
                  _line_panel),
    "sd_line":   (2, 3, "Standard deviation of shift (GHz)",
                  "Standard deviation of shift by region", _line_panel),
    "mean_box":  (0, 1, "Brillouin shift (GHz)",
                  "Shift, spread across spheroids", _box_panel),
    "sd_box":    (2, 3, "Standard deviation of shift (GHz)",
                  "Standard deviation of shift, spread across spheroids",
                  _box_panel),
}

# Every y axis is cut into this many equal tick intervals, so at the same
# panel height the gap between tick labels looks the same on every graph.
# The step in GHz differs per panel; it is the smallest round value
# (1, 1.5, 2, 2.5, 3, 4, 5 x 10^k) whose N intervals cover the data.
N_Y_INTERVALS = 7


# Room kept between the outermost marker and the frame, as a fraction of the
# axis height, so no marker is clipped by the top or bottom edge.
Y_MARGIN = 0.02


def _nice_axis(lo, tops, n=N_Y_INTERVALS, margin=Y_MARGIN):
    """(lo, hi, step): the tightest n equal round steps that keep everything
    in frame.

    lo   - lowest data value.
    tops - (y, frac) pairs that must fit under the top: a data value
           (frac 0), or a label whose bottom sits at y and which is frac of
           the axis height tall. Text height is fixed on screen, so its
           height in data units grows with the range being tried.
    """
    hi = max(y for y, _ in tops)
    k = np.floor(np.log10((hi - lo) / n))
    for e in (k - 1, k, k + 1):
        for m in (1, 1.5, 2, 2.5, 3, 4, 5):
            step = m * 10 ** e
            span = n * step
            # The axis may start on any multiple of the decade unit (0.01
            # for a 0.03 step), not only of the step - so 6.14 is allowed.
            unit = 10 ** e
            start = np.floor((lo - margin * span) / unit + 1e-9) * unit
            if lo >= 0:         # an SD axis must not dip below zero
                start = max(start, 0.0)
            need = max(y + (f + margin) * span for y, f in tops)
            if start + span < need - 1e-12:
                continue
            # Centre the spare room, in whole units.
            shift = int((start + span - need) // unit) // 2
            if lo >= 0:
                shift = min(shift, int(round(start / unit)))
            start -= shift * unit
            return start, start + span, step
    raise ValueError("no round step found")


def _draw_panel(ax, cfg, by_dose, key, axis=None):
    c_col, p_col, ylabel, title, draw = PANELS[key]
    labels = [cfg["labels"].get(f, f) for f in cfg["folders"]]
    draw(ax, by_dose, c_col, p_col, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    if axis is not None:
        _apply_axis(ax, axis)


def _apply_axis(ax, axis):
    lo, hi, step = axis
    ax.set_ylim(lo, hi)
    ax.set_yticks(lo + step * np.arange(N_Y_INTERVALS + 1))
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _y_extent(ax):
    """(lowest data value, [(y, frac), ...] that must fit under the top)."""
    renderer = ax.figure.canvas.get_renderer()
    ax_h = ax.get_window_extent(renderer).height
    lo, hi = ax.dataLim.y0, ax.dataLim.y1
    tops = [(hi, 0.0)]
    for t in ax.texts:          # the p-value labels above each box pair
        tops.append((t.get_position()[1],
                     t.get_window_extent(renderer).height / ax_h))
    return lo, tops


def _shared_axes(axes_by_key):
    """One (lo, hi, step) per quantity, from the union of its line and box
    panels, so the two panels of a row have identical axes."""
    ext = {}
    for key, ax in axes_by_key.items():
        c_col = PANELS[key][0]
        lo, tops = _y_extent(ax)
        if c_col in ext:
            lo, tops = min(lo, ext[c_col][0]), tops + ext[c_col][1]
        ext[c_col] = (lo, tops)
    nice = {c: _nice_axis(lo, tops) for c, (lo, tops) in ext.items()}
    return {key: nice[PANELS[key][0]] for key in axes_by_key}


def plot_core_periphery(cfg, by_dose):
    """Mean shift and within-spheroid SD by region, every spheroid overlaid.

    The bold line is the group mean; the faint markers behind it are the
    individual spheroids. With 2-5 spheroids per dose a box plot implies more
    distributional detail than the data support, so the raw points are shown
    directly and the reader can judge the spread themselves.

    Each panel is also saved on its own at the same size - the single graphs
    that go onto the slides.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.8))
    axes_by_key = dict(zip(PANELS, axes.T.ravel()))
    for key, ax in axes_by_key.items():
        _draw_panel(ax, cfg, by_dose, key)
    y_axes = _shared_axes(axes_by_key)
    for key, ax in axes_by_key.items():
        _apply_axis(ax, y_axes[key])

    if FIG_PATH:
        out = FIG_PATH.format(drug=cfg["drug"])
        os.makedirs(out, exist_ok=True)
        for key in PANELS:
            f1, ax1 = plt.subplots(figsize=(6.75, 5.4))
            _draw_panel(ax1, cfg, by_dose, key, axis=y_axes[key])
            f1.tight_layout()
            f1.savefig(os.path.join(out, f"{key}_{cfg['drug']}.png"),
                       dpi=300, bbox_inches="tight")
            plt.close(f1)

    n = [len(d) for d in by_dose]
    fig.suptitle(f"{cfg['drug']} - core vs periphery "
                 f"(regions from each spheroid's fitted tabletop; "
                 f"faint markers = individual spheroids; n = {n})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if FIG_PATH:
        out = FIG_PATH.format(drug=cfg["drug"])
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f"core_periphery_{cfg['drug']}.png"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_first_derivative(cfg, all_results):
    """Pooled d(BFS)/dx per dose - where the spheroid boundary actually is.

    The derivative is near zero across the medium and across the plateau and
    peaks at the boundary, so it shows directly which pixels the tabletop is
    calling 'edge' and whether that split is doing the right thing.
    """
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for folder in cfg["folders"]:
        xs, ys = [], []
        for r in all_results[folder]:
            x, d = first_derivative(r)
            if x is None:
                continue
            xs.append(x); ys.append(d)
        if not xs:
            continue
        x = np.concatenate(xs); y = np.concatenate(ys)
        edges = np.linspace(np.nanmin(x), np.nanmax(x), DERIV_BINS + 1)
        idx = np.clip(np.digitize(x, edges) - 1, 0, DERIV_BINS - 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        means = np.full(DERIV_BINS, np.nan)
        for b in range(DERIV_BINS):
            m = (idx == b) & np.isfinite(y)
            if m.any():
                means[b] = np.mean(y[m])
        ok = np.isfinite(means)
        ax.plot(centres[ok], means[ok], '-o', markersize=3.5, linewidth=1.8,
                color=cfg["colors"][folder],
                label=f"{cfg['labels'].get(folder, folder)} (n={len(xs)})")

    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle=":")
    ax.axvline(0.0, color="0.4", linewidth=1.0, linestyle=":")
    ax.set_xlabel(C.POS_LABEL)
    ax.set_ylabel("d(BFS)/dx  (MHz/µm)")
    ax.set_title(f"{cfg['drug']} - first derivative of the shift profile",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout()
    if FIG_PATH:
        out = FIG_PATH.format(drug=cfg["drug"])
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f"first_derivative_{cfg['drug']}.png"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def report(cfg, by_dose):
    print(f"\n{'=' * 88}")
    print(f"{cfg['drug']} - core vs periphery, regions taken from each "
          f"spheroid's fitted tabletop")
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
        all_results = {f: C.load_and_analyse(C.folder_path(f))
                       for f in cfg["folders"]}
        by_dose = [per_spheroid_stats(all_results[f]) for f in cfg["folders"]]
        report(cfg, by_dose)
        plot_core_periphery(cfg, by_dose)
        plot_first_derivative(cfg, all_results)
        # The rest of the set sent to Ashutosh, into the same folder.
        C.SAVE_PATH = FIG_PATH.format(drug=cfg["drug"])
        for key in ("bfs", "lw"):
            C.plot_all_stacked(all_results, key)
        for key in ("bfs_0", "lw_0"):
            C.plot_folder_means(all_results, key)
        C.plot_tabletop_dose_lines(all_results, "bfs")
        C.plot_summary(all_results)
        plt.close("all")
