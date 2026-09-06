"""BFS histograms annotated with every cutoff the background rule applies.

For each scan in data/csv, plot the distribution of Brillouin shift over the
valid pixels twice — once for the raw per-pixel fit, once for the spatially
smoothed fit — with a vertical line for each threshold that
`cell_mask_pipeline.confident_background_mask` uses:

  Otsu          : splits cell (high shift) from background (low shift); the
                  cell mask is built from the smoothed map's Otsu.
  median        : robust centre of the sub-Otsu (background) side.
  cut = med + k*MAD : the background acceptance cutoff. A pixel must fall
                  below this in BOTH maps to be eligible.

Bars are stacked by the class each pixel ended up in (background / uncertain /
cell). The classes deliberately do NOT separate cleanly at the cutoff line:
the value test is only step one, and the margin dilation plus the
minimum-component-size filter demote many below-cutoff pixels to 'uncertain'.
The gap between the cutoff line and where blue actually ends is exactly how
much work the spatial part of the rule is doing.
"""

from pathlib import Path
import importlib.util
import sys, os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
cmp_mod = _load("cell_mask_pipeline")

DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "background_endmember" / "bfs_histograms"

C_BG = "#2c7fb8"
C_UNC = "#d9d9d9"
C_CELL = "#c0392b"
N_BINS = 70


def cutoffs(bfs, valid, k_mad):
    """Otsu, robust background median/MAD and the acceptance cutoff."""
    thr = cmp_mod.otsu_threshold(bfs[valid])
    med, mad = cmp_mod._robust_background_level(bfs[valid], thr)
    return dict(otsu=float(thr), median=float(med), mad=float(mad),
                cut=float(med + k_mad * mad))


def draw(ax, bfs, valid, bg, cell, unc, cuts, k_mad, title):
    vals = bfs[valid]
    lo, hi = np.nanpercentile(vals, [0.2, 99.8])
    bins = np.linspace(lo, hi, N_BINS)

    ax.hist([bfs[bg], bfs[unc], bfs[cell]], bins=bins, stacked=True,
            color=[C_BG, C_UNC, C_CELL], edgecolor="none",
            label=[f"background ({int(bg.sum())})",
                   f"uncertain ({int(unc.sum())})",
                   f"cell ({int(cell.sum())})"])

    ax.axvline(cuts["median"], color="navy", ls=":", lw=1.6,
               label=f"bg median  {cuts['median']:.4f}")
    ax.axvline(cuts["cut"], color="navy", ls="-", lw=2.0,
               label=f"cut = med + {k_mad:g}·MAD  {cuts['cut']:.4f}")
    ax.axvline(cuts["otsu"], color="k", ls="--", lw=1.6,
               label=f"Otsu  {cuts['otsu']:.4f}")

    ax.set_xlabel("Brillouin shift (GHz)")
    ax.set_ylabel("pixels")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)


def process(path):
    stem = path.stem
    data = bge.load_cube(path)
    valid = data["valid"]

    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])

    bg, cell, unc, info = bge.find_background(data, bfs_raw, bfs_sm)
    k = bge.BG_K_MAD
    c_raw = cutoffs(bfs_raw, valid, k)
    c_sm = cutoffs(bfs_sm, valid, k)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    draw(axes[0], bfs_raw, valid, bg, cell, unc, c_raw, k,
         "raw per-pixel BFS")
    draw(axes[1], bfs_sm, valid, bg, cell, unc, c_sm, k,
         f"smoothed BFS (sigma={bge.MASK_SIGMA}) — cell mask comes from this Otsu")
    fig.suptitle(f"{stem}   |   k={k:g}, margin={info['margin']}, "
                 f"min_size={info['min_size']}   |   "
                 f"bg {int(bg.sum())} / unc {info['n_uncertain']} / "
                 f"cell {info['n_cell']}", fontsize=10)
    fig.savefig(OUTPUT_DIR / f"{stem}_bfs_cutoffs.png", dpi=140)
    plt.close(fig)

    # How many pixels pass the value test but are still rejected spatially?
    passes_value = valid & (bfs_raw <= c_raw["cut"]) & (bfs_sm <= c_sm["cut"])
    demoted = int((passes_value & ~bg).sum())
    return dict(sample=stem, k_mad=k,
                otsu_raw=c_raw["otsu"], median_raw=c_raw["median"],
                mad_raw=c_raw["mad"], cut_raw=c_raw["cut"],
                otsu_smoothed=c_sm["otsu"], median_smoothed=c_sm["median"],
                mad_smoothed=c_sm["mad"], cut_smoothed=c_sm["cut"],
                n_pass_value_test=int(passes_value.sum()),
                n_background=int(bg.sum()),
                n_demoted_by_spatial_rules=demoted,
                n_uncertain=info["n_uncertain"], n_cell=info["n_cell"]), \
           (stem, bfs_raw, valid, bg, cell, unc, c_raw, k)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []
    for path in sorted(DATA_DIR.glob("*.csv")):
        print(f"[{path.name}]")
        try:
            row, panel = process(path)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        panels.append(panel)
        print(f"  cut_raw {row['cut_raw']:.4f}  cut_sm {row['cut_smoothed']:.4f}"
              f"  otsu_sm {row['otsu_smoothed']:.4f} | "
              f"{row['n_pass_value_test']} pass value test -> "
              f"{row['n_background']} kept "
              f"({row['n_demoted_by_spatial_rules']} demoted spatially)")

    if not rows:
        return
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "bfs_cutoffs_summary.csv", index=False)

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.5 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax, (stem, bfs, valid, bg, cell, unc, cuts, k) in zip(
            axes.ravel(), panels):
        draw(ax, bfs, valid, bg, cell, unc, cuts, k, stem[:34])
        ax.legend(fontsize=5.5)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Raw BFS distribution with background cutoffs — all samples",
                 fontsize=13)
    fig.savefig(OUTPUT_DIR / "bfs_cutoffs_all_samples.png", dpi=130)
    plt.close(fig)

    print("\n" + pd.DataFrame(rows)[
        ["sample", "otsu_smoothed", "median_raw", "mad_raw", "cut_raw",
         "cut_smoothed", "n_pass_value_test", "n_background",
         "n_demoted_by_spatial_rules"]].to_string(index=False))


if __name__ == "__main__":
    main()
