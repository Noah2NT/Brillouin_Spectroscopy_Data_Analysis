"""Step-by-step: what decides each pixel's label, one gate at a time.

Each histogram is coloured strictly by the decision made at THAT step, so the
colouring is always explainable by what has been applied so far.

  STEP 0   every valid pixel, uncoloured.
  STEP 1   GATE 1 VALUE. Colour = pass/fail of `raw <= cut AND smoothed <= cut`.
           This colouring IS a pure function of the x-axis: a vertical split.
  STEP 2   GATE 2a REGIONS. The gate-1 failures are opened (erode once, dilate
           once) to delete strays, then every component of at least
           `min_cell_size` px is kept as a cell. Multiple cells all survive.
  STEP 3   GATE 2b MARGIN, applied only to gate-1 survivors. Colour = kept vs
           removed. The two colours overlap across the WHOLE x range, because
           this gate does not look at BFS at all.
  STEP 4   GATE 3 SIZE, then the final trimap.
  STEP 5   the real decision space: BFS on x, distance-to-nearest-cell on y.
           The boundary is a horizontal line at y = margin. Projecting this
           onto x is what a plain histogram does, which is why classes appear
           to interleave there.

`dilate_mask` iterates scipy's default 4-connected structuring element, so
`margin` iterations means CITYBLOCK distance <= margin. The y-axis uses the
matching metric, which puts the boundary exactly on the line.
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import (distance_transform_cdt, label as cc_label,
                           binary_erosion, binary_dilation)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import feature_extraction as fe


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bge = _load("background_endmember")

OUTPUT_DIR = Path("outputs") / "background_endmember" / "step_by_step"

DEFAULT_SAMPLES = [
    "Sample11_JiPSC-1000_Plate2_A1_FinalRawData.csv",
    "Sample12_JiPSC-1066_Plate2_A2_data1_new.csv",
    "XY_scanning_40um_60um_step1um_10s_fixed.csv",
]

C_PASS = "#2c7fb8"     # survives this gate
C_FAIL = "#d9d9d9"     # never entered this gate
C_CUT = "#e08214"      # removed BY this gate
C_CELL = "#c0392b"
N_BINS = 70


def _bare(ax):
    ax.set_xticks([]); ax.set_yticks([])


def hist_split(ax, bfs, keep, drop, rest, bins, labels, note=None):
    data, colors, names = [], [], []
    for m, c, n in ((rest, C_FAIL, labels[2]), (drop, C_CUT, labels[1]),
                    (keep, C_PASS, labels[0])):
        if m is not None:
            data.append(bfs[m]); colors.append(c); names.append(n)
    ax.hist(data, bins=bins, stacked=True, color=colors, label=names,
            edgecolor="none")
    ax.set_xlabel("raw BFS (GHz)"); ax.set_ylabel("pixels")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, bbox=dict(fc="white", ec="0.7", alpha=0.9))


def render(stem, data, bfs_raw, st, out_path):
    shape = data["shape"]
    valid = data["valid"]
    bins = np.linspace(*np.nanpercentile(bfs_raw[valid], [0.2, 99.8]), N_BINS)

    bg_like = st["bg_like"]
    cell = st["cell"]
    after_margin = st["after_margin"]
    background = st["background"]
    uncertain = st["uncertain"]
    info = st["info"]
    margin = info["margin"]

    # Cityblock distance to the nearest CELL pixel — the metric the
    # 4-connected dilation of the cell regions actually implements.
    dist = distance_transform_cdt(~cell, metric="taxicab")

    fig = plt.figure(figsize=(15, 28), constrained_layout=True)
    gs = GridSpec(6, 2, figure=fig, width_ratios=[1.25, 1])

    # ── STEP 0 ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(bfs_raw[valid], bins=bins, color="#999999", edgecolor="none")
    ax.axvline(info["cut_raw"], color="navy", lw=2.0,
               label=f"cut = med+{info['k_mad']:g}·MAD = {info['cut_raw']:.4f}")
    ax.axvline(info["otsu_thr"], color="k", ls="--", lw=1.6,
               label=f"Otsu = {info['otsu_thr']:.4f}")
    ax.set_xlabel("raw BFS (GHz)"); ax.set_ylabel("pixels")
    ax.set_title(f"STEP 0 · all {int(valid.sum())} valid pixels",
                 fontsize=11, loc="left")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(bfs_raw, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8, label="GHz")
    ax.set_title("raw BFS map", fontsize=10)
    _bare(ax)

    # ── STEP 1 — value gate (colour IS a function of x) ──────────
    ax = fig.add_subplot(gs[1, 0])
    hist_split(ax, bfs_raw, bg_like, None, valid & ~bg_like, bins,
               [f"passes ({int(bg_like.sum())})", "",
                f"fails ({int((valid & ~bg_like).sum())})"],
               note="colour is decided by x alone\nthe split is a vertical line")
    ax.axvline(info["cut_raw"], color="navy", lw=2.0)
    ax.set_title("STEP 1 · GATE 1 value test    raw<=cut AND smoothed<=cut",
                 fontsize=11, loc="left")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(bg_like.astype(int), origin="lower", vmin=0, vmax=1,
              cmap=ListedColormap([C_FAIL, C_PASS]), interpolation="nearest")
    ax.set_title(f"passes gate 1  [{int(bg_like.sum())} px]", fontsize=10)
    _bare(ax)

    # ── STEP 2 — cell regions from the gate-1 failures ───────────
    ax = fig.add_subplot(gs[2, 0])
    sizes = np.sort(st["component_sizes"])[::-1]
    colours = [C_CELL if s >= info["min_cell_size"] else C_CUT for s in sizes]
    ax.barh(np.arange(len(sizes)), sizes, color=colours)
    ax.axvline(info["min_cell_size"], color="k", ls="--", lw=1.8,
               label=f"min_cell_size = {info['min_cell_size']} px")
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(sizes)))
    ax.set_yticklabels([f"region {i+1}" for i in range(len(sizes))], fontsize=8)
    for i, s in enumerate(sizes):
        ax.text(s, i, f" {s}", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("component size (px, log scale)")
    ax.set_title(f"STEP 2 · GATE 2a regions — open(~gate 1), keep every "
                 f"component >= {info['min_cell_size']} px\n"
                 f"{info['n_candidate']} candidates → open → "
                 f"{info['n_opened']} → {info['n_cell_regions']} cell "
                 f"region(s), {info['n_cell']} px "
                 f"({info['n_dropped_small']} px dropped as strays)",
                 fontsize=11, loc="left")
    ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="x")

    ax = fig.add_subplot(gs[2, 1])
    lab, _ = cc_label(cell)
    ax.imshow(np.where(lab > 0, lab, np.nan), origin="lower", cmap="tab10",
              vmin=1, vmax=10, interpolation="nearest")
    strays = st["opened"] & ~cell
    if strays.any():
        ax.imshow(np.where(strays, 1.0, np.nan), origin="lower",
                  cmap=ListedColormap([C_CUT]), interpolation="nearest")
    ax.set_title(f"{info['n_cell_regions']} cell region(s), colour-coded\n"
                 "orange = strays dropped by opening / size", fontsize=10)
    _bare(ax)

    # ── STEP 3 — margin gate (colour is NOT a function of x) ─────
    removed = bg_like & ~after_margin
    ax = fig.add_subplot(gs[3, 0])
    hist_split(ax, bfs_raw, after_margin, removed, None, bins,
               [f"kept ({int(after_margin.sum())})",
                f"removed by margin ({int(removed.sum())})", ""],
               note=("both colours span the SAME x range\n"
                     "this gate never looks at BFS"))
    ax.axvline(info["cut_raw"], color="navy", lw=2.0)
    ax.set_title(f"STEP 3 · GATE 2b margin — of the gate-1 survivors only "
                 f"(cityblock <= {margin} px from a CELL REGION)",
                 fontsize=11, loc="left")

    ax = fig.add_subplot(gs[3, 1])
    layer = np.zeros(shape, dtype=int)
    layer[after_margin] = 1
    layer[removed] = 2
    layer[cell] = 3
    ax.imshow(layer, origin="lower", vmin=0, vmax=3,
              cmap=ListedColormap([C_FAIL, C_PASS, C_CUT, C_CELL]),
              interpolation="nearest")
    ax.set_title("kept (blue), removed by margin (orange), cell (red)",
                 fontsize=10)
    _bare(ax)

    # ── STEP 4 — size gate + final ───────────────────────────────
    removed4 = after_margin & ~background
    ax = fig.add_subplot(gs[4, 0])
    hist_split(ax, bfs_raw, background, removed4, None, bins,
               [f"final background ({int(background.sum())})",
                f"removed by min-size ({int(removed4.sum())})", ""],
               note=f"drops background blobs under {info['min_size']} px")
    ax.axvline(info["cut_raw"], color="navy", lw=2.0)
    ax.set_title(f"STEP 4 · GATE 3 minimum background component "
                 f"({info['min_size']} px)", fontsize=11, loc="left")

    ax = fig.add_subplot(gs[4, 1])
    trimap = np.zeros(shape, dtype=int)
    trimap[cell] = 1
    trimap[background] = 2
    ax.imshow(trimap, origin="lower", vmin=0, vmax=2,
              cmap=ListedColormap([C_FAIL, C_CELL, C_PASS]),
              interpolation="nearest")
    ax.set_title(f"FINAL   bg {int(background.sum())} | "
                 f"unc {int(uncertain.sum())} | cell {int(cell.sum())}",
                 fontsize=10)
    ax.legend(handles=[Patch(facecolor=C_PASS, label="background"),
                       Patch(facecolor=C_FAIL, label="uncertain"),
                       Patch(facecolor=C_CELL, label="cell")],
              fontsize=7, loc="upper right")
    _bare(ax)

    # ── STEP 5 — the real 2D decision space ──────────────────────
    ax = fig.add_subplot(gs[5, 0])
    rng = np.random.default_rng(0)
    for mask, colour, name in ((background, C_PASS, "kept as background"),
                               (removed, C_CUT, "removed by margin")):
        if not mask.any():
            continue
        x = bfs_raw[mask]
        y = dist[mask] + rng.uniform(-0.22, 0.22, x.size)
        ax.scatter(x, y, s=5, c=colour, alpha=0.55, linewidths=0, label=name)
    ax.axvline(info["cut_raw"], color="navy", lw=2.0, label="gate 1 cut")
    ax.axhline(margin + 0.5, color="k", lw=2.0, ls="--",
               label=f"gate 2b boundary (> {margin} px)")
    ax.set_xlabel("raw BFS (GHz)")
    ax.set_ylabel("cityblock distance to nearest cell px")
    ax.set_title("STEP 5 · the decision is 2-D — x alone cannot explain it",
                 fontsize=11, loc="left")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[5, 1])
    dshow = np.where(valid, dist, np.nan).astype(float)
    im = ax.imshow(np.clip(dshow, 0, 8), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8, label="distance (px)")
    ax.contour(background.astype(float), levels=[0.5], colors="w",
               linewidths=1.0)
    ax.set_title("distance to nearest cell pixel\n"
                 "(white outline = final background)", fontsize=10)
    _bare(ax)

    fig.suptitle(f"{stem}\nhow each gate decides — one step per row",
                 fontsize=13)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    overlap_lo = max(bfs_raw[background].min(), bfs_raw[removed].min())
    overlap_hi = min(bfs_raw[background].max(), bfs_raw[removed].max())
    both = valid & (bfs_raw >= overlap_lo) & (bfs_raw <= overlap_hi)
    return dict(
        sample=stem, n_valid=int(valid.sum()), n_gate1=int(bg_like.sum()),
        n_cell_regions=info["n_cell_regions"], n_cell=int(cell.sum()),
        n_dropped_small=info["n_dropped_small"],
        n_gate2=int(after_margin.sum()), n_background=int(background.sum()),
        n_removed_margin=int(removed.sum()), n_removed_size=int(removed4.sum()),
        cut_raw=info["cut_raw"], otsu_smoothed=info["otsu_thr"],
        pct_gate1_in_overlap=float(
            100 * (both & bg_like).sum() / max(bg_like.sum(), 1)),
        median_dist_kept=float(np.median(dist[background])),
        median_dist_removed=float(np.median(dist[removed])),
    )


def process(name):
    path = bge.DATA_DIR / name
    data = bge.load_cube(path)
    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])

    valid = data["valid"]
    # Reuse the module's own gates so the diagram cannot drift from the code.
    bg_like, _ = bge.value_gate(bfs_raw, bfs_sm, valid)
    background, cell, uncertain, info = bge.find_background(
        data, bfs_raw, bfs_sm)

    opened = binary_dilation(binary_erosion(valid & ~bg_like))
    lab, n = cc_label(opened)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([], dtype=int)

    keepout = fe.dilate_mask(cell, iterations=info["margin"]).astype(bool)
    after_margin = bg_like & ~keepout

    st = dict(bg_like=bg_like, cell=cell, opened=opened,
              after_margin=after_margin, background=background,
              uncertain=uncertain, info=info, component_sizes=sizes)

    row = render(path.stem, data, bfs_raw, st,
                 OUTPUT_DIR / f"{path.stem}_step_by_step.png")
    print(f"  gate1 {row['n_gate1']} -> {row['n_cell_regions']} cell region(s) "
          f"({row['n_cell']} px, {row['n_dropped_small']} strays dropped) -> "
          f"margin {row['n_gate2']} -> bg {row['n_background']}")
    print(f"  kept/removed overlap covers "
          f"{row['pct_gate1_in_overlap']:.0f}% of gate-1 survivors | "
          f"median dist to cell: kept {row['median_dist_kept']:.0f}, "
          f"removed {row['median_dist_removed']:.0f}")
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or DEFAULT_SAMPLES
    rows = []
    for n in names:
        n = n if n.endswith(".csv") else f"{n}.csv"
        if not (bge.DATA_DIR / n).exists():
            print(f"[skip] {n}: not found")
            continue
        print(f"[{n}]")
        rows.append(process(n))
    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / "step_by_step_summary.csv", index=False)


if __name__ == "__main__":
    main()
