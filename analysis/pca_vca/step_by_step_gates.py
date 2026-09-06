"""Step-by-step: what decides each pixel's label, one gate at a time.

The walkthrough figure coloured its histogram by the FINAL label while showing
only the BFS axis, which is unreadable — the final label is not a function of
BFS. This script fixes that by advancing one gate per row and colouring each
histogram strictly by the decision made at THAT step:

  STEP 0  every valid pixel, uncoloured.
  STEP 1  VALUE gate. Colour = pass/fail of `raw <= cut AND smoothed <= cut`.
          This colouring IS a pure function of the x-axis; the split is a
          clean vertical line.
  STEP 2  MARGIN gate, applied only to step-1 survivors. Colour = kept vs
          removed. The two colours overlap across the WHOLE x range, because
          this gate does not look at BFS at all. This is the step that makes
          the final histogram look scrambled.
  STEP 3  SIZE gate, same treatment.
  STEP 4  the real decision space: BFS on x, distance-to-nearest-suspicious
          on y. The boundary is a horizontal line at y = margin. Projecting
          this plot onto x is exactly what the earlier histogram did, which
          is why the classes appeared to interleave.

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
from scipy.ndimage import distance_transform_cdt

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
cmp_mod = _load("cell_mask_pipeline")

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
    """Histogram with at most three populations, drawn kept-on-top."""
    data, colors, names = [], [], []
    for m, c, n in ((rest, C_FAIL, labels[2]), (drop, C_CUT, labels[1]),
                    (keep, C_PASS, labels[0])):
        if m is not None:
            data.append(bfs[m]); colors.append(c); names.append(n)
    ax.hist(data, bins=bins, stacked=True, color=colors, label=names,
            edgecolor="none")
    ax.set_xlabel("raw BFS (GHz)"); ax.set_ylabel("pixels")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, bbox=dict(fc="white", ec="0.7", alpha=0.9))


def render(stem, data, bfs_raw, bfs_sm, st, out_path):
    shape = data["shape"]
    valid = data["valid"]
    bins = np.linspace(*np.nanpercentile(bfs_raw[valid], [0.2, 99.8]), N_BINS)

    bg_like = st["bg_like"]
    after_margin = st["after_margin"]
    background = st["background"]
    cell = st["cell"]
    margin = st["margin"]

    # Cityblock distance to the nearest suspicious pixel — the metric the
    # 4-connected dilation actually implements.
    dist = distance_transform_cdt(~st["suspicious"], metric="taxicab")

    fig = plt.figure(figsize=(15, 24), constrained_layout=True)
    gs = GridSpec(5, 2, figure=fig, width_ratios=[1.25, 1])

    # ── STEP 0 ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(bfs_raw[valid], bins=bins, color="#999999", edgecolor="none")
    ax.axvline(st["cut_raw"], color="navy", lw=2.0,
               label=f"cut = med+{st['k']:g}·MAD = {st['cut_raw']:.4f}")
    ax.axvline(st["thr_sm"], color="k", ls="--", lw=1.6,
               label=f"Otsu = {st['thr_sm']:.4f}")
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
    ax.axvline(st["cut_raw"], color="navy", lw=2.0)
    ax.set_title("STEP 1 · GATE 1 value test    raw<=cut AND smoothed<=cut",
                 fontsize=11, loc="left")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(bg_like.astype(int), origin="lower", vmin=0, vmax=1,
              cmap=ListedColormap([C_FAIL, C_PASS]), interpolation="nearest")
    ax.set_title(f"passes gate 1  [{int(bg_like.sum())} px]", fontsize=10)
    _bare(ax)

    # ── STEP 2 — margin gate (colour is NOT a function of x) ─────
    removed = bg_like & ~after_margin
    ax = fig.add_subplot(gs[2, 0])
    hist_split(ax, bfs_raw, after_margin, removed, None, bins,
               [f"kept ({int(after_margin.sum())})",
                f"removed by margin ({int(removed.sum())})", ""],
               note=("both colours span the SAME x range\n"
                     "this gate never looks at BFS"))
    ax.axvline(st["cut_raw"], color="navy", lw=2.0)
    ax.set_title(f"STEP 2 · GATE 2 margin — of the gate-1 survivors only "
                 f"(cityblock <= {margin} px from a suspicious pixel)",
                 fontsize=11, loc="left")

    ax = fig.add_subplot(gs[2, 1])
    layer = np.zeros(shape, dtype=int)
    layer[after_margin] = 1
    layer[removed] = 2
    ax.imshow(layer, origin="lower", vmin=0, vmax=2,
              cmap=ListedColormap([C_FAIL, C_PASS, C_CUT]),
              interpolation="nearest")
    ax.set_title("kept (blue) vs removed by margin (orange)", fontsize=10)
    _bare(ax)

    # ── STEP 3 — size gate ───────────────────────────────────────
    removed3 = after_margin & ~background
    ax = fig.add_subplot(gs[3, 0])
    hist_split(ax, bfs_raw, background, removed3, None, bins,
               [f"final background ({int(background.sum())})",
                f"removed by min-size ({int(removed3.sum())})", ""],
               note=f"drops blobs smaller than {st['min_size']} px")
    ax.axvline(st["cut_raw"], color="navy", lw=2.0)
    ax.set_title(f"STEP 3 · GATE 3 minimum component size "
                 f"({st['min_size']} px)", fontsize=11, loc="left")

    ax = fig.add_subplot(gs[3, 1])
    trimap = np.zeros(shape, dtype=int)
    trimap[cell] = 1
    trimap[background] = 2
    ax.imshow(trimap, origin="lower", vmin=0, vmax=2,
              cmap=ListedColormap([C_FAIL, C_CELL, C_PASS]),
              interpolation="nearest")
    ax.set_title(f"FINAL   bg {int(background.sum())} | "
                 f"unc {int(st['uncertain'].sum())} | cell {int(cell.sum())}",
                 fontsize=10)
    ax.legend(handles=[Patch(facecolor=C_PASS, label="background"),
                       Patch(facecolor=C_FAIL, label="uncertain"),
                       Patch(facecolor=C_CELL, label="cell")],
              fontsize=7, loc="upper right")
    _bare(ax)

    # ── STEP 4 — the real 2D decision space ──────────────────────
    ax = fig.add_subplot(gs[4, 0])
    rng = np.random.default_rng(0)
    for mask, colour, name in ((background, C_PASS, "kept as background"),
                               (removed, C_CUT, "removed by margin")):
        x = bfs_raw[mask]
        y = dist[mask] + rng.uniform(-0.22, 0.22, x.size)   # jitter integers
        ax.scatter(x, y, s=5, c=colour, alpha=0.55, linewidths=0, label=name)
    ax.axvline(st["cut_raw"], color="navy", lw=2.0, label="gate 1 cut")
    ax.axhline(margin + 0.5, color="k", lw=2.0, ls="--",
               label=f"gate 2 boundary (> {margin} px)")
    ax.set_xlabel("raw BFS (GHz)")
    ax.set_ylabel("cityblock distance to nearest suspicious px")
    ax.set_title("STEP 4 · the decision is 2-D — x alone cannot explain it",
                 fontsize=11, loc="left")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[4, 1])
    dshow = np.where(valid, dist, np.nan).astype(float)
    im = ax.imshow(np.clip(dshow, 0, 8), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8, label="distance (px)")
    ax.contour(background.astype(float), levels=[0.5], colors="w",
               linewidths=1.0)
    ax.set_title("distance to nearest suspicious pixel\n"
                 "(white outline = final background)", fontsize=10)
    _bare(ax)

    fig.suptitle(f"{stem}\nhow each gate decides — one step per row",
                 fontsize=13)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # Numbers that back the "x cannot explain it" claim.
    overlap_lo = max(bfs_raw[background].min(), bfs_raw[removed].min())
    overlap_hi = min(bfs_raw[background].max(), bfs_raw[removed].max())
    both = valid & (bfs_raw >= overlap_lo) & (bfs_raw <= overlap_hi)
    return dict(
        sample=stem, n_valid=int(valid.sum()), n_gate1=int(bg_like.sum()),
        n_gate2=int(after_margin.sum()), n_background=int(background.sum()),
        n_removed_margin=int(removed.sum()), n_removed_size=int(removed3.sum()),
        cut_raw=st["cut_raw"], otsu_smoothed=st["thr_sm"],
        overlap_lo_ghz=float(overlap_lo), overlap_hi_ghz=float(overlap_hi),
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
    k, margin, min_size = bge.BG_K_MAD, bge.BG_MARGIN, bge.BG_MIN_SIZE
    _, cell, thr_s = cmp_mod.mask_from_bfs(bfs_sm, valid)
    thr_r = cmp_mod.otsu_threshold(bfs_raw[valid])
    med_r, mad_r = cmp_mod._robust_background_level(bfs_raw[valid], thr_r)
    med_s, mad_s = cmp_mod._robust_background_level(bfs_sm[valid], thr_s)
    cut_r, cut_s = med_r + k * mad_r, med_s + k * mad_s

    finite = valid & np.isfinite(bfs_raw) & np.isfinite(bfs_sm)
    bg_like = finite & (bfs_raw <= cut_r) & (bfs_sm <= cut_s)
    suspicious = (~bg_like) | cell
    suspicious_d = fe.dilate_mask(suspicious, iterations=margin).astype(bool)
    after_margin = bg_like & ~suspicious_d
    background = cmp_mod._keep_components_at_least(after_margin, min_size)
    uncertain = valid & ~(background | cell)

    st = dict(cell=cell, bg_like=bg_like, suspicious=suspicious,
              after_margin=after_margin, background=background,
              uncertain=uncertain, thr_sm=thr_s, cut_raw=cut_r, cut_sm=cut_s,
              k=k, margin=margin, min_size=min_size)

    row = render(path.stem, data, bfs_raw, bfs_sm, st,
                 OUTPUT_DIR / f"{path.stem}_step_by_step.png")
    print(f"  gate1 {row['n_gate1']} -> gate2 {row['n_gate2']} -> "
          f"bg {row['n_background']}  (margin removed "
          f"{row['n_removed_margin']}, size removed {row['n_removed_size']})")
    print(f"  kept and removed overlap in BFS over "
          f"[{row['overlap_lo_ghz']:.4f}, {row['overlap_hi_ghz']:.4f}] GHz "
          f"= {row['pct_gate1_in_overlap']:.0f}% of gate-1 survivors")
    print(f"  median distance to suspicious px: kept "
          f"{row['median_dist_kept']:.0f}, removed "
          f"{row['median_dist_removed']:.0f}")
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
