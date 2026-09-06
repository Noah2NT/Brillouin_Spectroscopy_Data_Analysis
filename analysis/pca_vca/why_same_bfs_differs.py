"""Why two pixels with the same BFS get different labels.

The class of a pixel is not a function of its Brillouin shift. Three gates are
applied in order, and only the first looks at the pixel's own value:

  1. VALUE    raw BFS <= cut AND smoothed BFS <= cut
  2. MARGIN   at least `margin` pixels away from ANY pixel that failed gate 1,
              or is in the cell mask, or is invalid
  3. SIZE     belongs to a background component of at least `min_size` pixels

Gate 2 is why a pixel with a perfectly background-looking shift is still
rejected: its *neighbour* is suspect, which is evidence of partial-volume
contamination that the pixel's own noisy BFS cannot reveal.

Separately, the cell mask is morphological — binary_fill_holes claims
low-shift pixels that are enclosed by the cell outline — so a pixel can be
labelled cell while sitting far below the Otsu threshold.

This script isolates one narrow raw-BFS bin and maps where its pixels landed.
"""

from pathlib import Path
import importlib.util
import sys, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import binary_fill_holes, distance_transform_edt

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

OUTPUT_DIR = Path("outputs") / "background_endmember"
SAMPLE = ("Sample9_JiPSC_1064_Treated_Plate3_Untrated_"
          "xy_scan20um_30um_step0.5um_5s_24Nov_t.csv")
BIN_LO, BIN_HI = 5.7125, 5.7175      # narrow raw-BFS slice to dissect

C_BG, C_UNC, C_CELL = "#2c7fb8", "#d9d9d9", "#c0392b"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = bge.DATA_DIR / SAMPLE
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
    thr_r = cmp_mod.otsu_threshold(bfs_raw[valid])
    med_r, mad_r = cmp_mod._robust_background_level(bfs_raw[valid], thr_r)
    thr_s = cmp_mod.otsu_threshold(bfs_sm[valid])
    med_s, mad_s = cmp_mod._robust_background_level(bfs_sm[valid], thr_s)
    cut_r, cut_s = med_r + k * mad_r, med_s + k * mad_s

    # Rebuild the intermediate stages so each rejection can be attributed.
    finite = valid & np.isfinite(bfs_raw) & np.isfinite(bfs_sm)
    bg_like = finite & (bfs_raw <= cut_r) & (bfs_sm <= cut_s)
    suspicious = (~bg_like) | cell
    suspicious_d = fe.dilate_mask(suspicious,
                                  iterations=info["margin"]).astype(bool)
    after_margin = bg_like & ~suspicious_d
    dist = distance_transform_edt(~suspicious)

    sel = valid & (bfs_raw >= BIN_LO) & (bfs_raw < BIN_HI)
    n_sel = int(sel.sum())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)

    # (1) full trimap for orientation
    trimap = np.zeros(data["shape"], dtype=int)
    trimap[cell] = 1
    trimap[bg] = 2
    axes[0].imshow(trimap, origin="lower", vmin=0, vmax=2,
                   cmap=ListedColormap([C_UNC, C_CELL, C_BG]),
                   interpolation="nearest")
    axes[0].set_title(f"full classification\nbg {int(bg.sum())} | "
                      f"unc {int(unc.sum())} | cell {int(cell.sum())}",
                      fontsize=10)

    # (2) only the pixels inside one narrow raw-BFS bin
    slice_map = np.full(data["shape"], np.nan)
    slice_map[sel & bg] = 2
    slice_map[sel & unc] = 0
    slice_map[sel & cell] = 1
    axes[1].imshow(np.zeros(data["shape"]), origin="lower", cmap="gray",
                   vmin=0, vmax=1, alpha=0.06)
    axes[1].imshow(slice_map, origin="lower", vmin=0, vmax=2,
                   cmap=ListedColormap([C_UNC, C_CELL, C_BG]),
                   interpolation="nearest")
    axes[1].contour(cell.astype(float), levels=[0.5], colors="k",
                    linewidths=0.8)
    axes[1].set_title(
        f"ONLY pixels with raw BFS in [{BIN_LO}, {BIN_HI}) — n={n_sel}\n"
        f"identical shift, three different labels", fontsize=10)
    axes[1].legend(handles=[
        Patch(facecolor=C_BG, label=f"background  {int((sel & bg).sum())}"),
        Patch(facecolor=C_UNC, label=f"uncertain  {int((sel & unc).sum())}"),
        Patch(facecolor=C_CELL, label=f"cell  {int((sel & cell).sum())}")],
        fontsize=8, loc="upper right")

    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])

    # (3) what actually separates them: distance to the nearest suspect pixel
    ax = axes[2]
    bins = np.arange(0, 8.5, 0.5)
    ax.hist([dist[sel & bg], dist[sel & unc]], bins=bins,
            color=[C_BG, "#999999"], label=["background", "uncertain"])
    ax.axvline(info["margin"], color="k", ls="--", lw=1.8,
               label=f"margin = {info['margin']} px")
    ax.set_xlabel("distance to nearest suspicious pixel (px)")
    ax.set_ylabel("pixels in the bin")
    ax.set_title("the value is identical — the geometry is not", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"{Path(SAMPLE).stem}\n"
        f"in this bin every pixel passes BOTH value gates "
        f"(cut_raw={cut_r:.4f}, cut_sm={cut_s:.4f}); "
        f"the label is decided entirely by gates 2 and 3", fontsize=11)
    out = OUTPUT_DIR / "why_same_bfs_differs.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    u = sel & unc
    c = sel & cell
    filled = binary_fill_holes((bfs_sm > thr_s) & valid)
    print(f"bin [{BIN_LO}, {BIN_HI}): {n_sel} px")
    print(f"  background {int((sel & bg).sum())} | uncertain {int(u.sum())} "
          f"| cell {int(c.sum())}")
    print(f"  uncertain failing a value gate      : {int((u & ~bg_like).sum())}")
    print(f"  uncertain killed by margin dilation : "
          f"{int((u & bg_like & suspicious_d).sum())}")
    print(f"  uncertain killed by min-size        : "
          f"{int((u & after_margin & ~bg).sum())}")
    print(f"  cell px added by binary_fill_holes  : "
          f"{int((c & ~((bfs_sm > thr_s) & valid) & filled).sum())} / {int(c.sum())}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
