"""BFS map beside the final classification, for every scan in data/csv.

Two panels per sample — raw Brillouin shift, and the trimap produced by the
four gates in background_endmember.find_background:

    red   = cell       (a region of >= min_cell_size px after the opening)
    grey  = uncertain  (partial-volume ring, strays, anything not confident)
    blue  = background (the pixels that get co-added into the endmember)

Writes one PNG per sample plus a single contact sheet of all of them.
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

OUTPUT_DIR = Path("outputs") / "background_endmember" / "bfs_vs_mask"

C_BG, C_UNC, C_CELL = "#2c7fb8", "#d9d9d9", "#c0392b"
TRI_CMAP = ListedColormap([C_UNC, C_CELL, C_BG])   # 0 unc, 1 cell, 2 bg
LEGEND = [Patch(facecolor=C_CELL, label="cell"),
          Patch(facecolor=C_UNC, label="uncertain"),
          Patch(facecolor=C_BG, label="background")]


def classify(path):
    """Return (stem, bfs_raw, trimap, info)."""
    data = bge.load_cube(path)
    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])
    background, cell, _, info = bge.find_background(data, bfs_raw, bfs_sm)

    trimap = np.zeros(data["shape"], dtype=int)
    trimap[cell] = 1
    trimap[background] = 2
    return path.stem, bfs_raw, trimap, info


def draw_pair(ax_bfs, ax_mask, stem, bfs, trimap, info, fig, compact=False):
    lo, hi = np.nanpercentile(bfs, [1, 99])
    im = ax_bfs.imshow(bfs, origin="lower", cmap="magma", vmin=lo, vmax=hi)
    fig.colorbar(im, ax=ax_bfs, shrink=0.8, label="GHz")
    ax_bfs.set_title(f"{stem[:34]}\nraw BFS" if compact else "raw BFS map",
                     fontsize=9 if compact else 11)

    ax_mask.imshow(trimap, origin="lower", cmap=TRI_CMAP, vmin=0, vmax=2,
                   interpolation="nearest")
    ax_mask.set_title(
        f"cell {info['n_cell']} ({info['n_cell_regions']} region"
        f"{'s' if info['n_cell_regions'] != 1 else ''}) | "
        f"unc {info['n_uncertain']} | bg {info['n_background']}",
        fontsize=9 if compact else 11)
    if not compact:
        ax_mask.legend(handles=LEGEND, fontsize=8, loc="upper right")

    for ax in (ax_bfs, ax_mask):
        ax.set_xticks([]); ax.set_yticks([])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(bge.DATA_DIR.glob("*.csv")):
        print(f"[{path.name}]")
        try:
            stem, bfs, trimap, info = classify(path)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            continue
        results.append((stem, bfs, trimap, info))
        print(f"  cell {info['n_cell']} px in {info['n_cell_regions']} "
              f"region(s) | uncertain {info['n_uncertain']} | "
              f"background {info['n_background']}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                                 constrained_layout=True)
        draw_pair(axes[0], axes[1], stem, bfs, trimap, info, fig)
        fig.suptitle(stem, fontsize=11)
        fig.savefig(OUTPUT_DIR / f"{stem}_bfs_vs_mask.png", dpi=140)
        plt.close(fig)

    if not results:
        return

    # Contact sheet: two samples per row, each as a BFS/mask pair.
    per_row = 2
    nrow = int(np.ceil(len(results) / per_row))
    fig, axes = plt.subplots(nrow, per_row * 2,
                             figsize=(5.0 * per_row * 2, 4.2 * nrow),
                             constrained_layout=True, squeeze=False)
    flat = axes.ravel()
    for i, (stem, bfs, trimap, info) in enumerate(results):
        draw_pair(flat[2 * i], flat[2 * i + 1], stem, bfs, trimap, info, fig,
                  compact=True)
    for ax in flat[2 * len(results):]:
        ax.axis("off")
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Raw BFS and the final cell / uncertain / background "
                 "classification", fontsize=15)
    fig.savefig(OUTPUT_DIR / "all_samples_bfs_vs_mask.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {len(results)} pairs + contact sheet to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
