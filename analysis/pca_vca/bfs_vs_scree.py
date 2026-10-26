"""BFS map beside the PCA explained-variance curve (first 5 PCs), per scan.

PCA runs on every valid pixel's sum-normalised folded spectrum.
Line = cumulative variance explained by the first x components; the label at
each point is that component's individual share.
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

import brillouin_decomposition as bd

_spec = importlib.util.spec_from_file_location(
    "background_endmember", Path(__file__).with_name("background_endmember.py"))
bge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bge)

OUTPUT_DIR = Path("outputs") / "background_endmember" / "bfs_vs_scree"
N_PC = 5


def analyse(path):
    data = bge.load_cube(path)
    bfs, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                  data["yi"], data["xi"], data["shape"])
    _, lam, _ = bd.run_pca(data["folded_norm"])
    lam = np.clip(lam, 0, None)
    evr = lam / lam.sum()
    return path.stem, bfs, evr[:N_PC]


def draw(ax_bfs, ax_pc, stem, bfs, evr, fig, compact=False):
    lo, hi = np.nanpercentile(bfs, [1, 99])
    im = ax_bfs.imshow(bfs, origin="lower", cmap="magma", vmin=lo, vmax=hi)
    fig.colorbar(im, ax=ax_bfs, shrink=0.8, label="GHz")
    ax_bfs.set_title(f"{stem[:34]}\nraw BFS" if compact else "raw BFS map",
                     fontsize=9 if compact else 11)
    ax_bfs.set_xticks([]); ax_bfs.set_yticks([])

    x = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)
    ax_pc.plot(x, cum, "o-", lw=2, color="tab:blue")
    for xi, c, e in zip(x, cum, evr):
        ax_pc.annotate(f"{e * 100:.1f}%", (xi, c), textcoords="offset points",
                       xytext=(0, -14), ha="center", fontsize=7, color="0.35")
    ax_pc.set_xticks(x)
    ax_pc.set_ylim(0, 1.05)
    ax_pc.set_xlabel("number of principal components")
    ax_pc.set_ylabel("cumulative variance explained")
    ax_pc.set_title(f"PC1={evr[0]*100:.1f}%  PC1-2={cum[1]*100:.1f}%  "
                    f"PC1-5={cum[-1]*100:.1f}%", fontsize=9 if compact else 11)
    ax_pc.grid(alpha=0.3)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results, rows = [], []
    for path in sorted(bge.DATA_DIR.glob("*.csv")):
        print(f"[{path.name}]")
        stem, bfs, evr = analyse(path)
        results.append((stem, bfs, evr))
        rows.append(dict(sample=stem,
                         **{f"PC{i+1}": float(v) for i, v in enumerate(evr)},
                         cum_PC1_2=float(evr[:2].sum()),
                         cum_PC1_5=float(evr.sum())))
        print("  " + "  ".join(f"PC{i+1} {v*100:5.1f}%"
                               for i, v in enumerate(evr))
              + f"  | cum {evr.sum()*100:.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                                 constrained_layout=True)
        draw(axes[0], axes[1], stem, bfs, evr, fig)
        fig.suptitle(stem, fontsize=11)
        fig.savefig(OUTPUT_DIR / f"{stem}_bfs_vs_scree.png", dpi=140)
        plt.close(fig)

    per_row = 2
    nrow = int(np.ceil(len(results) / per_row))
    fig, axes = plt.subplots(nrow, per_row * 2,
                             figsize=(5.0 * per_row * 2, 4.0 * nrow),
                             constrained_layout=True, squeeze=False)
    flat = axes.ravel()
    for i, (stem, bfs, evr) in enumerate(results):
        draw(flat[2 * i], flat[2 * i + 1], stem, bfs, evr, fig, compact=True)
    for ax in flat[2 * len(results):]:
        ax.axis("off")
    fig.suptitle("Raw BFS and cumulative variance explained by the first "
                 f"{N_PC} principal components", fontsize=15)
    fig.savefig(OUTPUT_DIR / "all_samples_bfs_vs_scree.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "explained_variance.csv",
                              index=False)
    print(f"\nwrote {len(results)} pairs + contact sheet to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
