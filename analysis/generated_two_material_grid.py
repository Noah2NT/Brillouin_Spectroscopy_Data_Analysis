"""Create the five-stage exposure/noise diagnostic grid for a material pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import brillouin_preprocessing as bp
from simulator.core.writer import load_truth_sidecar


def match_to_truth(predicted, truth):
    table = np.array([[np.sum((truth == a) & (predicted == b)) for b in range(2)] for a in range(2)])
    rows, cols = linear_sum_assignment(-table)
    lookup = {col: row for row, col in zip(rows, cols)}
    matched = np.array([lookup[v] for v in predicted])
    return matched, float(np.mean(matched == truth))


def load_column(source, row, material_a, material_b):
    df = pd.read_csv(source / row.csv_file)
    raw = np.vstack(df["Raw Data"].map(lambda s: np.fromstring(s, sep=",")))
    freq = np.linspace(df.startFrequency.iloc[0], df.endFrequency.iloc[0], raw.shape[1])
    spectra, shift = bp.fold_brillouin_spectrum(raw, freq)
    keep = shift <= 13.3
    spectra, shift = spectra[:, keep], shift[keep]
    truth_file = load_truth_sidecar(str(source / row.truth_file))
    names = truth_file["meta"]["material_names"]
    a_i, b_i = names.index(material_a), names.index(material_b)
    truth = np.where(truth_file["dominant"] == a_i, 0, 1)
    shapes = spectra - np.percentile(spectra, 5, axis=1, keepdims=True)
    shapes = bp.sum_normalize_spectra(np.clip(shapes, 0, None))
    predicted = KMeans(n_clusters=2, n_init=20, random_state=int(row.seed)).fit_predict(shapes)
    predicted, accuracy = match_to_truth(predicted, truth)
    return dict(df=df, shift=shift, spectra=spectra, shapes=shapes,
                bfs=df.Frequency.to_numpy(), labels=predicted, truth=truth,
                purity=truth_file["purity"], accuracy=accuracy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--material-a", default="Water")
    parser.add_argument("--material-b", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source / "generated_five_stage_noise_grid.png"
    manifest = pd.read_csv(args.source / "manifest.csv").sort_values("exposure_s", ascending=False)
    columns = [(row, load_column(args.source, row, args.material_a, args.material_b))
               for row in manifest.itertuples(index=False)]
    ncols = len(columns)

    reference = columns[0][1]
    pure_pixels = []
    for material in (0, 1):
        candidates = np.where((reference["truth"] == material) & np.isclose(reference["purity"], 1))[0]
        pure_pixels.append(int(candidates[len(candidates) // 2]))

    # One fixed PCA basis (learned at 100 s) makes movement between columns meaningful.
    pca = PCA(n_components=2).fit(reference["shapes"])
    for _, item in columns:
        item["pca"] = pca.transform(item["shapes"])
    all_pca = np.vstack([item["pca"] for _, item in columns])
    lo, hi = all_pca.min(axis=0), all_pca.max(axis=0)
    pad = np.maximum((hi - lo) * .06, 1e-8)

    medians = [np.nanmedian(reference["bfs"][reference["truth"] == i]) for i in (0, 1)]
    bfs_pad = max(.10, abs(medians[1] - medians[0]) * .15)
    bfs_min, bfs_max = min(medians) - bfs_pad, max(medians) + bfs_pad
    grid = int(round(np.sqrt(len(reference["truth"]))))
    fig, axes = plt.subplots(5, ncols, figsize=(3.15*ncols, 15), squeeze=False,
                             sharex=False, sharey=False, constrained_layout=True)
    labels = [args.material_a, args.material_b]
    for col, (meta, item) in enumerate(columns):
        shift, spectra = item["shift"], item["spectra"]
        axes[0, col].set_title(f"{meta.exposure_s:g} s exposure", fontsize=9)
        for material, colour in ((0, "C0"), (1, "C1")):
            axes[0, col].plot(shift, spectra[pure_pixels[material]], color=colour, lw=1, label=labels[material])
        axes[1, col].imshow(item["bfs"].reshape(grid, grid), origin="lower", cmap="turbo",
                            vmin=bfs_min, vmax=bfs_max)
        axes[2, col].imshow(item["labels"].reshape(grid, grid), origin="lower",
                            cmap="coolwarm", vmin=0, vmax=1)
        axes[2, col].text(.5, .03, f"accuracy {100*item['accuracy']:.1f}%", transform=axes[2, col].transAxes,
                          ha="center", va="bottom", fontsize=8,
                          bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1.5))
        for material, colour in ((0, "C0"), (1, "C1")):
            member = item["labels"] == material
            if np.any(member):
                axes[3, col].plot(shift, spectra[member].mean(axis=0), color=colour, lw=1.25, label=labels[material])
            pts = item["pca"][member]
            axes[4, col].scatter(pts[:, 0], pts[:, 1], s=6, alpha=.45, color=colour,
                                 linewidths=0, label=labels[material])
        axes[4, col].set_xlim(lo[0]-pad[0], hi[0]+pad[0]); axes[4, col].set_ylim(lo[1]-pad[1], hi[1]+pad[1])
        axes[4, col].grid(alpha=.15)

    for row in (0, 3):
        for ax in axes[row]:
            ax.set_xlim(3, 13.2); ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8); axes[3, 0].legend(fontsize=8); axes[4, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("Pure truth pixels\nraw detector counts")
    axes[1, 0].set_ylabel(f"Fitted BFS map\n{bfs_min:.2f}–{bfs_max:.2f} GHz")
    axes[2, 0].set_ylabel("Clustered regions")
    axes[3, 0].set_ylabel("Region averages\nraw detector counts")
    axes[4, 0].set_ylabel("Every pixel\nPC2")
    for row in (1, 2):
        for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[4]: ax.set_xlabel("PC1")
    fig.suptitle(f"Cell Builder half-{args.material_a.lower()} / half-{args.material_b.lower()} exposure sweep")
    fig.savefig(output, dpi=170); plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
