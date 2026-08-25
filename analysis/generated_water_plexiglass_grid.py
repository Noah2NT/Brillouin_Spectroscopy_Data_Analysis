"""Plot the four-stage diagnostic grid for Cell Builder generated scans."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import brillouin_preprocessing as bp
from simulator.core.writer import load_truth_sidecar

SOURCE = REPO_ROOT / "outputs/simulated_water_plexiglass"
OUTPUT = SOURCE / "generated_four_stage_noise_grid.png"


def match_to_truth(predicted, truth):
    table = np.array([[np.sum((truth == a) & (predicted == b))
                       for b in range(2)] for a in range(2)])
    rows, cols = linear_sum_assignment(-table)
    lookup = {col: row for row, col in zip(rows, cols)}
    matched = np.array([lookup[v] for v in predicted])
    return matched, float(np.mean(matched == truth))


def load_column(row):
    csv_path = SOURCE / row.csv_file
    truth_path = SOURCE / row.truth_file
    df = pd.read_csv(csv_path)
    raw = np.vstack(df["Raw Data"].map(lambda s: np.fromstring(s, sep=",")))
    freq = np.linspace(df.startFrequency.iloc[0], df.endFrequency.iloc[0], raw.shape[1])
    folded, shift = bp.fold_brillouin_spectrum(raw, freq)
    keep = shift <= 13.3
    folded, shift = folded[:, keep], shift[keep]

    truth_file = load_truth_sidecar(str(truth_path))
    names = truth_file["meta"]["material_names"]
    water_i, plexi_i = names.index("Water"), names.index("Plexiglass")
    dominant = truth_file["dominant"]
    truth = np.where(dominant == water_i, 0, 1)

    shape_spectra = folded - np.percentile(folded, 5, axis=1, keepdims=True)
    shape_spectra = bp.sum_normalize_spectra(np.clip(shape_spectra, 0, None))
    predicted = KMeans(n_clusters=2, n_init=20,
                       random_state=int(row.seed)).fit_predict(shape_spectra)
    predicted, accuracy = match_to_truth(predicted, truth)

    return {"df": df, "shift": shift, "spectra": folded,
            "bfs": df.Frequency.to_numpy(), "labels": predicted,
            "truth": truth, "purity": truth_file["purity"],
            "accuracy": accuracy}


def main():
    manifest = pd.read_csv(SOURCE / "manifest.csv").sort_values("noise_multiplier")
    columns = [(row, load_column(row)) for row in manifest.itertuples(index=False)]
    # A limit test should end at the first failed condition, not continue with
    # progressively more meaningless spectra. Keep one failing column so the
    # boundary is visible. Here failure means <90% material-region accuracy.
    first_failure = next(
        (i for i, (_, item) in enumerate(columns) if item["accuracy"] < 0.90),
        None,
    )
    if first_failure is not None:
        columns = columns[:first_failure + 1]
    ncols = len(columns)

    # Use fixed, genuinely pure truth pixels throughout every noise level.
    reference = columns[0][1]
    water_candidates = np.where((reference["truth"] == 0) &
                                np.isclose(reference["purity"], 1))[0]
    plexi_candidates = np.where((reference["truth"] == 1) &
                                np.isclose(reference["purity"], 1))[0]
    pure_pixels = [int(water_candidates[len(water_candidates)//2]),
                   int(plexi_candidates[len(plexi_candidates)//2])]

    fig, axes = plt.subplots(4, ncols, figsize=(3.15*ncols, 12), squeeze=False,
                             sharex="row", sharey=False, constrained_layout=True)
    for col, (meta, item) in enumerate(columns):
        shift, spectra = item["shift"], item["spectra"]
        photon_pct = 100 / float(meta.noise_multiplier)**2
        axes[0, col].set_title(
            f"{meta.exposure_s:.4g} s exposure\n"
            f"{meta.noise_multiplier:g}x noise ({photon_pct:.3g}% photons)",
            fontsize=9)
        axes[0, col].plot(shift, spectra[pure_pixels[0]], color="C0", lw=1,
                          label="100% water")
        axes[0, col].plot(shift, spectra[pure_pixels[1]], color="C1", lw=1,
                          label="100% plexiglass")

        axes[1, col].imshow(item["bfs"].reshape(30, 30), origin="lower",
                            cmap="turbo", vmin=5.0, vmax=12.25)
        axes[2, col].imshow(item["labels"].reshape(30, 30), origin="lower",
                            cmap="coolwarm", vmin=0, vmax=1)
        axes[2, col].text(.5, .03, f"accuracy {100*item['accuracy']:.1f}%",
                          transform=axes[2, col].transAxes, ha="center", va="bottom",
                          fontsize=8, bbox=dict(facecolor="white", alpha=.75,
                                               edgecolor="none", pad=1.5))

        for material, colour, label in ((0, "C0", "water"),
                                        (1, "C1", "plexiglass")):
            mean = spectra[item["labels"] == material].mean(axis=0)
            axes[3, col].plot(shift, mean, color=colour, lw=1.25, label=label)

    for row in (0, 3):
        for ax in axes[row]:
            ax.set_xlim(3, 13.2)
            ax.set_ylim(bottom=0)  # independent auto-scaled Y axis per column
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    axes[3, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("Pure truth pixels\nraw detector counts")
    axes[1, 0].set_ylabel("Fitted BFS map\n5.0–12.25 GHz")
    axes[2, 0].set_ylabel("Clustered regions")
    axes[3, 0].set_ylabel("Region averages\nraw detector counts")
    for row in (1, 2):
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[3]:
        ax.set_xlabel("Brillouin shift (GHz)")
    fig.suptitle("Cell Builder half-water / half-plexiglass noise sweep")
    fig.savefig(OUTPUT, dpi=170)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
