"""Find the shot-noise level where water/plexiglass analysis breaks down.

The measured scan is used as the best available reference.  Progressively
noisier, structurally identical CSV files are made by binomially thinning the
detected photons.  At every level the script:

1. clusters pixels into two material regions from their folded spectra;
2. matches the two labels to the reference labels (so label swapping is free);
3. averages each recovered region and fits its dominant Brillouin peak with a
   Lorentzian; and
4. records segmentation and curve-recovery metrics over repeated noise draws.

Run from the repository root, for example::

    python analysis/noise_tolerance_water_plexiglass.py

The default input is ``data/csv/water_plexiglass.csv`` and results are written
under ``outputs/noise_tolerance_water_plexiglass``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import brillouin_fitting as bf
import brillouin_preprocessing as bp


def load_scan(path: Path):
    df = pd.read_csv(path)
    raw_col = "Raw Data" if "Raw Data" in df else "RawData"
    spectra = np.vstack(df[raw_col].map(lambda s: np.fromstring(str(s), sep=",")))
    freq = np.linspace(float(df.startFrequency.iloc[0]),
                       float(df.endFrequency.iloc[0]), spectra.shape[1])
    return df, raw_col, spectra, freq


def analysis_spectra(raw, freq, max_shift=13.5):
    folded, shift = bp.fold_brillouin_spectrum(raw, freq)
    keep = shift <= max_shift
    folded, shift = folded[:, keep], shift[keep]
    # Remove per-pixel offset and brightness: regions should differ by spectral
    # shape, not by total photon count.
    folded = folded - np.percentile(folded, 5, axis=1, keepdims=True)
    folded = np.clip(folded, 0, None)
    folded = bp.sum_normalize_spectra(folded)
    return folded, shift


def display_spectra(raw, freq, photon_fraction, max_shift=13.5):
    """Fold spectra and exposure-normalise them without shape normalisation."""
    folded, shift = bp.fold_brillouin_spectrum(raw, freq)
    keep = shift <= max_shift
    folded = folded[:, keep] / max(float(photon_fraction), 1e-12)
    return folded, shift[keep]


def cluster_materials(spectra, seed):
    return KMeans(n_clusters=2, n_init=20, random_state=seed).fit_predict(spectra)


def match_labels(reference, predicted):
    table = np.zeros((2, 2), dtype=int)
    for a in range(2):
        for b in range(2):
            table[a, b] = np.count_nonzero((reference == a) & (predicted == b))
    rows, cols = linear_sum_assignment(-table)
    remap = {int(col): int(row) for row, col in zip(rows, cols)}
    matched = np.array([remap[int(v)] for v in predicted])
    return matched, float(np.mean(matched == reference))


def dominant_peak_fit(shift, mean_spectrum, expected_center=None,
                      half_window=1.15):
    if expected_center is None:
        # Water and PMMA are safely beyond the Rayleigh tail; ignoring the
        # first 3 GHz also prevents a residual elastic shoulder winning.
        eligible = shift >= 3.0
        center = float(shift[eligible][np.argmax(mean_spectrum[eligible])])
    else:
        center = float(expected_center)
    keep = np.abs(shift - center) <= half_window
    curve, params = bf.fit_lorentzian_iterative(shift[keep], mean_spectrum[keep])
    residual = mean_spectrum[keep] - curve
    denom = np.sum((mean_spectrum[keep] - mean_spectrum[keep].mean()) ** 2)
    r2 = 1.0 - np.sum(residual ** 2) / denom if denom > 0 else np.nan
    return {"center": float(params[1]), "hwhm": float(abs(params[2])),
            "r2": float(r2), "x": shift[keep], "mean": mean_spectrum[keep],
            "fit": curve}


def region_curves(spectra, labels, shift, expected_centers=None):
    out = []
    for material in range(2):
        mean = spectra[labels == material].mean(axis=0)
        expected = None if expected_centers is None else expected_centers[material]
        out.append(dominant_peak_fit(shift, mean, expected))
    return out


def fitted_bfs(spectra, shift, centers, half_window=1.15):
    """Fit the stronger of the water/PMMA peak windows in every pixel."""
    windows = [np.abs(shift - center) <= half_window for center in centers]
    bfs = np.empty(len(spectra), dtype=float)
    for i, spectrum in enumerate(spectra):
        strengths = []
        for keep in windows:
            local = spectrum[keep]
            strengths.append(float(local.max() - np.percentile(local, 10)))
        chosen = int(np.argmax(strengths))
        keep = windows[chosen]
        _, params = bf.fit_lorentzian_iterative(shift[keep], spectrum[keep],
                                                max_iter=500)
        bfs[i] = params[1]
    return bfs


def thin_photons(raw, noise_multiplier, rng):
    """Increase relative shot noise while retaining integer detector counts."""
    if noise_multiplier < 1:
        raise ValueError("noise multipliers must be >= 1")
    if noise_multiplier == 1:
        return np.rint(np.clip(raw, 0, None)).astype(np.int32)
    keep_probability = 1.0 / float(noise_multiplier) ** 2
    return rng.binomial(np.rint(np.clip(raw, 0, None)).astype(np.int64),
                        keep_probability).astype(np.int32)


def write_variant(df, raw_col, noisy, path):
    copy = df.copy()
    copy[raw_col] = [",".join(map(str, row)) for row in noisy]
    copy.to_csv(path, index=False)


def parse_levels(text):
    levels = np.array([float(v) for v in text.split(",")], dtype=float)
    if len(levels) == 0 or np.any(levels < 1) or np.any(np.diff(levels) <= 0):
        raise argparse.ArgumentTypeError("levels must increase and all be >= 1")
    return levels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data/csv/water_plexiglass.csv")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "outputs/noise_tolerance_water_plexiglass")
    parser.add_argument("--levels", type=parse_levels,
                        default=parse_levels("1,5,10,20,40,60,80,100,110,120,150"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--accuracy-threshold", type=float, default=0.90)
    parser.add_argument("--curve-r2-threshold", type=float, default=0.90)
    parser.add_argument("--grid-levels", type=parse_levels,
                        default=parse_levels("1,20,60,80,100,110,120,150"),
                        help="noise levels shown as columns in the diagnostic grid")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df, raw_col, raw, freq = load_scan(args.input)
    reference_spectra, shift = analysis_spectra(raw, freq)
    reference_labels = cluster_materials(reference_spectra, args.seed)
    reference_curves = region_curves(reference_spectra, reference_labels, shift)
    # Give stable semantic labels based on the peak positions.
    order = np.argsort([c["center"] for c in reference_curves])
    semantic = np.empty(2, dtype=int); semantic[order] = [0, 1]
    reference_labels = semantic[reference_labels]
    reference_curves = region_curves(reference_spectra, reference_labels, shift)
    expected_centers = [c["center"] for c in reference_curves]
    material_names = ["water", "plexiglass"]

    rng = np.random.default_rng(args.seed)
    missing_grid = set(args.grid_levels) - set(args.levels)
    if missing_grid:
        parser.error(f"--grid-levels must be included in --levels; missing {sorted(missing_grid)}")

    # Pixels with the most extreme water:PMMA peak ratios are the best available
    # operational definition of 100%% pure in this measured file (which has no
    # composition truth sidecar).
    water_band = np.abs(shift - expected_centers[0]) <= .45
    pmma_band = np.abs(shift - expected_centers[1]) <= .45
    water_peak = reference_spectra[:, water_band].max(axis=1)
    pmma_peak = reference_spectra[:, pmma_band].max(axis=1)
    purity_score = np.log((water_peak + 1e-12) / (pmma_peak + 1e-12))
    pure_pixels = [int(np.argmax(purity_score)), int(np.argmin(purity_score))]

    rows, examples, grid_examples = [], {}, {}
    for level in args.levels:
        for repeat in range(args.repeats):
            noisy = thin_photons(raw, level, rng)
            spectra, noisy_shift = analysis_spectra(noisy, freq)
            labels = cluster_materials(spectra, args.seed + repeat)
            labels, accuracy = match_labels(reference_labels, labels)
            curves = region_curves(spectra, labels, noisy_shift, expected_centers)
            row = {"noise_multiplier": level, "repeat": repeat,
                   "photon_fraction": 1.0 / level ** 2,
                   "matched_accuracy": accuracy,
                   "adjusted_rand_index": adjusted_rand_score(reference_labels, labels)}
            for k, name in enumerate(material_names):
                row[f"{name}_center_ghz"] = curves[k]["center"]
                row[f"{name}_center_error_ghz"] = abs(curves[k]["center"] - expected_centers[k])
                row[f"{name}_lorentzian_r2"] = curves[k]["r2"]
            rows.append(row)
            if repeat == 0:
                examples[float(level)] = (labels, curves)
                write_variant(df, raw_col, noisy,
                              args.output / f"water_plexiglass_noise_{level:g}x.csv")
                if level in args.grid_levels:
                    display, display_shift = display_spectra(
                        noisy, freq, 1.0 / level ** 2)
                    bfs = fitted_bfs(display, display_shift, expected_centers)
                    grid_examples[float(level)] = {
                        "spectra": display, "shift": display_shift,
                        "labels": labels, "bfs": bfs,
                    }

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "metrics_all_repeats.csv", index=False)
    summary = metrics.groupby("noise_multiplier").agg(
        accuracy_mean=("matched_accuracy", "mean"),
        accuracy_sd=("matched_accuracy", "std"),
        ari_mean=("adjusted_rand_index", "mean"),
        water_r2_mean=("water_lorentzian_r2", "mean"),
        plexiglass_r2_mean=("plexiglass_lorentzian_r2", "mean"),
        water_center_error_mhz=("water_center_error_ghz", lambda x: 1000*x.mean()),
        plexiglass_center_error_mhz=("plexiglass_center_error_ghz", lambda x: 1000*x.mean()),
    ).reset_index()
    summary.to_csv(args.output / "metrics_summary.csv", index=False)
    failed = summary[(summary.accuracy_mean < args.accuracy_threshold) |
                     (summary[["water_r2_mean", "plexiglass_r2_mean"]].min(axis=1)
                      < args.curve_r2_threshold)]
    break_level = None if failed.empty else float(failed.noise_multiplier.iloc[0])
    report = {"reference": str(args.input), "seed": args.seed,
              "material_reference_centers_ghz": dict(zip(material_names, expected_centers)),
              "criteria": {"matched_accuracy": args.accuracy_threshold,
                           "minimum_lorentzian_r2": args.curve_r2_threshold},
              "first_failed_noise_multiplier": break_level}
    (args.output / "breakdown.json").write_text(json.dumps(report, indent=2))

    x = summary.noise_multiplier.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].errorbar(x, summary.accuracy_mean, yerr=summary.accuracy_sd,
                        marker="o", capsize=3)
    axes[0, 0].axhline(args.accuracy_threshold, color="r", ls="--")
    axes[0, 0].set(ylabel="Matched pixel accuracy", ylim=(0.45, 1.02),
                   title="Material-region recovery")
    axes[0, 1].plot(x, summary.water_r2_mean, "o-", label="water")
    axes[0, 1].plot(x, summary.plexiglass_r2_mean, "o-", label="plexiglass")
    axes[0, 1].axhline(args.curve_r2_threshold, color="r", ls="--")
    axes[0, 1].set(ylabel="Lorentzian fit $R^2$", ylim=(0, 1.02),
                   title="Region-average curve recovery")
    axes[0, 1].legend()
    for ax, level in zip(axes[1], (args.levels[0], args.levels[-1])):
        labels, _ = examples[float(level)]
        shape = (df.Y.nunique(), df.X.nunique())
        ax.imshow(labels.reshape(shape), origin="lower", cmap="coolwarm", vmin=0, vmax=1)
        ax.set_title(f"{level:g}× noise: predicted regions")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[0]:
        ax.set_xlabel("Relative shot-noise multiplier"); ax.grid(alpha=.25)
    fig.suptitle("Water/plexiglass noise tolerance")
    fig.savefig(args.output / "noise_tolerance_summary.png", dpi=160)
    plt.close(fig)

    # Show exactly what spatial averaging recovers at each noise level.  Curves
    # are normalised only for display, so their shapes remain comparable after
    # photon thinning changes the absolute count scale.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colours = plt.cm.viridis(np.linspace(0, 1, len(args.levels)))
    for colour, level in zip(colours, args.levels):
        _, curves = examples[float(level)]
        for k, ax in enumerate(axes):
            curve = curves[k]
            scale = max(float(np.max(curve["mean"])), 1e-12)
            ax.plot(curve["x"], curve["mean"] / scale, color=colour,
                    alpha=.42, lw=1)
            ax.plot(curve["x"], curve["fit"] / scale, color=colour,
                    lw=1.5, label=f"{level:g}x")
    for name, ax in zip(material_names, axes):
        ax.set(title=f"{name.title()} region average", xlabel="Shift (GHz)",
               ylabel="Normalised intensity")
        ax.grid(alpha=.2)
    axes[1].legend(title="noise", fontsize=8, ncol=2)
    fig.savefig(args.output / "region_average_lorentzians.png", dpi=160)
    plt.close(fig)

    # Requested diagnostic grid: noise varies by column, analysis stage by row.
    ncols = len(args.grid_levels)
    fig, axes = plt.subplots(4, ncols, figsize=(3.15*ncols, 12),
                             squeeze=False, sharex="row", sharey="row",
                             constrained_layout=True)
    global_spectrum_max = max(
        np.percentile(grid_examples[float(level)]["spectra"][:,
                      grid_examples[float(level)]["shift"] >= 3], 99.8)
        for level in args.grid_levels)
    map_shape = (df.Y.nunique(), df.X.nunique())
    for col, level in enumerate(args.grid_levels):
        item = grid_examples[float(level)]
        sx, sp = item["shift"], item["spectra"]
        ax = axes[0, col]
        ax.plot(sx, sp[pure_pixels[0]], color="C0", lw=1, label="100% water")
        ax.plot(sx, sp[pure_pixels[1]], color="C1", lw=1, label="100% plexiglass")
        ax.set_title(f"{level:g}x noise")
        ax.set_xlim(3, 13.2); ax.set_ylim(0, global_spectrum_max * 1.05)
        if col == 0:
            ax.legend(fontsize=8)

        axes[1, col].imshow(item["bfs"].reshape(map_shape), origin="lower",
                            cmap="turbo", vmin=expected_centers[0]-.5,
                            vmax=expected_centers[1]+.5)
        axes[2, col].imshow(item["labels"].reshape(map_shape), origin="lower",
                            cmap="coolwarm", vmin=0, vmax=1)

        for material, colour, label in ((0, "C0", "water"),
                                        (1, "C1", "plexiglass")):
            mean = sp[item["labels"] == material].mean(axis=0)
            axes[3, col].plot(sx, mean, color=colour, lw=1.25, label=label)
        axes[3, col].set_xlim(3, 13.2); axes[3, col].set_ylim(0, global_spectrum_max*1.05)
        if col == 0:
            axes[3, col].legend(fontsize=8)

    row_labels = ["Pure-pixel raw spectra", "Fitted BFS map",
                  "Clustered material regions", "Region-averaged spectra"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label + ("\nIntensity (counts)" if row in (0, 3) else ""))
    for row in (1, 2):
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[3]:
        ax.set_xlabel("Brillouin shift (GHz)")
    for ax in axes[0]:
        ax.grid(alpha=.15)
    for ax in axes[3]:
        ax.grid(alpha=.15)
    fig.suptitle("Water/plexiglass analysis as shot noise increases")
    fig.savefig(args.output / "four_stage_noise_grid.png", dpi=170)
    plt.close(fig)

    status = "not reached" if break_level is None else f"{break_level:g}x"
    print(f"Reference centers: water={expected_centers[0]:.3f} GHz, "
          f"plexiglass={expected_centers[1]:.3f} GHz")
    print(f"First failed level: {status}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
