"""Run and visualise the multi-band Canny detector on Brillouin cubes.

Batch mode: iterate over every 2D-scan CSV in DATA_DIR, preprocess into a
(H, W, B) spectral cube, run ``canny_edges`` on the cube (Di Zenzo
structure tensor) and on the scalar Brillouin-shift map from Lorentzian
fits, and build one combined grid figure comparing both across all
samples. Each row = one sample, columns = [BFS map, scalar Canny,
spectral Canny].
"""

from pathlib import Path
import sys
import os
import glob
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_preprocessing as bp
import brillouin_fitting as bf
from feature_extraction import canny_edges, plot_canny_stages


# Some CSVs use "Raw Data" / "Fitting Data" / "Half Width"; others use the
# space-less "RawData" / "FittingData" / "HalfWidth". Alias both.
_COLUMN_ALIASES = {
    "RawData":     "Raw Data",
    "FittingData": "Fitting Data",
    "HalfWidth":   "Half Width",
}


def _load_flex(path):
    """Read a Brillouin scan CSV tolerating both column-name variants."""
    df = pd.read_csv(path)
    renames = {src: dst for src, dst in _COLUMN_ALIASES.items()
               if src in df.columns and dst not in df.columns}
    if renames:
        df = df.rename(columns=renames)
    spectra = np.vstack(
        df["Raw Data"].apply(lambda s: np.array([float(v) for v in s.split(",")]))
    )
    start = df["startFrequency"].iloc[0]
    end   = df["endFrequency"].iloc[0]
    freq  = np.linspace(start, end, spectra.shape[1])
    return spectra, freq, df


DATA_DIR = r"C:\Users\noahk\OneDrive\Desktop\Brillouin data"
MAX_SHIFT_GHZ = 8.0

SPECTRAL_KW = dict(sigma=1.5, low_threshold=0.08, high_threshold=0.20)
BFS_KW      = dict(sigma=1.5, low_threshold=0.08, high_threshold=0.20)

OUTPUT_DIR = Path("outputs") / "canny_batch"


def load_cube(path):
    """Return (cube, shift_freq, y_vals, x_vals, bfs_map) for a 2D scan CSV."""
    spectra, freq, df = _load_flex(path)
    folded, shift_freq = bp.fold_brillouin_spectrum(spectra, freq)
    keep = shift_freq <= MAX_SHIFT_GHZ
    folded = folded[:, keep]
    shift_freq = shift_freq[keep]
    folded_norm = bp.sum_normalize_spectra(folded)

    x_vals, y_vals = bp.build_coordinate_arrays(df)
    ny, nx = len(y_vals), len(x_vals)

    # Place each row into its (y, x) grid slot; missing slots stay NaN so
    # canny_edges masks them out. Handles irregular / truncated scans.
    xi = np.searchsorted(x_vals, df["X"].to_numpy())
    yi = np.searchsorted(y_vals, df["Y"].to_numpy())
    B = folded_norm.shape[1]
    cube = np.full((ny, nx, B), np.nan)
    cube[yi, xi] = folded_norm

    _, params = bf.fit_all_spectra(shift_freq, folded_norm)
    _, shifts, _, _ = bf.extract_fit_parameters(params)
    bfs_map = np.full((ny, nx), np.nan)
    bfs_map[yi, xi] = shifts

    return cube, shift_freq, y_vals, x_vals, bfs_map


def _short(name):
    return name.rsplit(".csv", 1)[0]


def run_sample(path):
    """Load one CSV, run both cannys; return name, bfs_map, edges_bfs, edges_spec, stages_spec."""
    name = _short(os.path.basename(path))
    print(f"  loading {name} ...", flush=True)
    cube, shift_freq, y_vals, x_vals, bfs_map = load_cube(path)
    print(f"    cube {cube.shape}, bfs {bfs_map.shape}", flush=True)
    edges_spec, stages_spec = canny_edges(cube, return_stages=True, **SPECTRAL_KW)
    edges_bfs  = canny_edges(bfs_map, **BFS_KW)
    print(f"    edges: bfs={int(edges_bfs.sum())}  spectral={int(edges_spec.sum())}"
          f"  (strong={int(stages_spec['strong_edges'].sum())}, "
          f"weak={int(stages_spec['weak_edges'].sum())})",
          flush=True)
    return name, bfs_map, edges_bfs, edges_spec, stages_spec


def plot_edge_breakdown(name, stages, out_path):
    """Strong / weak / combined / final (weak-not-connected-to-strong dropped)."""
    strong = stages["strong_edges"]
    weak   = stages["weak_edges"]
    final  = stages["edges"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)

    axes[0].imshow(strong, origin="lower", cmap="gray")
    axes[0].set_title(f"Strong (hard) edges  [{int(strong.sum())}]")

    axes[1].imshow(weak, origin="lower", cmap="gray")
    axes[1].set_title(f"Weak (soft) edges  [{int(weak.sum())}]")

    # Combined: colour-code strong=red, weak=blue on white.
    combined = np.ones(strong.shape + (3,))
    combined[weak]   = [0.2, 0.4, 1.0]  # blue for weak
    combined[strong] = [1.0, 0.2, 0.2]  # red for strong
    axes[2].imshow(combined, origin="lower")
    axes[2].set_title("Strong (red) + Weak (blue)")

    axes[3].imshow(final, origin="lower", cmap="gray")
    axes[3].set_title(f"Final (hysteresis)  [{int(final.sum())}]")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name} — spectral Canny edge breakdown", fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_grid(results, out_path):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n),
                             constrained_layout=True, squeeze=False)
    for row, (name, bfs, e_bfs, e_spec, _stages) in enumerate(results):
        axes[row, 0].imshow(bfs, origin="lower", cmap="magma")
        axes[row, 0].set_ylabel(name, fontsize=8, rotation=0,
                                labelpad=90, ha="right", va="center")
        axes[row, 1].imshow(e_bfs, origin="lower", cmap="gray")
        axes[row, 2].imshow(e_spec, origin="lower", cmap="gray")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("BFS map (GHz)")
    axes[0, 1].set_title("Canny on BFS (scalar)")
    axes[0, 2].set_title("Spectral Canny (Di Zenzo)")
    fig.suptitle("Canny — scalar map vs full spectral cube (batch)")
    fig.savefig(out_path, dpi=140)
    print(f"saved grid to {out_path}")
    return fig


def plot_single(name, bfs, e_bfs, e_spec, _stages, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(bfs, origin="lower", cmap="magma")
    axes[0].set_title("BFS map (GHz)")
    fig.colorbar(im0, ax=axes[0], shrink=0.75)
    axes[1].imshow(e_bfs, origin="lower", cmap="gray")
    axes[1].set_title("Canny on BFS (scalar)")
    axes[2].imshow(e_spec, origin="lower", cmap="gray")
    axes[2].set_title("Spectral Canny (Di Zenzo)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(name, fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"found {len(csvs)} CSV(s) in {DATA_DIR}")

    results = []
    failures = []
    for path in csvs:
        name = _short(os.path.basename(path))
        try:
            r = run_sample(path)
            results.append(r)
            plot_single(*r, OUTPUT_DIR / f"{name}.png")
            plot_edge_breakdown(name, r[4],
                                OUTPUT_DIR / f"{name}_edges_breakdown.png")
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")
            traceback.print_exc()
            failures.append((name, exc))

    print(f"\n{len(results)} succeeded, {len(failures)} failed")
    if failures:
        for name, exc in failures:
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")

    if results:
        plot_grid(results, OUTPUT_DIR / "_all_samples_grid.png")
        plt.show()


if __name__ == "__main__":
    main()
