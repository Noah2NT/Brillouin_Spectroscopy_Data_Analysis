"""cv2.Canny baseline on BFS and linewidth maps for every 2D scan CSV."""

from pathlib import Path
import sys, os, glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_preprocessing as bp
import brillouin_fitting as bf

DATA_DIR   = "data/csv"
OUTPUT_DIR = Path("outputs") / "canny_cv2"
LOW, HIGH  = 60, 150   # cv2.Canny uint8 thresholds

ALIASES = {"RawData": "Raw Data", "FittingData": "Fitting Data", "HalfWidth": "Half Width"}


def load_maps(path):
    df = pd.read_csv(path).rename(columns={k: v for k, v in ALIASES.items()})
    spectra = np.vstack(df["Raw Data"].apply(
        lambda s: np.fromstring(s, sep=",")))
    freq = np.linspace(df["startFrequency"].iloc[0], df["endFrequency"].iloc[0],
                       spectra.shape[1])
    folded, shift = bp.fold_brillouin_spectrum(spectra, freq)
    folded, shift = folded[:, shift <= 8], shift[shift <= 8]
    _, params = bf.fit_all_spectra(shift, bp.sum_normalize_spectra(folded))
    _, shifts, linewidths, _ = bf.extract_fit_parameters(params)
    x, y = bp.build_coordinate_arrays(df)
    xi = np.searchsorted(x, df["X"]); yi = np.searchsorted(y, df["Y"])
    bfs = np.full((len(y), len(x)), np.nan); bfs[yi, xi] = shifts
    lw  = np.full((len(y), len(x)), np.nan); lw[yi, xi]  = linewidths
    return bfs, lw


def to_u8(img):
    m = np.isfinite(img)
    if not m.any():
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.nanmin(img), np.nanmax(img)
    out = np.where(m, (img - lo) / max(hi - lo, 1e-12) * 255, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        name = Path(path).stem
        print(f"processing {name}")
        try:
            bfs, lw = load_maps(path)
        except Exception as e:
            print(f"  skip: {e}"); continue

        bfs_u8, lw_u8 = to_u8(bfs), to_u8(lw)
        e_bfs = cv2.Canny(bfs_u8, LOW, HIGH)
        e_lw  = cv2.Canny(lw_u8,  LOW, HIGH)

        fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
        axes[0, 0].imshow(bfs, origin="lower", cmap="magma");   axes[0, 0].set_title("BFS map")
        axes[0, 1].imshow(e_bfs, origin="lower", cmap="gray");  axes[0, 1].set_title(f"cv2.Canny BFS ({int((e_bfs > 0).sum())} px)")
        axes[1, 0].imshow(lw,  origin="lower", cmap="viridis"); axes[1, 0].set_title("Linewidth map")
        axes[1, 1].imshow(e_lw,  origin="lower", cmap="gray");  axes[1, 1].set_title(f"cv2.Canny linewidth ({int((e_lw > 0).sum())} px)")
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(name, fontsize=10)
        fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
