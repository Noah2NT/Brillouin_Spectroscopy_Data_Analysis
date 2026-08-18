"""Cell isolation via Gaussian-blur + Otsu, BFS-in-cell view, and VCA(2) inside the mask.

Three phases per sample:
  1. σ sweep: visualise how mask cleanliness varies with pre-blur σ.
  2. BFS-in-cell: show the Brillouin shift map restricted to the mask.
  3. VCA(2) inside: run VCA on cell-only spectra; scatter abundances back.
"""

from pathlib import Path
import sys, os, glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd
import feature_extraction as fe


SAMPLES = [
    "Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv",
    "XY_scanning_40um_60um_step1um_10s_fixed.csv",              # Sample2
    "Sample11_JiPSC-1000_Plate2_A1_FinalRawData.csv",
    "Sample12_JiPSC-1066_Plate2_A2_data1_new.csv",
]
DATA_DIR = "data/csv"
OUTPUT_DIR = Path("outputs") / "cell_mask_pipeline"

SIGMAS = [round(s, 2) for s in np.linspace(0.1, 1.0, 10)]
INNER_THRESHOLDS = [round(t, 2) for t in np.linspace(0.1, 1.0, 10)]

# Per-sample chosen σ for phases 2 + 3; picked after inspecting Phase-1 grids.
# Update these once you've seen the sweeps.
CHOSEN_SIGMA = {
    "Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv":       0.5,
    "XY_scanning_40um_60um_step1um_10s_fixed.csv":                 0.5,
    "Sample11_JiPSC-1000_Plate2_A1_FinalRawData.csv":              0.5,
    "Sample12_JiPSC-1066_Plate2_A2_data1_new.csv":                 0.5,
}

ALIASES = {"RawData": "Raw Data", "FittingData": "Fitting Data",
           "HalfWidth": "Half Width"}


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def load_cube(path):
    """Return dict with cube, folded_norm_flat, shift_freq, y/x_vals, valid_mask, bfs_map (unsmoothed)."""
    df = pd.read_csv(path).rename(columns={k: v for k, v in ALIASES.items()})
    spectra = np.vstack(
        df["Raw Data"].apply(lambda s: np.fromstring(s, sep=",")))
    freq = np.linspace(df["startFrequency"].iloc[0],
                       df["endFrequency"].iloc[0], spectra.shape[1])
    folded, shift = bp.fold_brillouin_spectrum(spectra, freq)
    keep = shift <= 8.0
    folded, shift = folded[:, keep], shift[keep]
    folded_norm = bp.sum_normalize_spectra(folded)

    x_vals, y_vals = bp.build_coordinate_arrays(df)
    ny, nx = len(y_vals), len(x_vals)
    xi = np.searchsorted(x_vals, df["X"].to_numpy())
    yi = np.searchsorted(y_vals, df["Y"].to_numpy())

    cube = np.full((ny, nx, folded_norm.shape[1]), np.nan)
    cube[yi, xi] = folded_norm
    valid = np.zeros((ny, nx), dtype=bool)
    valid[yi, xi] = True

    return dict(cube=cube, folded_norm_flat=folded_norm, shift_freq=shift,
                y_vals=y_vals, x_vals=x_vals, xi=xi, yi=yi, valid=valid)


def bfs_map_from_flat(folded_norm_flat, shift_freq, yi, xi, shape):
    _, params = bf.fit_all_spectra(shift_freq, folded_norm_flat)
    _, shifts, _, _ = bf.extract_fit_parameters(params)
    m = np.full(shape, np.nan)
    m[yi, xi] = shifts
    return m


def smooth_cube_flat(folded_norm_flat, y_vals, x_vals, sigma):
    if sigma <= 0:
        return folded_norm_flat
    return bp.spatial_smooth_spectra(folded_norm_flat, y_vals, x_vals,
                                     sigma=sigma)


# ─────────────────────────────────────────────
# Otsu + cleanup
# ─────────────────────────────────────────────

def otsu_threshold(values):
    """Otsu threshold on a 1D array of finite values."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=256)
    p = hist.astype(float) / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * ((edges[:-1] + edges[1:]) / 2))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / np.where(denom > 0, denom, 1)
    idx = int(np.nanargmax(sigma_b2))
    return 0.5 * (edges[idx] + edges[idx + 1])


def mask_from_bfs(bfs_map, valid):
    """Otsu-threshold BFS inside valid pixels; return raw and cleaned masks."""
    thr = otsu_threshold(bfs_map[valid])
    raw = (bfs_map > thr) & valid
    filled = binary_fill_holes(raw)
    opened = fe.open_mask(filled).astype(bool)
    cleaned = fe.keep_largest_component(opened)
    return raw, cleaned, thr


# ─────────────────────────────────────────────
# Phase 1 — σ sweep grid
# ─────────────────────────────────────────────

def phase1_sigma_sweep(sample_name, data, out_path):
    ny, nx = data["valid"].shape
    n = len(SIGMAS)
    fig, axes = plt.subplots(n, 3, figsize=(10, 2.8 * n),
                             constrained_layout=True, squeeze=False)
    for row, sigma in enumerate(SIGMAS):
        flat = smooth_cube_flat(data["folded_norm_flat"], data["y_vals"],
                                data["x_vals"], sigma)
        bfs = bfs_map_from_flat(flat, data["shift_freq"],
                                data["yi"], data["xi"], (ny, nx))
        raw, cleaned, thr = mask_from_bfs(bfs, data["valid"])

        axes[row, 0].imshow(bfs, origin="lower", cmap="magma")
        axes[row, 0].set_title(f"sigma={sigma}  BFS (thr={thr:.3f})", fontsize=8)
        axes[row, 1].imshow(raw, origin="lower", cmap="gray")
        axes[row, 1].set_title(f"raw Otsu mask  [{int(raw.sum())} px]", fontsize=8)
        overlay = np.where(cleaned, bfs, np.nan)
        axes[row, 2].imshow(bfs, origin="lower", cmap="gray", alpha=0.35)
        axes[row, 2].imshow(overlay, origin="lower", cmap="magma")
        axes[row, 2].set_title(f"cleaned mask  [{int(cleaned.sum())} px]",
                               fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{sample_name} — σ sweep", fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────
# Phase 2 — BFS in cell
# ─────────────────────────────────────────────

def phase2_bfs_in_cell(sample_name, data, sigma, out_path):
    """Derive the cell mask from the smoothed BFS map, but display the RAW
    (unsmoothed) BFS map — smoothing is only used for cleaner masking."""
    ny, nx = data["valid"].shape

    # Mask selection uses the smoothed cube's BFS map.
    flat_smoothed = smooth_cube_flat(data["folded_norm_flat"], data["y_vals"],
                                     data["x_vals"], sigma)
    bfs_smoothed = bfs_map_from_flat(flat_smoothed, data["shift_freq"],
                                     data["yi"], data["xi"], (ny, nx))
    _, cleaned, _ = mask_from_bfs(bfs_smoothed, data["valid"])

    # Display uses the RAW BFS map (fit from unsmoothed spectra).
    bfs_raw = bfs_map_from_flat(data["folded_norm_flat"], data["shift_freq"],
                                data["yi"], data["xi"], (ny, nx))
    masked = np.where(cleaned, bfs_raw, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(bfs_raw, origin="lower", cmap="magma")
    axes[0].set_title(f"BFS (raw, mask via sigma={sigma})")
    fig.colorbar(im0, ax=axes[0], shrink=0.75)
    im1 = axes[1].imshow(masked, origin="lower", cmap="magma")
    axes[1].set_title(f"BFS inside cell  [{int(cleaned.sum())} px]")
    fig.colorbar(im1, ax=axes[1], shrink=0.75)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(sample_name, fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return cleaned


def _flat_index(mask, xi, yi):
    """Return the boolean subset of the flat spectra array whose (yi,xi) lies in the mask."""
    return mask[yi, xi]


# ─────────────────────────────────────────────
# Phase 3 — VCA(2) inside the cell
# ─────────────────────────────────────────────

def phase3_vca_inside(sample_name, data, sigma, cleaned, out_path):
    """Run VCA(2) on RAW (unsmoothed) cell-only spectra. Return
    (nucleus_map, endmembers, nucleus_like_idx) so a downstream threshold
    sweep can reuse the results."""
    ny, nx = data["valid"].shape
    mask_flat = _flat_index(cleaned, data["xi"], data["yi"])
    n_cell = int(mask_flat.sum())
    if n_cell < 10:
        print(f"  [warn] {sample_name}: only {n_cell} cell pixels; skipping VCA")
        return None
    spec_in = data["folded_norm_flat"][mask_flat]
    endmembers, _ = bd.vca(spec_in, num_endmembers=2, seed=42)
    abundances = bd.estimate_abundances(spec_in, endmembers)   # (n_cell, 2)

    maps = []
    for k in range(2):
        m = np.full((ny, nx), np.nan)
        m[data["yi"][mask_flat], data["xi"][mask_flat]] = abundances[:, k]
        maps.append(m)

    # Label whichever endmember is more concentrated centrally as "nucleus-like".
    ys, xs = np.where(cleaned)
    cy, cx = ys.mean(), xs.mean()
    r = np.hypot(np.arange(ny)[:, None] - cy, np.arange(nx)[None, :] - cx)
    r_norm = np.where(cleaned, r, np.nan)
    r_mid = np.nanmedian(r_norm)
    inner = cleaned & (r <= r_mid)
    central_mean = [float(np.nanmean(m[inner])) for m in maps]
    nucleus_like = int(np.argmax(central_mean))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for k in range(2):
        im = axes[k].imshow(maps[k], origin="lower", cmap="viridis",
                            vmin=0, vmax=1)
        tag = "  (nucleus-like)" if k == nucleus_like else ""
        axes[k].set_title(f"EM {k+1} abundance{tag}")
        fig.colorbar(im, ax=axes[k], shrink=0.75)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    for k in range(2):
        axes[2].plot(data["shift_freq"], endmembers[k],
                     label=f"EM {k+1}", linewidth=1.5)
    axes[2].set_xlabel("Shift (GHz)")
    axes[2].set_ylabel("Normalised intensity")
    axes[2].set_title("Endmember spectra")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle(f"{sample_name} — VCA(2) inside cell (sigma={sigma}, "
                 f"n_cell={n_cell})", fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return maps[nucleus_like], endmembers, nucleus_like


# ─────────────────────────────────────────────
# Phase 4 — threshold sweep on the nucleus-like abundance map
# ─────────────────────────────────────────────

def phase4_threshold_sweep(sample_name, data, cleaned, nucleus_map,
                           nucleus_like_idx, sigma, out_path):
    """Sweep 10 thresholds on the nucleus-like abundance; render a grid so
    the user can pick which cutoff best captures the nucleus."""
    ny, nx = data["valid"].shape
    bfs = bfs_map_from_flat(data["folded_norm_flat"], data["shift_freq"],
                            data["yi"], data["xi"], (ny, nx))

    n = len(INNER_THRESHOLDS)
    fig, axes = plt.subplots(n, 3, figsize=(10, 2.6 * n),
                             constrained_layout=True, squeeze=False)
    for row, thr in enumerate(INNER_THRESHOLDS):
        raw = (nucleus_map >= thr) & cleaned
        if raw.any():
            filled = binary_fill_holes(raw)
            opened = fe.open_mask(filled).astype(bool)
            inner = fe.keep_largest_component(opened) & cleaned
        else:
            inner = raw.copy()

        # Column 1: abundance with threshold contour
        axes[row, 0].imshow(nucleus_map, origin="lower", cmap="viridis",
                            vmin=0, vmax=1)
        axes[row, 0].contour(raw.astype(float), levels=[0.5], colors="red",
                             linewidths=0.8)
        axes[row, 0].set_title(
            f"thr={thr}  EM{nucleus_like_idx + 1} abundance", fontsize=8)

        # Column 2: nucleus abundance restricted to the kept pixels
        # (same viridis 0-1 colouring as the full abundance panel).
        kept_abundance = np.where(inner, nucleus_map, np.nan)
        axes[row, 1].imshow(kept_abundance, origin="lower", cmap="viridis",
                            vmin=0, vmax=1)
        axes[row, 1].set_title(
            f"kept abundance  [{int(inner.sum())} px]", fontsize=8)

        # Column 3: BFS overlaid with cell mask (grey) + nucleus mask (magma)
        cell_only = np.where(cleaned, bfs, np.nan)
        nucleus_only = np.where(inner, bfs, np.nan)
        axes[row, 2].imshow(cell_only, origin="lower", cmap="gray", alpha=0.5)
        axes[row, 2].imshow(nucleus_only, origin="lower", cmap="magma")
        axes[row, 2].set_title(f"BFS inside nucleus", fontsize=8)

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{sample_name} — nucleus threshold sweep "
                 f"(sigma={sigma}, EM{nucleus_like_idx + 1})", fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"[skip] {name}: not in {DATA_DIR}")
            continue
        stem = Path(name).stem
        print(f"loading {stem}")
        data = load_cube(path)
        print(f"  cube {data['cube'].shape}, valid {int(data['valid'].sum())} px")

        phase1_sigma_sweep(stem, data,
                           OUTPUT_DIR / f"{stem}_sigma_sweep.png")
        print(f"  phase 1 done")

        sigma = CHOSEN_SIGMA.get(name, 1.0)
        cleaned = phase2_bfs_in_cell(stem, data, sigma,
                                     OUTPUT_DIR / f"{stem}_bfs_in_cell.png")
        print(f"  phase 2 done (sigma={sigma}, mask={int(cleaned.sum())} px)")

        vca_result = phase3_vca_inside(
            stem, data, sigma, cleaned,
            OUTPUT_DIR / f"{stem}_vca2_in_cell.png")
        print(f"  phase 3 done")

        if vca_result is not None:
            nucleus_map, _endmembers, nucleus_like = vca_result
            phase4_threshold_sweep(
                stem, data, cleaned, nucleus_map, nucleus_like, sigma,
                OUTPUT_DIR / f"{stem}_nucleus_threshold_sweep.png")
            print(f"  phase 4 done")


if __name__ == "__main__":
    main()
