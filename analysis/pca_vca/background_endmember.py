"""Background endmember extraction for Brillouin cell scans.

Goal: find a set of pixels that are confidently *background* (medium, no cell),
co-add their spectra into a single high-SNR "pure" background endmember, then
feed that endmember into VCA as a fixed vertex so the background contribution
of every in-cell pixel becomes an explicit, subtractable abundance.

Per CSV in data/csv the script renders one figure:
  Row 1 : BFS map | cell/uncertain/background trimap | pure endmember | difference
  Row 2 : standard VCA(p)  — endmember spectra + abundance maps
  Row 3 : anchored VCA(p)  — background fixed as EM1, + abundance maps

Background selection is delegated to
cell_mask_pipeline.confident_background_mask, so this script and the cell-mask
work share one definition. That rule is stricter than "everything below Otsu":

  1. Otsu, fill, open, largest component on the smoothed BFS map -> cell mask.
  2. A robust background level (median + k*MAD of the sub-Otsu side) is applied
     to BOTH the raw and the smoothed BFS map. A pixel must read as background
     in both. Smoothing spreads cell signal into neighbours, so a partially
     filled pixel fails there even when noise pushed its raw value down.
  3. Everything that fails, plus the cell mask, plus invalid pixels, is treated
     as suspicious and dilated by `margin` — so the mixed ring around the cell
     AND the neighbourhood of any debris speck are thrown out.
  4. Connected components smaller than `min_size` are dropped as specks.

Pixels that are neither confident background nor cell are labelled 'uncertain'
and are simply not used. Precision matters here; recall does not.
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
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd
import feature_extraction as fe

# Background isolation comes from cell_mask_pipeline rather than being
# reimplemented here, so both scripts stay on one definition of "confident
# background". `confident_background_mask` is stricter than a plain
# below-Otsu rule: a pixel must read as background in BOTH the raw and the
# smoothed BFS map, and every pixel that fails is grown by a margin, so the
# mixed ring around the cell — and around any speck of debris — is excluded.
_spec = importlib.util.spec_from_file_location(
    "cell_mask_pipeline", Path(__file__).with_name("cell_mask_pipeline.py"))
cmp_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp_mod)


DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "background_endmember"

MASK_SIGMA = 0.5        # spatial blur (px) used only to stabilise the mask
MAX_SHIFT_GHZ = 8.0     # folded-axis cutoff
MIN_BG_PIXELS = 25      # below this the endmember is not trustworthy
N_ENDMEMBERS = 3

# Passed through to cell_mask_pipeline.confident_background_mask.
BG_K_MAD = cmp_mod.BG_K_MAD        # robust sigmas above the background median
BG_MARGIN = cmp_mod.BG_MARGIN      # keep-out ring grown around every suspect px
BG_MIN_SIZE = cmp_mod.BG_MIN_SIZE  # drop background blobs smaller than this

TRIMAP_CMAP = ListedColormap(["#d9d9d9", "#c0392b", "#2c7fb8"])

ALIASES = {"RawData": "Raw Data", "FittingData": "Fitting Data",
           "HalfWidth": "Half Width"}


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def load_cube(path):
    """Fold, trim and sum-normalise a scan; keep the raw folded counts too."""
    df = pd.read_csv(path).rename(columns={k: v for k, v in ALIASES.items()})
    spectra = np.vstack(df["Raw Data"].apply(lambda s: np.fromstring(s, sep=",")))
    freq = np.linspace(df["startFrequency"].iloc[0],
                       df["endFrequency"].iloc[0], spectra.shape[1])
    folded, shift = bp.fold_brillouin_spectrum(spectra, freq)
    keep = shift <= MAX_SHIFT_GHZ
    folded, shift = folded[:, keep], shift[keep]
    folded_norm = bp.sum_normalize_spectra(folded)

    x_vals, y_vals = bp.build_coordinate_arrays(df)
    ny, nx = len(y_vals), len(x_vals)
    xi = np.searchsorted(x_vals, df["X"].to_numpy())
    yi = np.searchsorted(y_vals, df["Y"].to_numpy())

    valid = np.zeros((ny, nx), dtype=bool)
    valid[yi, xi] = True

    return dict(folded_raw=folded, folded_norm=folded_norm, shift_freq=shift,
                y_vals=y_vals, x_vals=x_vals, xi=xi, yi=yi, valid=valid,
                shape=(ny, nx))


def smooth_flat(flat, yi, xi, shape, sigma):
    """Gaussian-smooth spectra over the scan grid, tolerating missing pixels.

    bp.spatial_smooth_spectra assumes N == ny*nx, which fails on rasters that
    were cut short. Normalized convolution (blur the data and the sampling
    mask, then divide) gives the same answer on a full grid and stays correct
    when pixels are absent.
    """
    if sigma <= 0:
        return flat
    ny, nx = shape
    B = flat.shape[1]
    cube = np.zeros((ny, nx, B))
    weight = np.zeros((ny, nx))
    cube[yi, xi] = flat
    weight[yi, xi] = 1.0

    den = gaussian_filter(weight, sigma=sigma)
    out = np.empty_like(cube)
    for b in range(B):
        num = gaussian_filter(cube[:, :, b], sigma=sigma)
        out[:, :, b] = np.divide(num, den, out=np.zeros_like(num),
                                 where=den > 1e-12)
    return out[yi, xi]


def bfs_from_flat(flat, shift_freq, yi, xi, shape):
    """Lorentzian-fit every spectrum; scatter peak centres back onto the grid."""
    _, params = bf.fit_all_spectra(shift_freq, flat)
    _, shifts, widths, _ = bf.extract_fit_parameters(params)
    bfs = np.full(shape, np.nan)
    bfs[yi, xi] = shifts
    return bfs, shifts, widths


# ─────────────────────────────────────────────
# Background region
# ─────────────────────────────────────────────

def otsu_threshold(values):
    """Kept as a thin alias so callers of this module keep working."""
    return cmp_mod.otsu_threshold(values)


def find_background(data, bfs_raw, bfs_smoothed):
    """Return (background_mask, cell_mask, uncertain_mask, info).

    Delegates to cell_mask_pipeline:
      * mask_from_bfs             -> the cell mask (Otsu, fill, open, largest CC)
      * confident_background_mask -> pixels confidently free of cell signal

    The background rule is deliberately not the complement of the cell mask.
    A pixel qualifies only if it reads below (median + k*MAD) of the
    background side in BOTH the raw and the spatially smoothed BFS map, and
    everything that fails is dilated by a margin, so the partial-volume ring
    around the cell and the neighbourhood of any debris speck are excluded.
    Whatever is neither confident background nor cell is reported as
    'uncertain' and simply left out of the co-add.
    """
    valid = data["valid"]
    _, cell, thr = cmp_mod.mask_from_bfs(bfs_smoothed, valid)

    background, uncertain, bg_info = cmp_mod.confident_background_mask(
        bfs_raw, bfs_smoothed, valid, cell_mask=cell,
        k_mad=BG_K_MAD, margin=BG_MARGIN, min_size=BG_MIN_SIZE)

    bg_vals = bfs_raw[background]
    info = dict(
        otsu_thr=float(thr),
        n_cell=int(cell.sum()),
        n_valid=int(valid.sum()),
        n_background=int(background.sum()),
        n_uncertain=int(uncertain.sum()),
        cut_raw=float(bg_info["cut_raw"]),
        cut_smoothed=float(bg_info["cut_smoothed"]),
        k_mad=float(bg_info["k_mad"]),
        margin=int(bg_info["margin"]),
        min_size=int(bg_info["min_size"]),
        bg_bfs_median=float(np.median(bg_vals)) if bg_vals.size else np.nan,
        bg_bfs_spread=float(np.std(bg_vals)) if bg_vals.size else np.nan,
    )
    return background, cell, uncertain, info

def build_endmember(data, bg_mask):
    """Co-add raw folded counts over background pixels -> one pure spectrum.

    Returns (endmember_norm, summed_counts, member_indices).
    """
    sel = bg_mask[data["yi"], data["xi"]]
    idx = np.flatnonzero(sel)
    summed = data["folded_raw"][idx].sum(axis=0)
    total = summed.sum()
    endmember = summed / (total if total != 0 else 1.0)
    return endmember, summed, idx


def tail_slice(n_bins):
    """Index slice of the far-from-Rayleigh baseline region."""
    return slice(n_bins - max(5, n_bins // 6), n_bins)


def baseline_noise(spectrum):
    """Successive-difference noise estimate on the baseline tail.

    std() of the raw tail is dominated by its smooth slope, so it barely
    improves on co-adding and understates the gain. Differencing adjacent
    bins cancels any smooth structure and leaves the bin-to-bin noise:
        sigma = std(diff(tail)) / sqrt(2)
    """
    tail = spectrum[tail_slice(len(spectrum))]
    return float(np.std(np.diff(tail)) / np.sqrt(2))


def peak_snr(spectrum):
    """Peak height above baseline / bin-to-bin baseline noise."""
    tail = spectrum[tail_slice(len(spectrum))]
    noise = baseline_noise(spectrum)
    if noise <= 0:
        return np.inf
    return float((spectrum.max() - np.median(tail)) / noise)


# ─────────────────────────────────────────────
# VCA with a fixed (anchored) endmember
# ─────────────────────────────────────────────

def vca_anchored(data_flat, num_endmembers, fixed, seed=42, candidates=None):
    """VCA (Nascimento & Bioucas-Dias) with the first k vertices pinned.

    `fixed` is (k, B); the simplex-growing loop starts with those columns
    already in A, so every remaining vertex is chosen in the subspace
    orthogonal to them. Returns (endmembers (p, B), indices) where the
    pinned rows carry index -1.

    `candidates` optionally restricts which pixels may be chosen as free
    vertices (the simplex geometry is still fitted to all of `data_flat`).
    Pass the in-cell indices to stop a noisy background pixel from winning
    a vertex that is supposed to describe cell material.
    """
    fixed = np.atleast_2d(fixed)
    rng = np.random.default_rng(seed)
    N, _ = data_flat.shape
    p = num_endmembers
    k = fixed.shape[0]
    if k >= p:
        raise ValueError("need at least one free endmember")

    mean_spectrum = data_flat.mean(axis=0)
    Y_centered = data_flat - mean_spectrum
    _, _, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    V_reduced = Vt[:p - 1, :]

    Y_aug = np.vstack([(Y_centered @ V_reduced.T).T, np.ones(N)])       # (p, N)
    F_aug = np.vstack([((fixed - mean_spectrum) @ V_reduced.T).T,
                       np.ones(k)])                                     # (p, k)

    A = np.zeros((p, p))
    A[:, :k] = F_aug
    indices = np.full(p, -1, dtype=int)
    pool = np.arange(N) if candidates is None else np.asarray(candidates)

    for i in range(k, p):
        w = rng.standard_normal(p)
        f = w - A @ np.linalg.pinv(A) @ w
        norm = np.linalg.norm(f)
        f = w / np.linalg.norm(w) if norm < 1e-12 else f / norm
        projections = f @ Y_aug[:, pool]
        j = int(pool[np.argmax(np.abs(projections))])
        A[:, i] = Y_aug[:, j]
        indices[i] = j

    endmembers = np.vstack([fixed, data_flat[indices[k:], :]])
    return endmembers, indices


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def scatter_map(values, yi, xi, shape):
    m = np.full(shape, np.nan)
    m[yi, xi] = values
    return m


def render(stem, data, bfs_raw, bg_mask, cell_mask, uncertain, info,
           endmember, summed, bg_idx,
           em_std, ab_std, em_anc, ab_anc, err_std, err_anc, out_path):
    shape = data["shape"]
    yi, xi, shift = data["yi"], data["xi"], data["shift_freq"]
    p = N_ENDMEMBERS

    fig = plt.figure(figsize=(4.1 * (p + 1), 11.5), constrained_layout=True)
    gs = GridSpec(3, p + 1, figure=fig)

    # ---- Row 1, panel 1: BFS map
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(bfs_raw, origin="lower", cmap="magma")
    ax.set_title("BFS map (raw fit)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="GHz")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- Row 1, panel 2: cell / uncertain / background trimap
    ax = fig.add_subplot(gs[0, 1])
    trimap = np.zeros(shape, dtype=int)      # 0 = uncertain
    trimap[cell_mask] = 1
    trimap[bg_mask] = 2
    ax.imshow(trimap, origin="lower", cmap=TRIMAP_CMAP, vmin=0, vmax=2,
              interpolation="nearest")
    if cell_mask.any():
        ax.contour(cell_mask.astype(float), levels=[0.5], colors="k",
                   linewidths=0.7)
    ax.set_title(f"background {int(bg_mask.sum())} px (blue) | "
                 f"uncertain {info['n_uncertain']} px (grey)\n"
                 f"cell {info['n_cell']} px (red) | "
                 f"k={info['k_mad']:g}, margin={info['margin']}, "
                 f"cut={info['cut_raw']:.3f} GHz", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # ---- Row 1, panel 3: the pure endmember
    mid = bg_idx[len(bg_idx) // 2]
    single_norm = data["folded_norm"][mid]
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(shift, single_norm, color="0.65", lw=1.0,
            label=f"one bg pixel (SNR≈{peak_snr(data['folded_raw'][mid]):.0f})")
    ax.plot(shift, endmember, color="tab:blue", lw=2.0,
            label=f"co-added n={len(bg_idx)} (SNR≈{peak_snr(summed):.0f})")
    cell_flat = cell_mask[yi, xi]
    if cell_flat.any():
        ax.plot(shift, data["folded_norm"][cell_flat].mean(axis=0),
                color="tab:red", lw=1.5, ls="--", label="mean in-cell")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("Norm. intensity")
    ax.set_title("pure background endmember")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)

    # Inset: the baseline tail, where the co-add gain is actually visible.
    tail = tail_slice(len(shift))
    axin = ax.inset_axes([0.56, 0.42, 0.42, 0.34])
    axin.plot(shift[tail], single_norm[tail], color="0.65", lw=0.9)
    axin.plot(shift[tail], endmember[tail], color="tab:blue", lw=1.4)
    gain = baseline_noise(single_norm) / max(baseline_noise(endmember), 1e-30)
    axin.set_title(f"baseline noise ↓{gain:.0f}×  (√n={np.sqrt(len(bg_idx)):.0f})",
                   fontsize=6.5)
    axin.tick_params(labelsize=5.5)
    axin.grid(alpha=0.3)

    # ---- Row 1, panel 4: in-cell mean vs background
    ax = fig.add_subplot(gs[0, 3])
    if cell_flat.any():
        cell_mean = data["folded_norm"][cell_flat].mean(axis=0)
        ax.plot(shift, cell_mean, color="tab:red", lw=1.2, label="mean in-cell")
        ax.plot(shift, endmember, color="tab:blue", lw=1.2,
                label="background EM")
        ax.plot(shift, cell_mean - endmember, color="k", lw=1.5,
                label="difference")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.legend(fontsize=7)
    ax.set_title("in-cell vs background")
    ax.set_xlabel("Shift (GHz)"); ax.grid(alpha=0.3)

    # ---- Rows 2 & 3: VCA comparison
    for row, (em, ab, err, label) in enumerate(
            [(em_std, ab_std, err_std, f"standard VCA({p})"),
             (em_anc, ab_anc, err_anc, f"anchored VCA({p}) — EM1 = background")],
            start=1):
        ax = fig.add_subplot(gs[row, 0])
        for j in range(p):
            style = dict(lw=2.2, color="tab:blue") if (row == 2 and j == 0) \
                else dict(lw=1.3)
            ax.plot(shift, em[j], label=f"EM {j+1}", **style)
        ax.set_title(f"{label}\nmean recon. err = {err:.2e}", fontsize=9)
        ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("Norm. intensity")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        for j in range(p):
            axj = fig.add_subplot(gs[row, j + 1])
            imj = axj.imshow(scatter_map(ab[:, j], yi, xi, shape),
                             origin="lower", cmap="viridis", vmin=0, vmax=1)
            tag = "  (background)" if (row == 2 and j == 0) else ""
            axj.set_title(f"EM {j+1} abundance{tag}", fontsize=9)
            fig.colorbar(imj, ax=axj, shrink=0.8)
            axj.set_xticks([]); axj.set_yticks([])

    fig.suptitle(
        f"{stem}\n"
        f"{info['n_valid']} valid px → {int(bg_mask.sum())} confident "
        f"background ({100 * bg_mask.sum() / max(info['n_valid'], 1):.0f}%), "
        f"{info['n_cell']} cell, {info['n_uncertain']} uncertain (excluded)",
        fontsize=11)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────

def process(path):
    stem = path.stem
    data = load_cube(path)
    ny, nx = data["shape"]
    print(f"  grid {ny}x{nx}, {data['folded_norm'].shape[0]} spectra, "
          f"{data['folded_norm'].shape[1]} bins")

    bfs_raw, _, _ = bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                  data["yi"], data["xi"], data["shape"])
    flat_sm = smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                          data["shape"], sigma=MASK_SIGMA)
    bfs_sm, _, _ = bfs_from_flat(flat_sm, data["shift_freq"],
                                 data["yi"], data["xi"], data["shape"])

    bg_mask, cell_mask, uncertain, info = find_background(data, bfs_raw, bfs_sm)
    n_bg = int(bg_mask.sum())
    print(f"  background {n_bg} px / {info['n_valid']} valid "
          f"({100 * n_bg / max(info['n_valid'], 1):.0f}%), "
          f"cell {info['n_cell']} px, uncertain {info['n_uncertain']} px "
          f"(cut {info['cut_raw']:.3f} GHz)")
    if n_bg < MIN_BG_PIXELS:
        print(f"  [skip] fewer than {MIN_BG_PIXELS} confident background px")
        return None

    endmember, summed, bg_idx = build_endmember(data, bg_mask)

    all_flat = data["folded_norm"]
    em_std, _ = bd.vca(all_flat, num_endmembers=N_ENDMEMBERS, seed=42)
    ab_std = bd.estimate_abundances(all_flat, em_std)
    err_std = float(bd.compute_reconstruction_error(all_flat, em_std,
                                                    ab_std).mean())

    em_anc, _ = vca_anchored(all_flat, N_ENDMEMBERS, endmember, seed=42)
    ab_anc = bd.estimate_abundances(all_flat, em_anc)
    err_anc = float(bd.compute_reconstruction_error(all_flat, em_anc,
                                                    ab_anc).mean())

    render(stem, data, bfs_raw, bg_mask, cell_mask, uncertain, info,
           endmember, summed, bg_idx, em_std, ab_std, em_anc, ab_anc,
           err_std, err_anc,
           OUTPUT_DIR / f"{stem}_background_endmember.png")

    np.savetxt(OUTPUT_DIR / f"{stem}_background_endmember.csv",
               np.column_stack([data["shift_freq"], endmember, summed]),
               delimiter=",", header="shift_ghz,endmember_norm,summed_counts",
               comments="")

    cell_flat = cell_mask[data["yi"], data["xi"]]
    bg_flat = bg_mask[data["yi"], data["xi"]]
    return dict(
        sample=stem, n_pixels=all_flat.shape[0],
        n_background=n_bg, n_cell=info["n_cell"],
        n_uncertain=info["n_uncertain"],
        bg_fraction=float(n_bg / max(info["n_valid"], 1)),
        otsu_ghz=info["otsu_thr"], cut_raw_ghz=info["cut_raw"],
        cut_smoothed_ghz=info["cut_smoothed"],
        bg_bfs_median=info["bg_bfs_median"],
        bg_bfs_spread=info["bg_bfs_spread"],
        snr_single=peak_snr(data["folded_raw"][bg_idx[len(bg_idx) // 2]]),
        snr_coadded=peak_snr(summed),
        noise_gain=float(
            baseline_noise(data["folded_norm"][bg_idx[len(bg_idx) // 2]])
            / max(baseline_noise(endmember), 1e-30)),
        sqrt_n=float(np.sqrt(n_bg)),
        bg_abundance_in_bg=float(np.nanmean(ab_anc[bg_flat, 0])),
        bg_abundance_in_cell=(float(np.nanmean(ab_anc[cell_flat, 0]))
                              if cell_flat.any() else np.nan),
        recon_err_standard=err_std, recon_err_anchored=err_anc,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        print(f"[{path.name}]")
        try:
            row = process(path)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            continue
        if row:
            rows.append(row)
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(OUTPUT_DIR / "background_endmember_summary.csv",
                       index=False)
        print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
