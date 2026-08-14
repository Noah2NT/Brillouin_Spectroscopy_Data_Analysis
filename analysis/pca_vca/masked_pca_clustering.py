"""
Masked PCA + clustering on a 2D Brillouin scan.

Pipeline:
  1. Spatially smooth each pixel's spectrum, Lorentzian-fit to get BFS.
  2. Run VCA on the ORIGINAL (un-smoothed) spectra for endmember abundances.
  3. Sweep thresholds on BFS and each EM abundance, choose the driver +
     threshold whose mask has the most pixels while being a single
     connected blob (after erode + dilate) and barely touching the image
     border.
  4. Run PCA on the in-mask pixel spectra; K-means cluster the PC scores.
  5. Render the chosen mask + masked-PCA cluster map.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import (binary_erosion, binary_dilation,
                           gaussian_filter, label as cc_label)

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd
import brillouin_plotting as bplt
from feature_extraction import find_knee_point


# ── Config ───────────────────────────────────────────
#DATA_PATH                   = "Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv"
#DATA_PATH                   = "Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv"
DATA_PATH                   = "XY_scanning_40um_60um_step1um_10s_fixed.csv"
MAX_SHIFT_GHZ               = 8.0
NORMALIZE_SPECTRA           = True  # per-pixel min-max normalisation before
                                     # smoothing — removes brightness gradients
                                     # so clustering responds to spectral SHAPE
SMOOTH_PCA_INPUT            = 1  # None → PCA + K-Means use normalised-only
                                     #          spectra (sharper boundaries)
                                     # int  → spatial Gaussian sigma to apply to
                                     #          the PCA + K-Means input.
                                     # (Mask selection ALWAYS uses sigma=1.)
FINE_PERCENTILES            = np.arange(20, 96, 1)
ERODE_ITERATIONS            = 1
DILATE_ITERATIONS           = 1
MAX_BORDER_PIXELS           = 15
N_ENDMEMBERS_FOR_COMPARISON = 3
INNER_N_ENDMEMBERS          = 2     # VCA endmembers used inside the cell
                                     # mask to find a nucleus sub-region
MAX_INNER_BOUNDARY_PIXELS   = 5     # max nucleus pixels allowed to sit on
                                     # the cell-mask perimeter
N_PCA                       = 3
MAX_K                       = 10    # upper bound for knee-point cluster search
MASK_N_CLUSTERS             = None  # set to an int (e.g. 4) to force a fixed k
                                     # for the masked clustering — None means
                                     # auto-detect via the knee-point method
RANDOM_SEED                 = 42


# ── Helpers ─────────────────────────────────────────
def smooth_spectra(peak_spec, x_vals, y_vals, spatial_sigma):
    """Spatial Gaussian smoothing — each frequency bin is averaged across
    the 2D pixel grid."""
    if not spatial_sigma or spatial_sigma <= 0:
        return peak_spec
    ny, nx = len(y_vals), len(x_vals)
    B = peak_spec.shape[1]
    cube = peak_spec.reshape(ny, nx, B)
    smoothed_cube = np.empty_like(cube)
    for b in range(B):
        smoothed_cube[:, :, b] = gaussian_filter(cube[:, :, b],
                                                 sigma=spatial_sigma)
    return smoothed_cube.reshape(ny * nx, B)


def erode_dilate(mask, n_erode=1, n_dilate=1):
    out = mask.copy()
    for _ in range(n_erode):
        out = binary_erosion(out)
    for _ in range(n_dilate):
        out = binary_dilation(out)
    return out


def border_touch_count(mask):
    return int(mask[0, :].sum() + mask[-1, :].sum()
               + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())


def auto_orient(signal_2d):
    """Flip signal so centre reads higher than edges."""
    h, w = signal_2d.shape
    centre_mean = np.nanmean(signal_2d[h // 3:2 * h // 3,
                                       w // 3:2 * w // 3])
    edge_vals = np.concatenate([
        signal_2d[:h // 6].ravel(), signal_2d[-h // 6:].ravel(),
        signal_2d[:, :w // 6].ravel(), signal_2d[:, -w // 6:].ravel(),
    ])
    if centre_mean >= np.nanmean(edge_vals):
        return signal_2d, False
    return -signal_2d, True


def sweep_driver(signal_flat, x_vals, y_vals, label):
    """Sweep thresholds; return best mask (most pixels, 1 component,
    border touch <= MAX_BORDER_PIXELS) plus metrics."""
    px = []; comp = []; brd = []; thr = []
    for pct in FINE_PERCENTILES:
        t = float(np.nanpercentile(signal_flat, pct))
        raw = bp.reshape_to_map((signal_flat > t).astype(int),
                                y_vals, x_vals).astype(bool)
        opened = erode_dilate(raw, ERODE_ITERATIONS, DILATE_ITERATIONS)
        _, n_c = cc_label(opened)
        thr.append(t); px.append(int(opened.sum()))
        comp.append(int(n_c)); brd.append(border_touch_count(opened))
    px, comp, brd, thr = map(np.array, (px, comp, brd, thr))

    valid = (comp == 1) & (brd <= MAX_BORDER_PIXELS)
    if valid.any():
        i = int(np.argmax(np.where(valid, px, -1)))
        is_valid = True
    else:
        i = int(np.argmin(comp))
        is_valid = False

    raw = bp.reshape_to_map((signal_flat > thr[i]).astype(int),
                            y_vals, x_vals).astype(bool)
    mask = erode_dilate(raw, ERODE_ITERATIONS, DILATE_ITERATIONS)
    return {"label": label, "threshold": float(thr[i]),
            "pct": int(FINE_PERCENTILES[i]), "pixels": int(px[i]),
            "n_comp": int(comp[i]), "border": int(brd[i]),
            "valid": is_valid, "mask": mask}


def find_optimal_k(data, max_k=MAX_K, seed=RANDOM_SEED):
    """Sweep k = 2..max_k, fit KMeans for each, return the knee-point k
    via the perpendicular-distance method in feature_extraction."""
    k_range = list(range(2, max_k + 1))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(data)
        inertias.append(km.inertia_)
    return int(find_knee_point(k_range, inertias))


def choose_best_mask(peak_spec, peak_freq, x_vals, y_vals):
    """Run the full mask-selection pipeline.
    peak_spec — raw spectra (used by VCA AND by the Lorentzian fit for BFS;
                no spatial smoothing applied here).
    """
    _, fit_params = bf.fit_all_spectra(peak_freq, peak_spec)
    _, shifts, _, _ = bf.extract_fit_parameters(fit_params)
    shifts = np.clip(shifts, peak_freq.min(), peak_freq.max())

    # VCA on original (un-smoothed) spectra
    endmembers, _ = bd.vca(peak_spec, N_ENDMEMBERS_FOR_COMPARISON, seed=42)
    abundances = bd.estimate_abundances(peak_spec, endmembers)

    # Drivers list
    drivers = [(shifts, "BFS")]
    for k in range(N_ENDMEMBERS_FOR_COMPARISON):
        abund_map = bp.reshape_to_map(abundances[:, k], y_vals, x_vals)
        oriented, inv = auto_orient(abund_map)
        drivers.append((oriented.flatten(),
                        f"EM{k+1}" + (" (inv)" if inv else "")))

    results = [sweep_driver(s, x_vals, y_vals, lbl) for s, lbl in drivers]
    valid = [r for r in results if r["valid"]]
    winner = max(valid or results, key=lambda r: r["pixels"])
    return winner, shifts, abundances, results


# ── Nested (inner) mask helpers ─────────────────────
def parent_boundary_mask(parent_mask):
    """1-pixel-thick perimeter of `parent_mask`: pixels IN parent_mask
    that are adjacent to a non-parent pixel."""
    return parent_mask & ~binary_erosion(parent_mask)


def parent_boundary_touch_count(candidate, parent_mask):
    """Number of `candidate` pixels lying on the parent mask's perimeter."""
    return int((candidate & parent_boundary_mask(parent_mask)).sum())


def auto_orient_within(signal_2d, parent_mask):
    """Like auto_orient, but flips so the doubly-eroded interior of
    `parent_mask` reads higher than its 1-pixel boundary."""
    if not parent_mask.any():
        return signal_2d, False
    interior = binary_erosion(binary_erosion(parent_mask))
    boundary = parent_boundary_mask(parent_mask)
    if not interior.any() or not boundary.any():
        return signal_2d, False
    if np.nanmean(signal_2d[interior]) >= np.nanmean(signal_2d[boundary]):
        return signal_2d, False
    return -signal_2d, True


def sweep_driver_inside(signal_flat, x_vals, y_vals, parent_mask_2d,
                        max_parent_boundary, label):
    """Variant of sweep_driver restricted to inside `parent_mask_2d`:
    - percentiles computed on inside-parent values only
    - candidate mask forced to lie inside parent
    - 'border' criterion = candidate pixels on the parent mask perimeter
    """
    parent_flat = parent_mask_2d.flatten()
    inside_vals = signal_flat[parent_flat]

    px = []; comp = []; brd = []; thr = []
    for pct in FINE_PERCENTILES:
        t = float(np.nanpercentile(inside_vals, pct))
        raw_flat = (signal_flat > t) & parent_flat
        raw_map  = bp.reshape_to_map(raw_flat.astype(int),
                                     y_vals, x_vals).astype(bool)
        opened = erode_dilate(raw_map, ERODE_ITERATIONS, DILATE_ITERATIONS)
        _, n_c = cc_label(opened)
        thr.append(t); px.append(int(opened.sum()))
        comp.append(int(n_c))
        brd.append(parent_boundary_touch_count(opened, parent_mask_2d))
    px, comp, brd, thr = map(np.array, (px, comp, brd, thr))

    valid = (comp == 1) & (brd <= max_parent_boundary)
    if valid.any():
        i = int(np.argmax(np.where(valid, px, -1)))
        is_valid = True
    else:
        i = int(np.argmin(comp))
        is_valid = False

    raw_flat = (signal_flat > thr[i]) & parent_flat
    raw_map  = bp.reshape_to_map(raw_flat.astype(int),
                                 y_vals, x_vals).astype(bool)
    mask = erode_dilate(raw_map, ERODE_ITERATIONS, DILATE_ITERATIONS)
    return {"label": label, "threshold": float(thr[i]),
            "pct": int(FINE_PERCENTILES[i]), "pixels": int(px[i]),
            "n_comp": int(comp[i]), "border": int(brd[i]),
            "valid": is_valid, "mask": mask}


def choose_best_inner_mask(peak_spec, shifts, mask_flat, mask_2d,
                            x_vals, y_vals):
    """Run VCA(INNER_N_ENDMEMBERS) on inside-mask spectra, then sweep both
    the BFS shifts AND each endmember abundance to find the best inner
    sub-mask (the 'nucleus'). BFS is auto-oriented inside the cell too.
    Returns (winner_dict, abundances_full, endmembers, results).
    """
    spec_in = peak_spec[mask_flat]
    endmembers, _ = bd.vca(spec_in, INNER_N_ENDMEMBERS, seed=42)
    abundances_in = bd.estimate_abundances(spec_in, endmembers)

    # Scatter back to the full grid; outside-mask pixels stay at 0
    n_total = peak_spec.shape[0]
    abundances_full = np.zeros((n_total, INNER_N_ENDMEMBERS), dtype=float)
    abundances_full[mask_flat] = abundances_in

    drivers = []
    # BFS driver — auto-orient inside the cell so interior reads high
    shifts_map = bp.reshape_to_map(shifts, y_vals, x_vals)
    bfs_oriented, bfs_inv = auto_orient_within(shifts_map, mask_2d)
    drivers.append((bfs_oriented.flatten(),
                    "BFS" + (" (inv)" if bfs_inv else "")))
    # VCA endmember drivers
    for k in range(INNER_N_ENDMEMBERS):
        abund_map = bp.reshape_to_map(abundances_full[:, k], y_vals, x_vals)
        oriented, inv = auto_orient_within(abund_map, mask_2d)
        lbl = f"innerEM{k+1}" + (" (inv)" if inv else "")
        drivers.append((oriented.flatten(), lbl))

    results = [sweep_driver_inside(s, x_vals, y_vals, mask_2d,
                                    MAX_INNER_BOUNDARY_PIXELS, lbl)
               for s, lbl in drivers]
    valid = [r for r in results if r["valid"]]
    winner = max(valid or results, key=lambda r: r["pixels"])
    return winner, abundances_full, endmembers, results


# ── Load + Stokes/anti-Stokes folding (same as
#     Background Data Visualisations.py) ──────────────
spectra, freq, df = bp.load_brillouin_csv(DATA_PATH)
x_vals, y_vals    = bp.build_coordinate_arrays(df)

folded_spectra, shift_freq = bp.fold_brillouin_spectrum(spectra, freq)
keep = shift_freq <= MAX_SHIFT_GHZ
folded_spectra = folded_spectra[:, keep]
shift_freq     = shift_freq[keep]

# aliases so the rest of the pipeline keeps its existing names
peak_spec = folded_spectra
peak_freq = shift_freq
print(f"Folded spectra shape: {peak_spec.shape}")
print(f"Shift freq range:     {peak_freq[0]:.3f} - {peak_freq[-1]:.3f} GHz")


# ── Preprocessing branches ──
#   peak_spec       = raw folded         → VCA + BFS Lorentzian fit (un-smoothed)
#   peak_spec_norm  = normalised only    → baseline for PCA
peak_spec_norm = (bp.normalize_spectra(peak_spec)
                  if NORMALIZE_SPECTRA else peak_spec)


# ── Choose mask ────────────────────────────────────
print("Selecting best mask driver...")
winner, shifts, abundances, outer_results = choose_best_mask(
    peak_spec, peak_freq, x_vals, y_vals,
)
mask_2d   = winner["mask"]
mask_flat = mask_2d.flatten()
n_in = int(mask_flat.sum())
print(f"Winning driver : {winner['label']}")
print(f"  threshold    : {winner['threshold']:.3f}")
print(f"  pixels       : {n_in} ({100 * n_in / mask_flat.size:.1f}%)")
print(f"  components   : {winner['n_comp']}, border touch: {winner['border']}")


# ── Driver-signal vs best-mask grid for the OUTER (whole-image) sweep ──
shift_map_2d = bp.reshape_to_map(shifts, y_vals, x_vals)
signal_maps = []
em_iter = 0
for r in outer_results:
    if r["label"] == "BFS":
        signal_maps.append((r["label"], shift_map_2d, "inferno", "GHz"))
    else:
        abund_2d = bp.reshape_to_map(abundances[:, em_iter],
                                      y_vals, x_vals)
        signal_maps.append((r["label"], abund_2d, "viridis", "abundance"))
        em_iter += 1

n_outer = len(outer_results)
fig, axes = plt.subplots(2, n_outer, figsize=(3.6 * n_outer, 7))
if n_outer == 1:
    axes = axes.reshape(2, 1)

for i, (lbl, sig_map, cmap, cbar_label) in enumerate(signal_maps):
    im = axes[0, i].imshow(sig_map, origin="lower", cmap=cmap)
    axes[0, i].set_title(f"{lbl} signal", fontsize=10)
    plt.colorbar(im, ax=axes[0, i], fraction=0.046, label=cbar_label)
    axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

for i, r in enumerate(outer_results):
    is_winner = (r is winner)
    axes[1, i].imshow(r["mask"], origin="lower", cmap="gray")
    title = (f"T={r['threshold']:.2f}, {r['pixels']} px\n"
             f"comp={r['n_comp']}, border={r['border']}")
    color = ('green' if is_winner
             else 'red' if not r['valid']
             else 'black')
    axes[1, i].set_title(title, fontsize=9, color=color)
    axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

plt.suptitle("Driver signal (top) and best mask (bottom) — "
             "green = recommended, red = invalid", fontsize=12)
plt.tight_layout()
plt.show()


# ── Nested 3-region segmentation (outside / membrane / nucleus) ──
print("\nSelecting nucleus sub-mask via BFS + VCA inside cell...")
nucleus_winner, inner_abundances, inner_endmembers, inner_results = \
    choose_best_inner_mask(
        peak_spec, shifts, mask_flat, mask_2d, x_vals, y_vals,
    )
nucleus_mask_2d   = nucleus_winner["mask"]
nucleus_mask_flat = nucleus_mask_2d.flatten()
membrane_mask_2d  = mask_2d & ~nucleus_mask_2d
outside_mask_2d   = ~mask_2d

region_map = np.zeros(mask_2d.shape, dtype=int)
region_map[membrane_mask_2d] = 1
region_map[nucleus_mask_2d]  = 2
print(f"  inner driver : {nucleus_winner['label']}  "
      f"(valid={nucleus_winner['valid']})")
print(f"  threshold    : {nucleus_winner['threshold']:.4f}")
print(f"  nucleus  px  : {int(nucleus_mask_2d.sum())}")
print(f"  membrane px  : {int(membrane_mask_2d.sum())}")
print(f"  outside  px  : {int(outside_mask_2d.sum())}")
print(f"  components   : {nucleus_winner['n_comp']}, "
      f"perimeter touch: {nucleus_winner['border']}")

# [commented out — 3-region segmentation map (outside / membrane / nucleus)]
# fig, ax = plt.subplots(figsize=(7, 6))
# ax.imshow(region_map, origin="lower", cmap="tab10", vmin=0, vmax=10)
# ax.set_title(f"3-region segmentation: outside / membrane / nucleus\n"
#              f"inner driver = {nucleus_winner['label']}, "
#              f"nucleus = {int(nucleus_mask_2d.sum())} px")
# ax.set_xlabel("X"); ax.set_ylabel("Y")
# plt.tight_layout()
# plt.show()


# Driver-signal vs best-mask grid for the INNER (inside-cell) sweep
n_inner = len(inner_results)
fig, axes = plt.subplots(2, n_inner, figsize=(3.6 * n_inner, 7))
if n_inner == 1:
    axes = axes.reshape(2, 1)

em_iter = 0
for i, r in enumerate(inner_results):
    # Pick the right "signal" for the driver — BFS uses shift_map,
    # endmember drivers use their abundance map.
    if r["label"].startswith("BFS"):
        sig_map = bp.reshape_to_map(shifts, y_vals, x_vals)
        sig_display = np.where(mask_2d, sig_map, np.nan)
        cmap = "inferno"; cbar_label = "GHz"
    else:
        abund_map = bp.reshape_to_map(inner_abundances[:, em_iter],
                                       y_vals, x_vals)
        sig_display = np.where(mask_2d, abund_map, np.nan)
        cmap = "viridis"; cbar_label = "abundance"
        em_iter += 1

    im = axes[0, i].imshow(sig_display, origin="lower", cmap=cmap)
    axes[0, i].set_title(f"{r['label']} signal", fontsize=10)
    plt.colorbar(im, ax=axes[0, i], fraction=0.046, label=cbar_label)
    axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

    # Bottom row: winning sub-mask for this driver
    is_winner = (r is nucleus_winner)
    axes[1, i].imshow(r["mask"], origin="lower", cmap="gray")
    title = (f"T={r['threshold']:.2f}, {r['pixels']} px\n"
             f"comp={r['n_comp']}, border={r['border']}")
    color = ('green' if is_winner
             else 'red' if not r['valid']
             else 'black')
    axes[1, i].set_title(title, fontsize=9, color=color)
    axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

plt.suptitle("Inner driver signal (top) and best nucleus sub-mask "
             "(bottom) — green = chosen, red = invalid",
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Masked PCA + clustering ────────────────────────
# Pick the input for PCA + K-Means:
#   SMOOTH_PCA_INPUT = None → use normalised-only spectra
#   SMOOTH_PCA_INPUT = int  → smooth normalised spectra with that sigma
if SMOOTH_PCA_INPUT is None:
    pca_input = peak_spec_norm
    pca_input_label = "normalised-only"
else:
    pca_input = smooth_spectra(peak_spec_norm, x_vals, y_vals,
                               spatial_sigma=SMOOTH_PCA_INPUT)
    pca_input_label = f"smoothed (sigma={SMOOTH_PCA_INPUT})"
print(f"PCA input: {pca_input_label} spectra")

spec_in = pca_input[mask_flat]
eigvec_m, eigval_m, mean_m = bd.run_pca(spec_in)
scores_in = bd.compute_pc_scores(spec_in, eigvec_m, mean_m, N_PCA)

scores_full = np.full((peak_spec.shape[0], N_PCA), np.nan)
scores_full[mask_flat] = scores_in

# Cluster count for the masked dataset — manual override or knee-point
if MASK_N_CLUSTERS is not None:
    k_masked = int(MASK_N_CLUSTERS)
    print(f"Masked k (manual override) : {k_masked}")
else:
    k_masked = find_optimal_k(scores_in)
    print(f"Masked k (knee-point)      : {k_masked}")
km = KMeans(k_masked, random_state=RANDOM_SEED, n_init=10).fit(scores_in)
labels_full = np.full(peak_spec.shape[0], -1, dtype=int)
labels_full[mask_flat] = km.labels_


# ── Baseline PCA + KMeans on ALL pixels (matches the masked branch
#     so the comparison is apples-to-apples) ──
eigvec_all, eigval_all, mean_all = bd.run_pca(pca_input)
scores_all = bd.compute_pc_scores(pca_input, eigvec_all, mean_all, N_PCA)
k_full = find_optimal_k(scores_all)
print(f"Knee-point k (full PCA)   : {k_full}")
km_all = KMeans(k_full, random_state=RANDOM_SEED, n_init=10).fit(scores_all)

ev_m_pct   = 100 * eigval_m[:N_PCA] / eigval_m.sum()
ev_all_pct = 100 * eigval_all[:N_PCA] / eigval_all.sum()
print("\nVariance explained by top PCs:")
for k in range(N_PCA):
    print(f"  PC{k+1}:  masked={ev_m_pct[k]:5.2f}%   "
          f"all={ev_all_pct[k]:5.2f}%")


# ── Figures ────────────────────────────────────────
# [commented out — mask + masked-PCA PC1/PC2 score maps]
# bplt.plot_component_maps(
#     [mask_flat.astype(float), scores_full[:, 0], scores_full[:, 1]],
#     x_vals, y_vals,
#     labels=[f"Mask (driver={winner['label']}, "
#             f"T={winner['threshold']:.2f})",
#             "Masked PC1", "Masked PC2"],
#     suptitle="Auto-selected mask and masked-PCA scores",
#     cmap="inferno",
# )

# [commented out — side-by-side cluster map comparison (full vs masked PCA)]
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# full_map   = bp.reshape_to_map(km_all.labels_, y_vals, x_vals)
# masked_map = bp.reshape_to_map(labels_full.astype(float), y_vals, x_vals)
# masked_map[masked_map < 0] = np.nan
#
# axes[0].imshow(full_map, origin="lower", cmap="tab10")
# axes[0].set_title(f"PCA + KMeans on ALL pixels (k={k_full})")
# axes[1].imshow(masked_map, origin="lower", cmap="tab10")
# axes[1].set_title(f"PCA + KMeans on MASKED pixels only (k={k_masked})")
# for ax in axes:
#     ax.set_xlabel("X"); ax.set_ylabel("Y")
# plt.suptitle("Clustering comparison: full vs masked PCA")
# plt.tight_layout()
# plt.show()

print("\nMasked cluster sizes:")
for c in range(k_masked):
    n = int((km.labels_ == c).sum())
    print(f"  cluster {c}: {n} pixels")


# [commented out — 3D scatter of clusters in their own PCA spaces
#  (full PCA on left, masked PCA on right)]
# if N_PCA >= 3:
#     fig = plt.figure(figsize=(14, 6))
#     ax_full = fig.add_subplot(1, 2, 1, projection="3d")
#     ax_full.scatter(
#         scores_all[:, 0], scores_all[:, 1], scores_all[:, 2],
#         c=km_all.labels_, cmap="tab10", s=20, alpha=0.7,
#     )
#     ax_full.set_xlabel("PC1"); ax_full.set_ylabel("PC2")
#     ax_full.set_zlabel("PC3")
#     ax_full.set_title(f"Full-data PCA  (k={k_full})")
#     ax_mask = fig.add_subplot(1, 2, 2, projection="3d")
#     ax_mask.scatter(
#         scores_in[:, 0], scores_in[:, 1], scores_in[:, 2],
#         c=km.labels_, cmap="tab10", s=20, alpha=0.7,
#     )
#     ax_mask.set_xlabel("PC1"); ax_mask.set_ylabel("PC2")
#     ax_mask.set_zlabel("PC3")
#     ax_mask.set_title(f"Masked PCA  (k={k_masked})")
#     plt.suptitle("Clusters in PCA space (each PCA basis is independent)",
#                  fontsize=13)
#     plt.tight_layout()
#     plt.show()


# [commented out — sigma sweep cluster maps
#  (2 rows × 11 cols of cluster maps, σ from 0 to 1 in 0.1 steps,
#   top row masked, bottom row full)]
# SIGMA_SWEEP = np.round(np.arange(0.0, 1.05, 0.1), 1)
# n_cols = len(SIGMA_SWEEP)
# fig, axes = plt.subplots(2, n_cols, figsize=(2.0 * n_cols, 5.0))
# for col, sigma in enumerate(SIGMA_SWEEP):
#     sigma_f = float(sigma)
#     if sigma_f <= 0:
#         spec_for_pca = peak_spec_norm
#     else:
#         spec_for_pca = smooth_spectra(peak_spec_norm, x_vals, y_vals,
#                                        spatial_sigma=sigma_f)
#     sp_in = spec_for_pca[mask_flat]
#     ev_m_loc, _, mn_m_loc = bd.run_pca(sp_in)
#     sc_in_loc = bd.compute_pc_scores(sp_in, ev_m_loc, mn_m_loc, N_PCA)
#     km_loc = KMeans(k_masked, random_state=RANDOM_SEED,
#                     n_init=10).fit(sc_in_loc)
#     masked_labels = np.full(spec_for_pca.shape[0], -1, dtype=int)
#     masked_labels[mask_flat] = km_loc.labels_
#     masked_map_loc = bp.reshape_to_map(masked_labels.astype(float),
#                                        y_vals, x_vals)
#     masked_map_loc[masked_map_loc < 0] = np.nan
#     ev_a_loc, _, mn_a_loc = bd.run_pca(spec_for_pca)
#     sc_all_loc = bd.compute_pc_scores(spec_for_pca, ev_a_loc, mn_a_loc,
#                                        N_PCA)
#     km_all_loc = KMeans(k_full, random_state=RANDOM_SEED,
#                         n_init=10).fit(sc_all_loc)
#     all_map_loc = bp.reshape_to_map(km_all_loc.labels_, y_vals, x_vals)
#     axes[0, col].imshow(masked_map_loc, origin="lower", cmap="tab10")
#     axes[0, col].set_title(f"σ = {sigma_f:.1f}", fontsize=9)
#     axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
#     axes[1, col].imshow(all_map_loc, origin="lower", cmap="tab10")
#     axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
# axes[0, 0].set_ylabel(f"Masked\n(k={k_masked})",
#                       fontsize=10, rotation=0, ha="right", va="center")
# axes[1, 0].set_ylabel(f"All pixels\n(k={k_full})",
#                       fontsize=10, rotation=0, ha="right", va="center")
# plt.suptitle(f"PCA-input smoothing sweep — sigma 0 → 1 in 0.1 steps "
#              f"(22 cluster maps)", fontsize=12)
# plt.tight_layout()
# plt.show()


# [commented out — sigma sweep in 3D PCA space
#  (22 3D scatters, top row masked PCA, bottom row full PCA)]
# if N_PCA >= 3:
#     fig = plt.figure(figsize=(2.6 * n_cols, 6.0))
#     for col, sigma in enumerate(SIGMA_SWEEP):
#         sigma_f = float(sigma)
#         if sigma_f <= 0:
#             spec_for_pca = peak_spec_norm
#         else:
#             spec_for_pca = smooth_spectra(peak_spec_norm, x_vals, y_vals,
#                                            spatial_sigma=sigma_f)
#         sp_in = spec_for_pca[mask_flat]
#         ev_m_loc, _, mn_m_loc = bd.run_pca(sp_in)
#         sc_in_loc = bd.compute_pc_scores(sp_in, ev_m_loc, mn_m_loc, N_PCA)
#         km_loc = KMeans(k_masked, random_state=RANDOM_SEED,
#                         n_init=10).fit(sc_in_loc)
#         ev_a_loc, _, mn_a_loc = bd.run_pca(spec_for_pca)
#         sc_all_loc = bd.compute_pc_scores(spec_for_pca, ev_a_loc,
#                                            mn_a_loc, N_PCA)
#         km_all_loc = KMeans(k_full, random_state=RANDOM_SEED,
#                             n_init=10).fit(sc_all_loc)
#         ax_m = fig.add_subplot(2, n_cols, col + 1, projection="3d")
#         ax_m.scatter(sc_in_loc[:, 0], sc_in_loc[:, 1], sc_in_loc[:, 2],
#                      c=km_loc.labels_, cmap="tab10", s=6, alpha=0.7)
#         ax_m.set_title(f"σ = {sigma_f:.1f}", fontsize=9)
#         ax_m.set_xticks([]); ax_m.set_yticks([]); ax_m.set_zticks([])
#         ax_a = fig.add_subplot(2, n_cols, n_cols + col + 1,
#                                projection="3d")
#         ax_a.scatter(sc_all_loc[:, 0], sc_all_loc[:, 1], sc_all_loc[:, 2],
#                      c=km_all_loc.labels_, cmap="tab10", s=4, alpha=0.55)
#         ax_a.set_xticks([]); ax_a.set_yticks([]); ax_a.set_zticks([])
#         if col == 0:
#             ax_m.text2D(-0.15, 0.5, f"Masked\n(k={k_masked})",
#                         transform=ax_m.transAxes,
#                         ha="right", va="center", fontsize=10)
#             ax_a.text2D(-0.15, 0.5, f"All pixels\n(k={k_full})",
#                         transform=ax_a.transAxes,
#                         ha="right", va="center", fontsize=10)
#     plt.suptitle("PCA-input smoothing sweep in 3D PCA space "
#                  "(22 scatters — each uses its OWN PCA basis)",
#                  fontsize=12)
#     plt.tight_layout()
#     plt.show()


# [commented out — final side-by-side cluster comparison
#  (full PCA with background clusters merged into one, vs masked PCA
#   with an explicit extra cluster for out-of-mask pixels)]
# full_labels = km_all.labels_.copy()
# is_bg_cluster = np.zeros(k_full, dtype=bool)
# for c in range(k_full):
#     cluster_mask = full_labels == c
#     if cluster_mask.any():
#         frac_outside = float((cluster_mask & ~mask_flat).sum()) / cluster_mask.sum()
#         is_bg_cluster[c] = frac_outside > 0.5
# merged_full = np.full_like(full_labels, fill_value=-1)
# new_id = 1
# for c in range(k_full):
#     if is_bg_cluster[c]:
#         merged_full[full_labels == c] = 0
#     else:
#         merged_full[full_labels == c] = new_id
#         new_id += 1
# k_full_merged = int(merged_full.max() + 1)
# masked_labels_extended = labels_full.copy()
# bg_id = k_masked
# masked_labels_extended[~mask_flat] = bg_id
# k_masked_extended = k_masked + 1
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# full_merged_map = bp.reshape_to_map(merged_full, y_vals, x_vals)
# masked_ext_map  = bp.reshape_to_map(masked_labels_extended, y_vals, x_vals)
# axes[0].imshow(full_merged_map, origin="lower", cmap="tab10")
# axes[0].set_title(f"Full PCA, background merged (k={k_full_merged})")
# axes[1].imshow(masked_ext_map, origin="lower", cmap="tab10")
# axes[1].set_title(f"Masked PCA + explicit background (k={k_masked_extended})")
# for ax in axes:
#     ax.set_xlabel("X"); ax.set_ylabel("Y")
# plt.suptitle("Final comparison: background unified on both sides")
# plt.tight_layout()
# plt.show()
# print("\nFinal cluster counts:")
# print(f"  Full (background-merged) : {k_full_merged} total "
#       f"({k_full_merged - 1} cell clusters + 1 background)")
# print(f"  Masked (bg appended)     : {k_masked_extended} total "
#       f"({k_masked} cell clusters + 1 background)")
