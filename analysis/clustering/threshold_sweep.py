"""
Threshold-mask sweep for the Cell_Col Healthy 2D scan.

Computes the Brillouin frequency shift at every pixel (Lorentzian fit),
then sweeps several threshold values to show how the mask changes.

Two flavours of each mask are shown:
  - RAW:     just `shift > threshold` (noisy — isolated hot pixels outside,
             holes inside).
  - CLEANED: raw mask after morphological opening + filling holes + keeping
             ONLY the largest connected component. This is what guarantees
             the lit pixels are inside one contiguous blob (i.e. the cell).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_opening, binary_fill_holes,
                           binary_erosion, binary_dilation,
                           gaussian_filter, label as cc_label)

import brillouin_preprocessing as bp
import brillouin_fitting as bf


# ── Config ───────────────────────────────────────────
DATA_PATH      = "Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv"
PEAK_BIN_START = 305
PEAK_BIN_END   = 405
MAX_SHIFT_GHZ  = 8.0

# Threshold values to sweep. Defined as percentiles of the per-pixel
# shift distribution so the values automatically scale to whatever
# dataset you swap in.
THRESHOLD_PERCENTILES = [40, 50, 60, 70, 80, 90]

# Morphology parameters
OPENING_ITERATIONS = 1   # 0 to disable noise-removal
FILL_HOLES         = True

# Pre-mask Gaussian spatial smoothing of every signal map (BFS + abundances)
SMOOTH_SIGMA       = 1.0   # set to 0 to disable


# ── Load + crop ─────────────────────────────────────
spectra, freq, df = bp.load_brillouin_csv(DATA_PATH)
x_vals, y_vals    = bp.build_coordinate_arrays(df)

peak_spec, peak_freq = bp.extract_frequency_window(
    spectra, freq, PEAK_BIN_START, PEAK_BIN_END,
)
keep = peak_freq <= MAX_SHIFT_GHZ
peak_spec = peak_spec[:, keep]
peak_freq = peak_freq[keep]
print(f"Loaded {peak_spec.shape[0]} pixel spectra")


# ── Spatially smooth each pixel's spectrum BEFORE Lorentzian fit ──
# Smoother input → cleaner Lorentzian fits → cleaner BFS map → cleaner
# mask. VCA later still uses the ORIGINAL (un-smoothed) spectra.
def _spatial_smooth_spectra(peak_spec, x_vals, y_vals, sigma):
    if sigma is None or sigma <= 0:
        return peak_spec
    ny, nx = len(y_vals), len(x_vals)
    B = peak_spec.shape[1]
    cube = peak_spec.reshape(ny, nx, B)
    out = np.empty_like(cube)
    for b in range(B):
        out[:, :, b] = gaussian_filter(cube[:, :, b], sigma=sigma)
    return out.reshape(ny * nx, B)


peak_spec_smoothed = _spatial_smooth_spectra(
    peak_spec, x_vals, y_vals, SMOOTH_SIGMA,
)

# Visualise the smoothing on one example pixel
demo_idx = peak_spec.shape[0] // 2
fig_demo, ax = plt.subplots(figsize=(7, 4))
ax.plot(peak_freq, peak_spec[demo_idx], 'b-', alpha=0.5,
        linewidth=1.0, label='raw spectrum')
ax.plot(peak_freq, peak_spec_smoothed[demo_idx], 'r-',
        linewidth=1.6, label=f'spatially smoothed (σ={SMOOTH_SIGMA})')
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Intensity")
ax.set_title(f"Pixel {demo_idx}: raw vs spatially-smoothed spectrum")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ── Lorentzian fit on the smoothed spectra ─────────
print("Fitting Lorentzians (on spatially-smoothed spectra)...")
_, fit_params = bf.fit_all_spectra(peak_freq, peak_spec_smoothed)
_, shifts, _, _ = bf.extract_fit_parameters(fit_params)
shifts = np.clip(shifts, peak_freq.min(), peak_freq.max())
shift_map = bp.reshape_to_map(shifts, y_vals, x_vals)


# ── Mask cleanup helpers ───────────────────────────
def clean_mask(raw_mask):
    """Morphological opening, hole-filling, and largest-component keep.

    Guarantees the result is a single connected blob (or empty if the
    raw mask was already noise-only).
    """
    mask = raw_mask.copy()
    if OPENING_ITERATIONS > 0:
        mask = binary_opening(mask, iterations=OPENING_ITERATIONS)
    if FILL_HOLES:
        mask = binary_fill_holes(mask)

    # Keep the largest connected component only
    labelled, n_components = cc_label(mask)
    if n_components <= 1:
        return mask.astype(bool), n_components
    sizes = np.array([(labelled == k).sum()
                      for k in range(1, n_components + 1)])
    largest_label = 1 + int(np.argmax(sizes))
    return (labelled == largest_label), n_components


# ── Sweep ──────────────────────────────────────────
n_panels = len(THRESHOLD_PERCENTILES) + 1     # +1 for the BFS reference
fig, axes = plt.subplots(2, n_panels, figsize=(3 * n_panels, 6))

# Top-left: BFS heatmap for reference
ax = axes[0, 0]
im = ax.imshow(shift_map, origin="lower", cmap="inferno")
ax.set_title("BFS heatmap\n(reference)")
plt.colorbar(im, ax=ax, label="GHz")
axes[1, 0].axis("off")   # nothing to put under the reference

print(f"\n{'pct':>4} {'threshold':>11} {'raw %':>7} {'clean %':>8} "
      f"{'#components':>12}")
print("-" * 50)

for col, pct in enumerate(THRESHOLD_PERCENTILES, start=1):
    threshold = float(np.nanpercentile(shifts, pct))
    raw_flat  = shifts > threshold
    raw_map   = bp.reshape_to_map(raw_flat.astype(int),
                                  y_vals, x_vals).astype(bool)
    clean_map, n_comp = clean_mask(raw_map)

    raw_pct   = 100 * raw_map.sum()   / raw_map.size
    clean_pct = 100 * clean_map.sum() / clean_map.size
    print(f"{pct:>4} {threshold:>11.3f} {raw_pct:>6.1f}% "
          f"{clean_pct:>7.1f}% {n_comp:>12}")

    axes[0, col].imshow(raw_map, origin="lower", cmap="gray")
    axes[0, col].set_title(f"p={pct}, T={threshold:.2f}\n"
                            f"raw  ({raw_pct:.0f}%)", fontsize=9)

    axes[1, col].imshow(clean_map, origin="lower", cmap="gray")
    axes[1, col].set_title(f"cleaned  ({clean_pct:.0f}%)\n"
                            f"{n_comp} → 1 component", fontsize=9)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(
    f"Threshold sweep — raw (top) vs cleaned (bottom)\n"
    f"opening iter={OPENING_ITERATIONS}, fill_holes={FILL_HOLES}, "
    f"largest-component only",
    fontsize=12,
)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# Automatic threshold selection: best = most lit pixels where the
# mask is still ONE connected blob after erode + dilate.
# ─────────────────────────────────────────────

# Fine sweep of percentiles
FINE_PERCENTILES   = np.arange(20, 96, 1)   # 20, 22, ..., 94
ERODE_ITERATIONS   = 1
DILATE_ITERATIONS  = 1
MAX_BORDER_PIXELS  = 5      # mask can touch image edge with at most this many


def erode_dilate(mask, n_erode=1, n_dilate=1):
    """Erode then dilate (== morphological opening with N iterations each)."""
    out = mask.copy()
    for _ in range(n_erode):
        out = binary_erosion(out)
    for _ in range(n_dilate):
        out = binary_dilation(out)
    return out


def border_touch_count(mask):
    """Number of lit pixels sitting on the outermost row or column."""
    return int(mask[0, :].sum() + mask[-1, :].sum()
               + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())


sweep_pcts = []
sweep_thresholds = []
sweep_pixel_counts = []
sweep_n_components = []
sweep_border = []

for pct in FINE_PERCENTILES:
    threshold = float(np.nanpercentile(shifts, pct))
    raw_flat = shifts > threshold
    raw_map  = bp.reshape_to_map(raw_flat.astype(int),
                                 y_vals, x_vals).astype(bool)
    opened = erode_dilate(raw_map, ERODE_ITERATIONS, DILATE_ITERATIONS)
    _, n_comp = cc_label(opened)
    sweep_pcts.append(int(pct))
    sweep_thresholds.append(threshold)
    sweep_pixel_counts.append(int(opened.sum()))
    sweep_n_components.append(int(n_comp))
    sweep_border.append(border_touch_count(opened))

sweep_pcts         = np.array(sweep_pcts)
sweep_thresholds   = np.array(sweep_thresholds)
sweep_pixel_counts = np.array(sweep_pixel_counts)
sweep_n_components = np.array(sweep_n_components)
sweep_border       = np.array(sweep_border)

# Best = most pixels where:
#   - components == 1 (after erode/dilate)
#   - mask touches image border with <= MAX_BORDER_PIXELS pixels
valid = (sweep_n_components == 1) & (sweep_border <= MAX_BORDER_PIXELS)
if not valid.any():
    print("\n[WARNING] No threshold satisfied BOTH the 1-component test "
          "AND the border-touch limit. Falling back to "
          "single-component-only.")
    valid = sweep_n_components == 1
    if not valid.any():
        print("[WARNING] Still no candidate. Picking fewest components.")
        best_idx = int(np.argmin(sweep_n_components))
    else:
        candidate_counts = np.where(valid, sweep_pixel_counts, -1)
        best_idx = int(np.argmax(candidate_counts))
else:
    candidate_counts = np.where(valid, sweep_pixel_counts, -1)
    best_idx = int(np.argmax(candidate_counts))

best_pct        = sweep_pcts[best_idx]
best_threshold  = sweep_thresholds[best_idx]
best_pixels     = sweep_pixel_counts[best_idx]
best_n_comp     = sweep_n_components[best_idx]
best_border     = sweep_border[best_idx]

print(f"\n=== Automatic best threshold ===")
print(f"  percentile     : {best_pct}")
print(f"  threshold      : {best_threshold:.3f} GHz")
print(f"  pixels         : {best_pixels} "
      f"({100 * best_pixels / shift_map.size:.1f}%)")
print(f"  components     : {best_n_comp}  (after erode + dilate)")
print(f"  border touched : {best_border} pixels  "
      f"(limit = {MAX_BORDER_PIXELS})")


# ── Sweep visualisation with example failure-mode masks ────
from matplotlib.gridspec import GridSpec


def _mask_at(idx):
    """Recompute the post-erode+dilate mask at a given sweep index."""
    threshold = sweep_thresholds[idx]
    raw_flat  = shifts > threshold
    raw_map   = bp.reshape_to_map(raw_flat.astype(int),
                                  y_vals, x_vals).astype(bool)
    return erode_dilate(raw_map, ERODE_ITERATIONS, DILATE_ITERATIONS)


red_pool    = np.where(sweep_n_components > 1)[0]
red_idx     = int(red_pool[len(red_pool) // 2]) if len(red_pool) else None
orange_pool = np.where(
    (sweep_n_components == 1) & (sweep_border > MAX_BORDER_PIXELS)
)[0]
orange_idx  = int(orange_pool[len(orange_pool) // 2]) if len(orange_pool) else None

fig = plt.figure(figsize=(17, 5))
gs = GridSpec(1, 4, width_ratios=[2.6, 1, 1, 1], figure=fig, wspace=0.25)
ax1       = fig.add_subplot(gs[0, 0])
ax_red    = fig.add_subplot(gs[0, 1])
ax_orange = fig.add_subplot(gs[0, 2])
ax_green  = fig.add_subplot(gs[0, 3])

# Pixel count curve (left y-axis), colour-coded by which criteria pass:
#   green  = 1 component AND border-touch within limit (fully valid)
#   orange = 1 component but border-touch too high
#   red    = more than 1 component
colors = np.where(
    (sweep_n_components == 1) & (sweep_border <= MAX_BORDER_PIXELS),
    'tab:green',
    np.where(sweep_n_components == 1, 'tab:orange', 'tab:red'),
)
ax1.plot(sweep_pcts, sweep_pixel_counts, 'k-', linewidth=1.0, alpha=0.4)
ax1.scatter(sweep_pcts, sweep_pixel_counts, c=colors, s=30,
            label=(f'lit pixels  '
                   f'(green=valid, orange=touches border >{MAX_BORDER_PIXELS}px, '
                   f'red=>1 component)'))
ax1.axvline(best_pct, color='blue', linestyle='--', linewidth=1.5,
            label=f'best p={best_pct} (T={best_threshold:.2f} GHz)')
if red_idx is not None:
    ax1.scatter([sweep_pcts[red_idx]], [sweep_pixel_counts[red_idx]],
                marker='o', s=180, facecolors='none',
                edgecolors='tab:red', linewidths=2, label='red example')
if orange_idx is not None:
    ax1.scatter([sweep_pcts[orange_idx]], [sweep_pixel_counts[orange_idx]],
                marker='o', s=180, facecolors='none',
                edgecolors='tab:orange', linewidths=2,
                label='orange example')
ax1.set_xlabel("Threshold percentile")
ax1.set_ylabel("Lit pixels after erode+dilate")
ax1.grid(alpha=0.3)

# Connected-component count on the right y-axis
ax2 = ax1.twinx()
ax2.plot(sweep_pcts, sweep_n_components, 'm-', alpha=0.5, linewidth=1.0)
ax2.set_ylabel("# connected components", color='m')
ax2.tick_params(axis='y', labelcolor='m')

ax1.set_title("Threshold sweep with erode+dilate connectivity check")


def _show_example(ax, idx, colour, word):
    if idx is None:
        ax.text(0.5, 0.5, f"no {word} example",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return
    mask = _mask_at(idx)
    ax.imshow(mask, origin="lower", cmap="gray")
    ax.set_title(
        f"{word.upper()} example\n"
        f"p={sweep_pcts[idx]}, T={sweep_thresholds[idx]:.2f}\n"
        f"px={sweep_pixel_counts[idx]}, "
        f"comp={sweep_n_components[idx]}, "
        f"border={sweep_border[idx]}",
        fontsize=9, color=colour,
    )
    ax.set_xticks([]); ax.set_yticks([])


_show_example(ax_red,    red_idx,    'tab:red',    'red')
_show_example(ax_orange, orange_idx, 'tab:orange', 'orange')
_show_example(ax_green,  best_idx,   'tab:green',  'green')

plt.show()


# ── Show the chosen best mask ─────────────────────
best_raw_flat = shifts > best_threshold
best_raw_map  = bp.reshape_to_map(best_raw_flat.astype(int),
                                  y_vals, x_vals).astype(bool)
best_opened   = erode_dilate(best_raw_map,
                             ERODE_ITERATIONS, DILATE_ITERATIONS)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(shift_map, origin="lower", cmap="inferno")
axes[0].set_title("BFS heatmap")
axes[1].imshow(best_raw_map, origin="lower", cmap="gray")
axes[1].set_title(f"Raw mask (p={best_pct}, T={best_threshold:.2f} GHz)")
axes[2].imshow(best_opened, origin="lower", cmap="gray")
axes[2].set_title(f"After erode+dilate  ({best_pixels} px, "
                  f"{best_n_comp} component)")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Automatically-selected best threshold mask (BFS)",
             fontsize=12)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# Multi-driver comparison: BFS vs every VCA endmember
# Pick the driver whose best mask has the most pixels (among valid ones).
# ─────────────────────────────────────────────

import brillouin_decomposition as bd

N_ENDMEMBERS_FOR_COMPARISON = 3


def auto_orient(signal_2d):
    """Flip the signal so the central region reads HIGHER than the edges.
    Returns (oriented_signal_2d, inverted_bool)."""
    h, w = signal_2d.shape
    centre_mean = np.nanmean(signal_2d[h // 3:2 * h // 3,
                                       w // 3:2 * w // 3])
    edge_vals = np.concatenate([
        signal_2d[:h // 6].ravel(),
        signal_2d[-h // 6:].ravel(),
        signal_2d[:, :w // 6].ravel(),
        signal_2d[:, -w // 6:].ravel(),
    ])
    if centre_mean >= np.nanmean(edge_vals):
        return signal_2d, False
    return -signal_2d, True


def sweep_driver(signal_flat, label):
    """Sweep thresholds on a flat per-pixel signal; return best mask + metrics."""
    sweep = {"pct": [], "threshold": [], "pixels": [],
             "n_comp": [], "border": []}
    for pct in FINE_PERCENTILES:
        threshold = float(np.nanpercentile(signal_flat, pct))
        raw_flat = signal_flat > threshold
        raw_map  = bp.reshape_to_map(raw_flat.astype(int),
                                     y_vals, x_vals).astype(bool)
        opened = erode_dilate(raw_map, ERODE_ITERATIONS, DILATE_ITERATIONS)
        _, n_comp = cc_label(opened)
        sweep["pct"].append(int(pct))
        sweep["threshold"].append(threshold)
        sweep["pixels"].append(int(opened.sum()))
        sweep["n_comp"].append(int(n_comp))
        sweep["border"].append(border_touch_count(opened))
    for k in sweep:
        sweep[k] = np.array(sweep[k])

    valid = ((sweep["n_comp"] == 1) &
             (sweep["border"] <= MAX_BORDER_PIXELS))
    if valid.any():
        candidate_counts = np.where(valid, sweep["pixels"], -1)
        best_i = int(np.argmax(candidate_counts))
        fully_valid = True
    else:
        best_i = int(np.argmin(sweep["n_comp"]))
        fully_valid = False

    threshold = float(sweep["threshold"][best_i])
    raw_flat = signal_flat > threshold
    raw_map  = bp.reshape_to_map(raw_flat.astype(int),
                                 y_vals, x_vals).astype(bool)
    mask = erode_dilate(raw_map, ERODE_ITERATIONS, DILATE_ITERATIONS)

    return {
        "label":      label,
        "threshold":  threshold,
        "pct":        int(sweep["pct"][best_i]),
        "pixels":     int(sweep["pixels"][best_i]),
        "n_comp":     int(sweep["n_comp"][best_i]),
        "border":     int(sweep["border"][best_i]),
        "valid":      fully_valid,
        "mask":       mask,
    }


# Compute VCA endmember abundances on the ORIGINAL (un-smoothed) spectra
print(f"\nRunning VCA with {N_ENDMEMBERS_FOR_COMPARISON} endmembers "
      f"(on original spectra)...")
endmembers, _ = bd.vca(peak_spec, N_ENDMEMBERS_FOR_COMPARISON, seed=42)
abundances = bd.estimate_abundances(peak_spec, endmembers)

# Build driver list — BFS uses the smoothed-fit shifts; EM abundances are
# auto-oriented but NOT further smoothed (since VCA saw the raw spectra).
drivers = [(shifts, "BFS")]
for k in range(N_ENDMEMBERS_FOR_COMPARISON):
    abund_map = bp.reshape_to_map(abundances[:, k], y_vals, x_vals)
    oriented_map, inverted = auto_orient(abund_map)
    label = f"EM{k+1}" + (" (inv)" if inverted else "")
    drivers.append((oriented_map.flatten(), label))

# Sweep each driver
results = [sweep_driver(sig, lbl) for sig, lbl in drivers]


# ── Comparison table ─────────────────────────────
print(f"\n{'driver':<12} {'pct':>4} {'threshold':>11} "
      f"{'pixels':>7} {'n_comp':>7} {'border':>7} {'valid':>6}")
print("-" * 60)
for r in results:
    print(f"{r['label']:<12} {r['pct']:>4} {r['threshold']:>11.3f} "
          f"{r['pixels']:>7} {r['n_comp']:>7} {r['border']:>7} "
          f"{str(r['valid']):>6}")

# Pick the winner: among valid drivers, the one with the most pixels
valid_results = [r for r in results if r["valid"]]
if valid_results:
    winner = max(valid_results, key=lambda r: r["pixels"])
else:
    winner = max(results, key=lambda r: r["pixels"])
    print("\n[WARNING] No driver produced a fully-valid mask; "
          "falling back to the largest invalid mask.")
print(f"\n>>> Recommended driver: {winner['label']} "
      f"(threshold = {winner['threshold']:.3f}, "
      f"{winner['pixels']} pixels)")


# ── Comparison figure ────────────────────────────
n_cols = len(results) + 1   # +1 for the BFS reference heatmap
fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 4))
axes[0].imshow(shift_map, origin="lower", cmap="inferno")
axes[0].set_title("BFS heatmap")
for i, r in enumerate(results, start=1):
    is_winner = (r is winner)
    axes[i].imshow(r["mask"], origin="lower", cmap="gray")
    title = (f"{r['label']}\n"
             f"T={r['threshold']:.2f}, {r['pixels']} px\n"
             f"comp={r['n_comp']}, border={r['border']}")
    color = 'green' if is_winner else 'red' if not r['valid'] else 'black'
    axes[i].set_title(title, fontsize=9, color=color)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Mask driver comparison — green title = recommended, "
             "red = invalid", fontsize=12)
plt.tight_layout()
plt.show()


# ── Combined plot: signal heatmaps (top) + best masks (bottom) ──
# Build the heatmap for each driver in the same order as `results`
signal_maps = []
em_iter = 0
for r in results:
    if r["label"] == "BFS":
        signal_maps.append(("BFS", shift_map, "inferno", "GHz"))
    else:
        # The label may be "EM1" or "EM1 (inv)" — strip suffix for indexing
        abund_2d = bp.reshape_to_map(abundances[:, em_iter],
                                     y_vals, x_vals)
        signal_maps.append((r["label"], abund_2d, "viridis", "abundance"))
        em_iter += 1

n_cols = len(results)
fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7))
if n_cols == 1:
    axes = axes.reshape(2, 1)

# Top row: signal heatmaps
for i, (label, sig_map, cmap, cbar_label) in enumerate(signal_maps):
    im = axes[0, i].imshow(sig_map, origin="lower", cmap=cmap)
    axes[0, i].set_title(f"{label} signal", fontsize=10)
    plt.colorbar(im, ax=axes[0, i], fraction=0.046, label=cbar_label)
    axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

# Bottom row: chosen masks (colour-coded title for winner / invalid)
for i, r in enumerate(results):
    is_winner = (r is winner)
    axes[1, i].imshow(r["mask"], origin="lower", cmap="gray")
    title = (f"T={r['threshold']:.2f}, {r['pixels']} px\n"
             f"comp={r['n_comp']}, border={r['border']}")
    color = 'green' if is_winner else 'red' if not r['valid'] else 'black'
    axes[1, i].set_title(title, fontsize=9, color=color)
    axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

plt.suptitle("Driver signal (top) and best mask (bottom) — "
             "green = recommended, red = invalid",
             fontsize=12)
plt.tight_layout()
plt.show()
