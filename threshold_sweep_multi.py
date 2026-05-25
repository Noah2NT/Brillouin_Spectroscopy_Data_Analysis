"""
Run the threshold-sweep + multi-driver mask comparison on several
2D Brillouin datasets and compare the results side by side.

For each dataset:
  - Lorentzian-fit every pixel to get the Brillouin shift.
  - Run VCA with 3 endmembers.
  - Sweep thresholds on BFS + each EM abundance, picking the best
    mask (most pixels, single connected component, low border touch).
  - Show the sweep curve with red/orange/green example masks.
  - Show the driver-signal-vs-best-mask comparison grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import (binary_erosion, binary_dilation,
                           gaussian_filter, label as cc_label)

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd


# ── Datasets to compare ─────────────────────────────
DATASETS = [
    ("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv",
     "Cell_Col Healthy"),
    ("XY_scanning_40um_60um_step1um_10s_fixed.csv",
     "XY_scanning_40um_60um"),
    ("Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv",
     "Hydrogel"),
]

# ── Pipeline config (same as threshold_sweep.py) ───
PEAK_BIN_START               = 305
PEAK_BIN_END                 = 405
MAX_SHIFT_GHZ                = 8.0
FINE_PERCENTILES             = np.arange(20, 96, 1)
ERODE_ITERATIONS             = 1
DILATE_ITERATIONS            = 1
MAX_BORDER_PIXELS            = 15
N_ENDMEMBERS_FOR_COMPARISON  = 3
SMOOTH_SIGMA                 = 1.0    # Gaussian spatial smoothing before masking


# ── Shared helpers ─────────────────────────────────
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


def smooth_flat(signal_flat, x_vals, y_vals, sigma=SMOOTH_SIGMA):
    """Reshape flat per-pixel signal to 2D, Gaussian-smooth, flatten back."""
    if sigma is None or sigma <= 0:
        return signal_flat
    map_2d = bp.reshape_to_map(signal_flat, y_vals, x_vals)
    smoothed = gaussian_filter(map_2d, sigma=sigma)
    return smoothed.flatten()


def spatial_smooth_spectra(peak_spec, x_vals, y_vals, sigma):
    """Gaussian-smooth each frequency bin across the 2D spatial grid.

    Returns a new spectra array where each pixel's spectrum has been
    averaged with its spatial neighbours (per frequency bin), so the
    subsequent Lorentzian fit sees a denoised input.
    """
    if sigma is None or sigma <= 0:
        return peak_spec
    ny, nx = len(y_vals), len(x_vals)
    B = peak_spec.shape[1]
    cube = peak_spec.reshape(ny, nx, B)
    out = np.empty_like(cube)
    for b in range(B):
        out[:, :, b] = gaussian_filter(cube[:, :, b], sigma=sigma)
    return out.reshape(ny * nx, B)


def sweep_driver(signal_flat, x_vals, y_vals, label):
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
        "sweep":      sweep,
        "best_idx":   best_i,
    }


# ── Per-dataset pipeline ────────────────────────────
def run_pipeline(data_path, dataset_label):
    print(f"\n{'=' * 70}\n>>> {dataset_label}  ({data_path})\n{'=' * 70}")

    spectra, freq, df = bp.load_brillouin_csv(data_path)
    x_vals, y_vals    = bp.build_coordinate_arrays(df)
    peak_spec, peak_freq = bp.extract_frequency_window(
        spectra, freq, PEAK_BIN_START, PEAK_BIN_END,
    )
    keep = peak_freq <= MAX_SHIFT_GHZ
    peak_spec = peak_spec[:, keep]
    peak_freq = peak_freq[keep]
    print(f"  loaded {peak_spec.shape[0]} pixel spectra")

    # Spatially smooth the spectra BEFORE Lorentzian fitting so the BFS
    # map starts out clean. VCA still runs on the ORIGINAL spectra.
    peak_spec_smoothed = spatial_smooth_spectra(
        peak_spec, x_vals, y_vals, sigma=SMOOTH_SIGMA,
    )

    # Lorentzian fit → BFS per pixel (on smoothed spectra)
    _, fit_params = bf.fit_all_spectra(peak_freq, peak_spec_smoothed)
    _, shifts, _, _ = bf.extract_fit_parameters(fit_params)
    shifts = np.clip(shifts, peak_freq.min(), peak_freq.max())
    shift_map = bp.reshape_to_map(shifts, y_vals, x_vals)

    # VCA on ORIGINAL (un-smoothed) spectra
    print(f"  running VCA k={N_ENDMEMBERS_FOR_COMPARISON} (original spectra)")
    endmembers, _ = bd.vca(peak_spec, N_ENDMEMBERS_FOR_COMPARISON, seed=42)
    abundances = bd.estimate_abundances(peak_spec, endmembers)

    # Show what the spectrum-level smoothing actually did — one mid-image
    # pixel, raw vs smoothed Brillouin curve overlay.
    demo_idx = peak_spec.shape[0] // 2
    fig_demo, ax = plt.subplots(figsize=(7, 4))
    ax.plot(peak_freq, peak_spec[demo_idx], 'b-', alpha=0.5,
            linewidth=1.0, label='raw spectrum')
    ax.plot(peak_freq, peak_spec_smoothed[demo_idx], 'r-',
            linewidth=1.6, label=f'spatially smoothed (σ={SMOOTH_SIGMA})')
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{dataset_label} — pixel {demo_idx}: "
                 f"raw vs spatially-smoothed spectrum")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Drivers — BFS uses the smoothed-fit shifts; VCA abundances stay raw
    # (no further smoothing applied to abundance maps).
    drivers = [(shifts, "BFS")]
    for k in range(N_ENDMEMBERS_FOR_COMPARISON):
        abund_map = bp.reshape_to_map(abundances[:, k], y_vals, x_vals)
        oriented_map, inverted = auto_orient(abund_map)
        lbl = f"EM{k+1}" + (" (inv)" if inverted else "")
        drivers.append((oriented_map.flatten(), lbl))

    # Local alias for the BFS sweep visualisation later
    shifts_s = shifts

    # Sweep each driver
    results = [sweep_driver(sig, x_vals, y_vals, lbl) for sig, lbl in drivers]

    # Print comparison table
    print(f"\n  {'driver':<12} {'pct':>4} {'threshold':>11} "
          f"{'pixels':>7} {'comp':>5} {'border':>7} {'valid':>6}")
    print("  " + "-" * 58)
    for r in results:
        print(f"  {r['label']:<12} {r['pct']:>4} {r['threshold']:>11.3f} "
              f"{r['pixels']:>7} {r['n_comp']:>5} {r['border']:>7} "
              f"{str(r['valid']):>6}")

    valid_results = [r for r in results if r["valid"]]
    if valid_results:
        winner = max(valid_results, key=lambda r: r["pixels"])
    else:
        winner = max(results, key=lambda r: r["pixels"])
        print("  [WARNING] No driver was fully valid; "
              "falling back to largest invalid.")
    print(f"\n  >>> Recommended driver: {winner['label']} "
          f"(T={winner['threshold']:.3f}, {winner['pixels']} pixels)")

    # ── Figure 1: BFS sweep curve + red/orange/green examples ──
    bfs_result = results[0]
    sweep = bfs_result["sweep"]
    best_i = bfs_result["best_idx"]
    red_pool    = np.where(sweep["n_comp"] > 1)[0]
    red_idx     = int(red_pool[len(red_pool) // 2]) if len(red_pool) else None
    orange_pool = np.where(
        (sweep["n_comp"] == 1) & (sweep["border"] > MAX_BORDER_PIXELS)
    )[0]
    orange_idx  = int(orange_pool[len(orange_pool) // 2]) if len(orange_pool) else None

    fig = plt.figure(figsize=(17, 5))
    gs = GridSpec(1, 4, width_ratios=[2.6, 1, 1, 1], figure=fig, wspace=0.25)
    ax1       = fig.add_subplot(gs[0, 0])
    ax_red    = fig.add_subplot(gs[0, 1])
    ax_orange = fig.add_subplot(gs[0, 2])
    ax_green  = fig.add_subplot(gs[0, 3])

    colors = np.where(
        (sweep["n_comp"] == 1) & (sweep["border"] <= MAX_BORDER_PIXELS),
        'tab:green',
        np.where(sweep["n_comp"] == 1, 'tab:orange', 'tab:red'),
    )
    ax1.plot(sweep["pct"], sweep["pixels"], 'k-', linewidth=1.0, alpha=0.4)
    ax1.scatter(sweep["pct"], sweep["pixels"], c=colors, s=30,
                label='lit pixels (green/orange/red)')
    ax1.axvline(sweep["pct"][best_i], color='blue', linestyle='--',
                linewidth=1.5,
                label=f"best p={sweep['pct'][best_i]} "
                      f"(T={sweep['threshold'][best_i]:.2f} GHz)")
    if red_idx is not None:
        ax1.scatter([sweep["pct"][red_idx]], [sweep["pixels"][red_idx]],
                    marker='o', s=180, facecolors='none',
                    edgecolors='tab:red', linewidths=2)
    if orange_idx is not None:
        ax1.scatter([sweep["pct"][orange_idx]],
                    [sweep["pixels"][orange_idx]],
                    marker='o', s=180, facecolors='none',
                    edgecolors='tab:orange', linewidths=2)
    ax1.set_xlabel("Threshold percentile")
    ax1.set_ylabel("Lit pixels after erode+dilate")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(sweep["pct"], sweep["n_comp"], 'm-', alpha=0.5, linewidth=1.0)
    ax2.set_ylabel("# connected components", color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    ax1.legend(loc='lower left', fontsize=8)
    ax1.set_title(f"{dataset_label} — BFS sweep")

    def _mask_at(idx):
        threshold = sweep["threshold"][idx]
        raw_flat  = shifts_s > threshold    # smoothed BFS
        raw_map   = bp.reshape_to_map(raw_flat.astype(int),
                                      y_vals, x_vals).astype(bool)
        return erode_dilate(raw_map, ERODE_ITERATIONS, DILATE_ITERATIONS)

    def _show(ax, idx, colour, word):
        if idx is None:
            ax.text(0.5, 0.5, f"no {word} example",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            return
        mask = _mask_at(idx)
        ax.imshow(mask, origin="lower", cmap="gray")
        ax.set_title(f"{word.upper()}\np={sweep['pct'][idx]}, "
                     f"T={sweep['threshold'][idx]:.2f}\n"
                     f"px={sweep['pixels'][idx]}, "
                     f"comp={sweep['n_comp'][idx]}, "
                     f"border={sweep['border'][idx]}",
                     fontsize=9, color=colour)
        ax.set_xticks([]); ax.set_yticks([])

    _show(ax_red,    red_idx,    'tab:red',    'red')
    _show(ax_orange, orange_idx, 'tab:orange', 'orange')
    _show(ax_green,  best_i,     'tab:green',  'green')
    plt.suptitle(f"{dataset_label} — sweep + failure-mode examples",
                 fontsize=12)
    plt.show()

    # ── Figure 2: driver-signal + best-mask comparison ──
    signal_maps = []
    em_iter = 0
    for r in results:
        if r["label"] == "BFS":
            signal_maps.append(("BFS", shift_map, "inferno", "GHz"))
        else:
            abund_2d = bp.reshape_to_map(abundances[:, em_iter],
                                         y_vals, x_vals)
            signal_maps.append((r["label"], abund_2d, "viridis",
                                "abundance"))
            em_iter += 1

    n_cols = len(results)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    for i, (lbl, sig_map, cmap, cbar_label) in enumerate(signal_maps):
        im = axes[0, i].imshow(sig_map, origin="lower", cmap=cmap)
        axes[0, i].set_title(f"{lbl} signal", fontsize=10)
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, label=cbar_label)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
    for i, r in enumerate(results):
        is_winner = (r is winner)
        axes[1, i].imshow(r["mask"], origin="lower", cmap="gray")
        title = (f"T={r['threshold']:.2f}, {r['pixels']} px\n"
                 f"comp={r['n_comp']}, border={r['border']}")
        color = 'green' if is_winner else 'red' if not r['valid'] else 'black'
        axes[1, i].set_title(title, fontsize=9, color=color)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
    plt.suptitle(f"{dataset_label} — driver signal vs best mask",
                 fontsize=12)
    plt.tight_layout()
    plt.show()

    return {"dataset": dataset_label, "winner": winner,
            "results": results}


# ── Run all datasets ────────────────────────────────
summary = []
for path, label in DATASETS:
    try:
        out = run_pipeline(path, label)
        summary.append(out)
    except FileNotFoundError:
        print(f"\n[SKIP] {label}: file not found ({path})")
    except Exception as e:
        print(f"\n[ERROR] {label}: {e!r}")


# ── Cross-dataset summary ──────────────────────────
print(f"\n{'=' * 70}\n>>> Cross-dataset summary\n{'=' * 70}")
print(f"{'dataset':<22} {'winner':<12} {'T':>8} {'pixels':>7} "
      f"{'comp':>5} {'border':>7}")
print("-" * 65)
for s in summary:
    w = s["winner"]
    print(f"{s['dataset']:<22} {w['label']:<12} "
          f"{w['threshold']:>8.3f} {w['pixels']:>7} "
          f"{w['n_comp']:>5} {w['border']:>7}")
