"""
Carboplatin line-scan visualisations.

Reads every CSV under '1.5.26 20x Carboplatin/{control,100um,200um,400um}/'
and produces four figures per chosen signal:

  1. Per-folder grid: one panel per CSV, region shading + edges from the
     shift-based tabletop fit, title = filename, no legend.
  2. Per-folder overlay: every scan in the folder on a single axes, one
     colour per folder, with dotted red vertical lines at each scan's
     fitted tabletop edges.
  3. All four folder scans overlaid on ONE axes, colour-coded.
  4. Per-folder mean interpolated onto a common grid — one line per folder.

The plotted quantity is chosen by SIGNAL_KIND at the bottom of the file:
  'shift'  — Lorentzian-fit Brillouin frequency shift (GHz)
  'simple' — pixel spectral gradient with the [-1, 1] kernel across pixels
  'sobel'  — Sobel-Feldman kernel across pixels
  'lap'    — Laplacian kernel across pixels
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brillouin_preprocessing as bp
import brillouin_fitting as bf


BASE = "1.5.26 20x Carboplatin"
FOLDERS = ["control", "100um", "200um", "400um"]
FOLDER_COLORS = {
    "control": "green",
    "100um":   "blue",
    "200um":   "orange",
    "400um":   "red",
}

REGION_COLORS = {"outside": "#dddddd", "edge": "#ffd9b3", "center": "#b3e6ff"}

SIGNAL_LABELS = {
    "shift":     "Brillouin shift (GHz)",
    "linewidth": "Brillouin linewidth (GHz)",
    "simple":    "|spectral gradient| — kernel [-1, 1]",
    "sobel":     "|spectral gradient| — Sobel [1, 2, 0, -2, -1]",
    "lap":       "|spectral gradient| — Laplacian [1, -2, 1]",
}

TABLETOP_SIGNALS = {"shift", "linewidth"}


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def _split_indices_by_pass(positions):
    n = len(positions)
    if n == 0:
        return []
    starts = [0]
    direction = 0
    for i in range(1, n):
        step = positions[i] - positions[i - 1]
        if direction == 0:
            if step != 0:
                direction = 1 if step > 0 else -1
            continue
        if direction * step < 0:
            starts.append(i)
            direction = 0
    starts.append(n)
    return [np.arange(starts[k], starts[k + 1])
            for k in range(len(starts) - 1)]


def _load_line_scan_folder_robust(folder_path):
    out = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".csv"):
            continue
        full = os.path.join(folder_path, fname)
        df = pd.read_csv(full)
        lengths = df["Raw Data"].apply(lambda s: len(s.split(","))).to_numpy()
        unique_lens = sorted(set(lengths))
        if len(unique_lens) > 1:
            print(f"  [info] {fname}: mixed Raw Data lengths {unique_lens} "
                  f"— grouping by length")
        axis = bp._detect_scan_axis(df)
        for L in unique_lens:
            sub = df.loc[lengths == L].reset_index(drop=True)
            positions_all = sub[axis].to_numpy()
            passes = _split_indices_by_pass(positions_all)
            if len(passes) > 1:
                print(f"  [info] {fname} (len={L}): {len(passes)} sweeps "
                      f"detected — splitting into {len(passes)} scans")
            for idx in passes:
                if len(idx) < 10:
                    print(f"  [info] {fname} (len={L}): skipping sweep with "
                          f"only {len(idx)} pixel(s)")
                    continue
                sub_pass = sub.iloc[idx].reset_index(drop=True)
                spectra = np.vstack(
                    sub_pass["Raw Data"].apply(
                        lambda s: np.array([float(v) for v in s.split(",")]))
                )
                start = sub_pass["startFrequency"].iloc[0]
                end   = sub_pass["endFrequency"].iloc[0]
                freq  = np.linspace(start, end, spectra.shape[1])
                positions = sub_pass[axis].to_numpy()
                out.append({
                    "spectra":   spectra,
                    "freq":      freq,
                    "positions": positions,
                    "filename":  fname,
                })
    return out


# ─────────────────────────────────────────────
# Per-scan analysis: shift + spectral-gradient kernels
# ─────────────────────────────────────────────

def spectral_gradient_per_pixel(spectra, kind):
    """Apply a 1D kernel ACROSS pixels; return |summed response| per pixel."""
    n = spectra.shape[0]
    out = np.full(n, np.nan)
    if kind == "simple":       # [-1, 1] forward difference
        for i in range(n - 1):
            out[i] = np.sum(np.abs(spectra[i + 1] - spectra[i]))
    elif kind == "lap":        # [1, -2, 1]
        for i in range(1, n - 1):
            r = spectra[i - 1] - 2.0 * spectra[i] + spectra[i + 1]
            out[i] = np.sum(np.abs(r))
    elif kind == "sobel":      # [1, 2, 0, -2, -1]
        for i in range(2, n - 2):
            r = (spectra[i - 2] + 2.0 * spectra[i - 1]
                 - 2.0 * spectra[i + 1] - spectra[i + 2])
            out[i] = np.sum(np.abs(r))
    else:
        raise ValueError(f"unknown gradient kind: {kind}")
    return out


def load_and_analyse(folder_path):
    """
    For every scan: fit Brillouin shift, compute pixel gradient for each
    kernel kind, fit a tabletop to the shift for region shading + edges.
    """
    scans = _load_line_scan_folder_robust(folder_path)
    scans = bp.sum_normalize_line_scans(
        bp.fold_line_scans(scans, max_shift_ghz=8))
    results = []
    for s in scans:
        _, params = bf.fit_all_spectra(s["shift_freq"], s["folded_spectra_norm"])
        _, shifts, linewidths, _ = bf.extract_fit_parameters(params)
        spec_norm = s["folded_spectra_norm"]
        try:
            tab_params, tab_curve = bf.fit_tabletop(s["positions"], shifts)
            region = bf.tabletop_regions(s["positions"], tab_params, edge_pad=2.0)
        except Exception:
            tab_params, tab_curve, region = None, None, None
        results.append({
            "scan":       s,
            "shift":      shifts,
            "linewidth":  linewidths,
            "simple":     spectral_gradient_per_pixel(spec_norm, "simple"),
            "sobel":      spectral_gradient_per_pixel(spec_norm, "sobel"),
            "lap":        spectral_gradient_per_pixel(spec_norm, "lap"),
            "tab_params": tab_params,
            "tab_curve":  tab_curve,
            "region":     region,
        })
    return results


# ─────────────────────────────────────────────
# Region shading helpers (from shift-based tabletop fit)
# ─────────────────────────────────────────────

def _runs(mask):
    in_run, start = False, 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            yield start, i
    if in_run:
        yield start, len(mask)


def shade_regions(ax, positions, region):
    if region is None:
        return
    for r, color in REGION_COLORS.items():
        for start, end in _runs(region == r):
            ax.axvspan(positions[start],
                       positions[min(end, len(positions) - 1)],
                       color=color, alpha=0.5, zorder=0)


def _center_slice(r, signal_key):
    """Return (positions, signal) restricted to the shift-tabletop centre.
    If no region info is available, returns the full arrays."""
    positions = np.asarray(r["scan"]["positions"], dtype=float)
    sig = np.asarray(r[signal_key], dtype=float)
    region = r["region"]
    if region is None:
        return positions, sig
    mask = region == "center"
    if not mask.any():
        return positions, sig
    return positions[mask], sig[mask]


# ─────────────────────────────────────────────
# 1) Per-folder grid
# ─────────────────────────────────────────────

def plot_folder_grid(results, folder_label, signal_key, ncols=3):
    ylabel = SIGNAL_LABELS[signal_key]
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.0, nrows * 3.2),
                             squeeze=False)
    axes = axes.flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        positions = r["scan"]["positions"]
        if signal_key in TABLETOP_SIGNALS:
            shade_regions(ax, positions, r["region"])
        ax.plot(positions, r[signal_key], 'k.', markersize=4)
        if signal_key in TABLETOP_SIGNALS and r["tab_params"] is not None:
            if signal_key == "shift" and r["tab_curve"] is not None:
                ax.plot(positions, r["tab_curve"], 'r-', linewidth=1.8)
            ax.axvline(r["tab_params"]["x_left"],
                       color='r', linestyle='--', alpha=0.5)
            ax.axvline(r["tab_params"]["x_right"],
                       color='r', linestyle='--', alpha=0.5)
        ax.set_title(r["scan"]["filename"].rsplit(".csv", 1)[0], fontsize=10)
        ax.set_xlabel("X position (um)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Carboplatin {folder_label}", fontsize=13)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 2) Per-folder overlay
# ─────────────────────────────────────────────

def plot_folder_overlay(results, folder_label, color, signal_key,
                        ax=None, show_edges=True, show_title=True):
    ylabel = SIGNAL_LABELS[signal_key]
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    for r in results:
        positions, sig = _center_slice(r, signal_key)
        ax.plot(positions, sig, 'o-', color=color,
                markersize=4, linewidth=1.0, alpha=0.85)
        if (show_edges and signal_key in TABLETOP_SIGNALS
                and r["tab_params"] is not None):
            ax.axvline(r["tab_params"]["x_left"],
                       color='red', linestyle=':', alpha=0.7, linewidth=1.0)
            ax.axvline(r["tab_params"]["x_right"],
                       color='red', linestyle=':', alpha=0.7, linewidth=1.0)
    if show_title:
        ax.set_title(f"Carboplatin {folder_label} (n={len(results)})")
    ax.set_xlabel("X position (um)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if standalone:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────
# 3) All folders overlaid on ONE axes
# ─────────────────────────────────────────────

def plot_all_stacked(all_results, signal_key):
    ylabel = SIGNAL_LABELS[signal_key]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for folder in FOLDERS:
        color = FOLDER_COLORS[folder]
        results = all_results[folder]
        for i, r in enumerate(results):
            positions, sig = _center_slice(r, signal_key)
            ax.plot(positions, sig, 'o-', color=color,
                    markersize=4, linewidth=1.0, alpha=0.75,
                    label=folder if i == 0 else None)
    ax.set_title(f"Carboplatin dose series — {ylabel}")
    ax.set_xlabel("X position (um)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(title="dose", loc="best")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 4) Per-folder mean, one line per folder
# ─────────────────────────────────────────────

def _folder_mean(results, signal_key, n_grid=100):
    if not results:
        return None, None
    sliced = [_center_slice(r, signal_key) for r in results]
    sliced = [(p, s) for p, s in sliced if len(p)]
    if not sliced:
        return None, None
    lo = min(float(np.min(p)) for p, _ in sliced)
    hi = max(float(np.max(p)) for p, _ in sliced)
    grid = np.linspace(lo, hi, n_grid)
    stacked = []
    for pos, sig in sliced:
        order = np.argsort(pos)
        pos, sig = pos[order], sig[order]
        finite = np.isfinite(sig)
        if finite.sum() < 2:
            continue
        stacked.append(np.interp(grid, pos[finite], sig[finite],
                                 left=np.nan, right=np.nan))
    if not stacked:
        return grid, None
    return grid, np.nanmean(np.vstack(stacked), axis=0)


def plot_folder_means(all_results, signal_key):
    ylabel = SIGNAL_LABELS[signal_key]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for folder in FOLDERS:
        grid, mean = _folder_mean(all_results[folder], signal_key)
        if mean is None:
            continue
        ax.plot(grid, mean, 'o-', color=FOLDER_COLORS[folder],
                markersize=4, linewidth=1.5, alpha=0.9,
                label=f"{folder} (n={len(all_results[folder])})")
    ax.set_title(f"Carboplatin dose series — per-folder mean {ylabel}")
    ax.set_xlabel("X position (um)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(title="dose", loc="best")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def make_all_plots(all_results, signal_key):
    for folder in FOLDERS:
        plot_folder_grid(all_results[folder], folder, signal_key, ncols=3)
    for folder in FOLDERS:
        plot_folder_overlay(all_results[folder], folder,
                            FOLDER_COLORS[folder], signal_key)
    plot_all_stacked(all_results, signal_key)
    plot_folder_means(all_results, signal_key)


if __name__ == "__main__":
    # Change to 'shift', 'simple', 'sobel', or 'lap'.
    SIGNAL_KIND = "sobel"

    all_results = {}
    for folder in FOLDERS:
        path = os.path.join(BASE, folder)
        print(f"Loading {path} ...")
        all_results[folder] = load_and_analyse(path)
        print(f"  {len(all_results[folder])} scans")

    make_all_plots(all_results, SIGNAL_KIND)
